import asyncio
import time
from functools import partial
from typing import (AsyncIterator, Callable, Dict, Iterable, List, Optional,
                    Set, Tuple, Type, Union)
import os

from transformers import PreTrainedTokenizer

import vllm.envs as envs
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
# from vllm.executor.ray_utils import initialize_ray_cluster, ray
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest, MultiModalData, SamplerOutput
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        
    def __str__(self):
        return f"AsyncStream _queue: {self._queue}"

    def __repr__(self):
        return self.__str__()

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        if request_output:
            request_id = request_output.request_id
            print("request_id, request_output", request_id, request_output)
            self._request_streams[request_id].put(request_output)
            logger.debug("In process_request_output, request_output = " + str(request_output))
            if request_output.finished:
                if verbose:
                    logger.info("Finished request %s.", request_id)
                self.abort_request(request_id)

    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))
        
        logger.debug("self._new_requests empty3"+str(self._new_requests.empty()))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        logger.debug("START step_async")
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        
        logger.debug("In step_async, after schedule, seq_group_metadata_list="+str(seq_group_metadata_list))
        logger.debug("In step_async, after schedule, scheduler_outputs="+str(scheduler_outputs))
        logger.debug("In step_async, not scheduler_outputs.is_empty() "+str(not scheduler_outputs.is_empty()))
        
        for seq_group in self.scheduler.running:
            logger.debug("seq_group" + str(seq_group))

        if not scheduler_outputs.is_empty():
            # Execute the model.
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )
            output = await self.model_executor.execute_model_async(
                execute_model_req)
            output = []
            if output:
                print("if output", output)
                request_outputs = self._process_model_outputs(
                output, scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
                # Log stats.
                self.do_log_stats(scheduler_outputs, output)
            else:
                request_outputs = []
            
        else:
            output = []
            request_outputs = self._process_model_outputs(
                output, scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
            # Log stats.
            self.do_log_stats(scheduler_outputs, output)
        
        # Save scheduler outputs and metadata for later use in queues
        self.scheduler_outputs_queue.append(scheduler_outputs)
        self.seq_group_metadata_queue.append(seq_group_metadata_list)
        
        logger.debug("END step_async")
        
        self.scheduler.temp_remove_pp0_seq_groups()
        
        return [request_outputs, not scheduler_outputs.is_empty()]

    async def encode_request_async(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(
                request_id=request_id,
                prompt=prompt,
                lora_request=lora_request)
        return prompt_token_ids

    async def add_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.encode_request_async(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        return self.add_request(request_id,
                                prompt=prompt,
                                prompt_token_ids=prompt_token_ids,
                                sampling_params=sampling_params,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                multi_modal_data=multi_modal_data)

    async def check_health_async(self) -> None:
        self.model_executor.check_health()
        
    async def process_last_pp_logits(self):            
        scheduler_outputs = None
        if self.scheduler_outputs_queue and self.seq_group_metadata_queue:
            scheduler_outputs = self.scheduler_outputs_queue.popleft()
            seq_group_metadata_list = self.seq_group_metadata_queue.popleft()
            
        if scheduler_outputs and not scheduler_outputs.is_empty():  # TODO
            logger.debug("Start process_last_pp_logits")
            from mpi4py import MPI
            
            xft_pipeline_stage = int(os.environ.get('XFT_PIPELINE_STAGE'))
            print(xft_pipeline_stage)
            
            if xft_pipeline_stage > 1:
                comm = MPI.COMM_WORLD  # 初始化一个全局的通讯器
                
                logger.debug("Start recv")
                # logits = comm.recv(source = xft_pipeline_stage-1)  # 从last pp rank的tp rank 0接收数据到recv_tmp_tensor
                # logger.debug(str(help(comm.recv)))
                # logger.debug(str(help(comm.irecv)))
                # logger.debug(str(help(comm.Irecv)))
                # logger.debug(str(help(comm.send)))
                # logger.debug("DONE recv, logits shape = " + str(logits.shape) + " logits dtype = " + str(logits.dtype))  # 为啥是torch.float32的？
                
                # 改用非堵塞的mpi recv
                # buf = torch.empty((1, 32000), dtype=torch.float32)
                # logger.debug("Receiving tensor. Allocated buffer with shape: " + str(buf.shape) + " and dtype: " + str(buf.dtype))
                #                     # while not request.Test():
                    #     await asyncio.sleep(0.01)
                
                try:
                    logger.debug("Start recv")
                    # import torch
                    # import numpy as np
                    # buf = np.empty((1, 32000), dtype=np.float32)
                    # request = comm.irecv(buf, source=xft_pipeline_stage - 1)
                    # logits = request.wait()
                    # logger.debug("DONE recv")
                    # logger.debug(f"Received data: {logits}")
                    
                    # buf = np.empty(5, dtype=np.float32)
                    # data = comm.recv(source = xft_pipeline_stage-1)
                    
                    request = comm.irecv(source=xft_pipeline_stage - 1)
                    while not request.Test(): # 检查是否完成接收
                        await asyncio.sleep(0) # 未完成接收则执行其他协程任务
                    
                    data = request.wait()
                    logger.debug(f"Received data: {data}")

                except Exception as e:
                    logger.debug("Error in mpi irecv: "+ str(e))
                
                
                # TODO 先不sampling，随便搞个output来，但是要贴batch size
                import pickle
                with open('/home/johnson/qiuyu/vllm-xft/debug/pp3_output.pkl', 'rb') as file:
                    output = [pickle.load(file)]
                print(output)
                
                if output:
                    request_outputs = self._process_model_outputs(
                    output, scheduler_outputs.scheduled_seq_groups,
                    scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
                    
                    for request_output in request_outputs:
                        await self.output_queue.put(request_output)
                


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker
        
        self.has_requests_in_pp3_progress = False

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        if True:
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutorAsync
            executor_class = NeuronExecutorAsync
        elif engine_config.device_config.device_type == "cpu":
            assert not engine_config.parallel_config.worker_use_ray, (
                "Ray is not supported with the CPU backend.")
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        else:
            assert engine_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        # Create the async LLM engine.
        engine = cls(
            engine_config.parallel_config.worker_use_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )
        return engine

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_tokenizer(self) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.engine.get_tokenizer.remote()  # type: ignore
        else:
            return self.engine.get_tokenizer()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        engine_class = self._engine_class
        # if not self.engine_use_ray:
        # elif self.worker_use_ray:
        #     engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        # else:
        #     # FIXME(woosuk): This is a bit hacky. Be careful when changing the
        #     # order of the arguments.
        #     cache_config = kwargs["cache_config"]
        #     parallel_config = kwargs["parallel_config"]
        #     if parallel_config.tensor_parallel_size == 1:
        #         num_gpus = cache_config.gpu_memory_utilization
        #     else:
        #         num_gpus = 1
        #     engine_class = ray.remote(num_gpus=num_gpus)(
        #         self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())
        
        logger.debug("In engine_step, get_new_and_finished_requests, new_requests = "+str(new_requests))

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.engine.add_request.remote(  # type: ignore
                        **new_request)
                else:
                    await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()  # type: ignore
        else:
            [request_outputs,scheduler_outputs_status] = await self.engine.step_async()
        
        logger.debug("engine step, scheduler_outputs="+str(scheduler_outputs_status))    
        # Push the outputs to the async queue
        for request_output in request_outputs:
            await self.engine.output_queue.put(request_output)

        return scheduler_outputs_status

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)  # type: ignore
        else:
            self.engine.abort_request(request_ids)
            
    async def process_last_pp_outputs(self):
        while True:
            if self.engine.output_queue.empty():
                self.has_requests_in_pp3_progress = False
                break
            request_output = await self.engine.output_queue.get()
            self.has_requests_in_pp3_progress = True
            # 处理 request_output
            logger.debug("Start process_last_pp_outputs")
            self._request_tracker.process_request_output(request_output, verbose=self.log_requests)
            logger.debug("Finish process_last_pp_outputs, self.has_requests_in_pp3_progress = " + str(self.has_requests_in_pp3_progress))
            self.engine.scheduler.put_back_pp3_seq_groups(request_output.request_id)
            logger.debug("Finish self.engine.scheduler.put_back_pp3_seq_groups " + str(request_output.request_id))
            self._request_tracker.new_requests_event.set()
            logger.debug("new_requests_event.set()")
            
    async def continuous_process_last_pp_logits(self):
        while True:
            try:
                await self.engine.process_last_pp_logits()
            except Exception as e:
                logger.debug("Error in process_last_pp_logits: "+ str(e))
                continue
            await asyncio.sleep(0)

    async def continuous_process_last_pp_outputs(self):
        while True:
            await self.process_last_pp_outputs()
            await asyncio.sleep(0)

    async def run_engine_loop(self): 
        # 启动持续任务
        asyncio.create_task(self.continuous_process_last_pp_logits())
        asyncio.create_task(self.continuous_process_last_pp_outputs())
        
        has_requests_in_pp0_progress = False
        while True:           
            logger.debug("self.has_requests_in_pp3_progress " + str(self.has_requests_in_pp3_progress) + " has_requests_in_pp0_progress " + str(has_requests_in_pp0_progress))
            logger.debug("self._request_tracker._new_requests"+str(self._request_tracker._new_requests))
            logger.debug("self._request_tracker._new_requests empty 1"+str(self._request_tracker._new_requests.empty()))
            # if not self.has_requests_in_pp3_progress and not has_requests_in_pp0_progress:
            if not has_requests_in_pp0_progress:
                logger.debug("self._request_tracker._new_requests empty 2"+str(self._request_tracker._new_requests.empty()))
                logger.debug("Waiting for new requests...")
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                has_requests_in_pp0_progress = await asyncio.wait_for(self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
                # await asyncio.wait_for(self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
                logger.debug("Finish asyncio.wait_for self.engine_step(), has_requests_in_pp0_progress= "+str(has_requests_in_pp0_progress))
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)
            

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(
                "Received request %s: prompt: %r, "
                "sampling_params: %s, prompt_token_ids: %s, "
                "lora_request: %s.", request_id, shortened_prompt,
                sampling_params, shortened_token_ids, lora_request)

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        if self.engine_use_ray:
            prompt_token_ids = await (
                self.engine.encode_request_async.remote(  # type: ignore
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    lora_request=lora_request))
        else:
            prompt_token_ids = await self.engine.encode_request_async(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                lora_request=lora_request)

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
        )

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data per request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        # Preprocess the request.
        arrival_time = time.time()

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()  # type: ignore
        else:
            return self.engine.get_model_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_decoding_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_decoding_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote(  # type: ignore
                scheduler_outputs, model_output)
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()  # type: ignore
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)
