import importlib
from typing import Dict, List, Optional

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import (get_ip, get_open_port, get_distributed_init_method,
                        make_async)

logger = init_logger(__name__)

# A map between the device type (in device config) to its worker module.
DEVICE_TO_WORKER_MODULE_MAP = {
    # "cuda": "vllm.worker.worker",
    # "neuron": "vllm.worker.neuron_worker",
    "cpu": "vllm.worker.worker",
}


class CPUExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config = None,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = None
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        # Instantiate the worker and load the model to CPU.
        self._init_worker()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

    def _dispatch_worker(self):
        worker_module = DEVICE_TO_WORKER_MODULE_MAP[
            self.device_config.device_type]
        imported_worker = importlib.import_module(worker_module)
        Worker = imported_worker.Worker
        return Worker

    def _init_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        Worker = self._dispatch_worker()

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=None,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker.init_model()
        self.driver_worker.load_model()

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.profile_num_available_blocks(
                block_size=self.cache_config.block_size,
                gpu_memory_utilization=self.cache_config.
                gpu_memory_utilization,
                cpu_swap_space=self.cache_config.swap_space_bytes,
                cache_dtype=self.cache_config.cache_dtype,
            ))

        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self.driver_worker.init_cache_engine(cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self.driver_worker.warm_up_model()

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output
    
    def add_lora(self, lora_request) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> List[int]:
        raise NotImplementedError

    def check_health(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return


class CPUExecutorAsync(CPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy)
        return output

    async def check_health_async(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return
