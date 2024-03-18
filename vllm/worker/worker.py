"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = None
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        device_config,
                                        lora_config=None,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None


    def init_model(self, cupy_port: Optional[int] = None) -> None:
        if self.device_config.device.type == "cpu":
           pass
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        # init_distributed_environment(self.parallel_config, self.rank,
        #                              cupy_port, self.distributed_init_method)
        # Initialize the model.
        # set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """

        return 0, 0

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        # self.cache_config = cache_config
        # self.gpu_cache = self.cache_engine.gpu_cache
        # self.model_runner.set_block_size(self.cache_engine.block_size)
        pass

    def warm_up_model(self) -> None:
        pass

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        pass

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None
        data = {
            "num_seq_groups": num_seq_groups,
            "blocks_to_swap_in": blocks_to_swap_in,
            "blocks_to_swap_out": blocks_to_swap_out,
            "blocks_to_copy": blocks_to_copy,
        }

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 None)
        return output

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self, block_size: int,
                                   cache_dtype: str) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return 114514

