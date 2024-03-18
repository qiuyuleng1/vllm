import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import get_model, SamplingMetadata

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
    SequenceOutput,
    SequenceGroupOutput,
    Logprob,
)
from vllm.utils import in_wsl, measure_cuda_memory

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = None
        self.is_driver_worker = is_driver_worker

        self.padding_token = (
            self.model_config.hf_config.pad_token_id
            if self.model_config.hf_config.pad_token_id
            else self.model_config.hf_config.eos_token_id
        )

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.block_size = None  # Set after initial profiling.

        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()
        self.kv_cache_dtype = kv_cache_dtype

    def load_model(self) -> None:
        self.model = get_model(self.model_config,
                                self.device_config,
                                lora_config=self.lora_config,
                                parallel_config=self.parallel_config,
                                scheduler_config=self.scheduler_config)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (self.max_context_len_to_capture + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> torch.Tensor:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []

        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)
            input_tokens.append(prompt_tokens)

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len > 0
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_prompt_len,
                                             pad=self.padding_token,
                                             dtype=torch.long,
                                             device=self.device)

        return input_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:

        if seq_group_metadata_list[0].is_prompt:
            # set config and input
            input_tokens = self._prepare_prompt(seq_group_metadata_list)
            self.model.set_input(input_tokens)
            # assume all request has the same sampling_params
            sampling_params = seq_group_metadata_list[0].sampling_params
            max_length=sampling_params.max_tokens + input_tokens.shape[-1]
            num_beams=sampling_params.n
            num_return_sequences=sampling_params.best_of
            length_penalty=sampling_params.length_penalty
            early_stopping=sampling_params.early_stopping
            temperature=sampling_params.temperature
            top_k=sampling_params.top_k
            top_p=sampling_params.top_p
            repetition_penalty=sampling_params.repetition_penalty
            stop_words_ids = sampling_params.stop_token_ids
            do_sample=bool(sampling_params.temperature)

            # print(input_tokens)

            self.model.set_config(
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_words_ids=stop_words_ids,
            )

        # Execute the model.
        output_ids = self.model.generate().view(-1).tolist()
        seq_output_list:List[SequenceGroupOutput]=[]
        for i, id in enumerate(output_ids):
            seq_id = list(seq_group_metadata_list[i].seq_data.keys())[0]
            parent_seq_id = seq_id
            logprob = Logprob(0.9, None)
            seq_output = SequenceOutput(parent_seq_id, id,{id: logprob})
            seq_output_list.append(SequenceGroupOutput([seq_output], None, self.model.is_done()))
        output = SamplerOutput(seq_output_list)
        return output

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    # padding left
    return [pad] * (max_len - len(x)) + x


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)
