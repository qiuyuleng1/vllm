"""Utilities for selecting and loading models."""
import torch
import torch.nn as nn

from vllm.config import DeviceConfig, ModelConfig

import xfastertransformer



# def _get_model_architecture(model_config: ModelConfig) -> Type[nn.Module]:
#     architectures = getattr(model_config.hf_config, "architectures", [])
#     if (model_config.quantization is not None
#             and "MixtralForCausalLM" in architectures):
#         architectures = ["QuantMixtralForCausalLM"]

#     for arch in architectures:
#         model_cls = ModelRegistry.load_model_cls(arch)
#         if model_cls is not None:
#             return model_cls
#     raise ValueError(
#         f"Model architectures {architectures} are not supported for now. "
#         f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, device_config: DeviceConfig, **kwargs) -> nn.Module:
    lora_config = None
    # model_class = _get_model_architecture(model_config)
    linear_method = None

    model = xFTModel(model_config.hf_config)
    
    model.load_weights(model_config.model, model_config.xft_dtype)
    return model


class xFTModel():

    def __init__(
        self,
        config = None,
        linear_method = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = None

    def set_input(self, input_ids: torch.Tensor):
        self.model.input(input_ids)
        
    def set_config(self, *args, **kwargs):
        self.model.config(**kwargs)
    
    def generate(self) -> torch.Tensor:
        next_tokens = self.model.forward()
        return next_tokens
        # while not model.is_done():

    def load_weights(self,
                     model_name_or_path: str,
                     xft_dtype:str = 'fp16'):
        self.model=xfastertransformer.AutoModel.from_pretrained(model_name_or_path, dtype = xft_dtype)
        # import os

        # # 获取环境变量并打印
        # for key, value in os.environ.items():
        #     print(f'{key}: {value}')
        # print(f"#######[INFO] Rank is {self.model.rank}###########")
        # if self.model.rank != 0:
        #     # Slave
        #     while True:
        #         self.model.generate()
    
    def is_done(self) -> bool:
        return self.model.is_done()
