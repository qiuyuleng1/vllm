"""Utils for model executor."""
import random
import importlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from vllm.config import DeviceConfig, ModelConfig


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> torch.nn.Module:
    model_loader_module = "model_loader"
    imported_model_loader = importlib.import_module(
        f"vllm.model_executor.{model_loader_module}")
    get_model_fn = imported_model_loader.get_model
    return get_model_fn(model_config, device_config, **kwargs)
