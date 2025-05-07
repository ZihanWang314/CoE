# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
A lightweight one-file FSDP SFT Trainer
"""

import os
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, DistributedSampler
from verl.utils.dataset import SFTDataset
from verl.utils.tracking import Tracking
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from verl.utils.distributed import initialize_global_process_group

from coe.trainer.base_trainer import BaseTrainer

# set up all random seeds to be 42
import numpy
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


import hydra

@hydra.main(config_path='config', config_name='train', version_base=None)
def main(config):    
    if config.get("fsdp", True):
        local_rank, rank, world_size = initialize_global_process_group()

        device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type='cuda',
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=('dp', 'sp')
        )
        
        trainer = BaseTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    else:
        raise NotImplementedError
        # below is for debugging in single-process manner
        # trainer = BaseTrainer(config=config)
    
    trainer.fit()


if __name__ == '__main__':
    main()