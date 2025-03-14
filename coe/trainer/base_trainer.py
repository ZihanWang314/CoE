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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn, optim
from tensordict import TensorDict
from verl.utils.dataset import SFTDataset
from torch.distributed.device_mesh import DeviceMesh
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from verl.utils.torch_functional import get_cosine_schedule_with_warmup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

from coe.utils.dataset.base_dataset import BaseDataset
from coe.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from coe.utils.debug.performance import get_gpu_memory_usage
import wandb
import time

# temporary for debugging
from typing import List, Union
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.utils.tracking import Tracking



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


class BaseTrainer(FSDPSFTTrainer):

    def __init__(self, config, device_mesh: DeviceMesh=None, ulysses_device_mesh: DeviceMesh=None, dataset_class = BaseDataset):
        if config.get("fsdp", True):
            super().__init__(config, device_mesh, ulysses_device_mesh, dataset_class)
        else:
            raise NotImplementedError
        
            # below is for debugging in single-process manner
            # self.config = config
            # local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
            # from verl.utils import hf_tokenizer
            # self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
            # if self.config.data.chat_template is not None:
            #     raise ValueError('Apply Chat template from config is not supported yet.')

            # self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)

            # self._build_dataloader()
            # self._build_model_optimizer()

            # # TODO: add checkpoint manager
            # if self.device_mesh.get_rank() == 0:
            #     print(self.config)
        
    def _build_dataset(self):
        config = self.config
        # build dataset
        self.train_dataset = self.dataset_class(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )
        self.val_dataset = self.dataset_class(
            parquet_files=config.data.val_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )

    def _get_model(self, local_model_path, config, trust_remote_code):
        for key, value in self.config.model.override_config.items():
            setattr(config, key, value)
        if self.config.model.from_config:
            model: PreTrainedModel = AutoModelForCausalLM.from_config(config, 
                                                                            torch_dtype=torch.float32,
                                                                            attn_implementation=config._attn_implementation,
                                                                            trust_remote_code=trust_remote_code)
        else:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path, 
                                                                            torch_dtype=torch.float32,
                                                                            attn_implementation=config._attn_implementation,
                                                                            trust_remote_code=trust_remote_code)
            
        print("MODEL TOTAL PARAMS:", sum(p.numel() for p in model.parameters()))
        print(model)
        return model
    
    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        # assert only 1 is valid
        
        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
        # Configure validation interval
        validation_interval = self.config.trainer.validation_interval_steps if hasattr(self.config.trainer, 'validation_interval_steps') else None
        # Configure checkpoint saving interval
        save_interval = self.config.trainer.save_interval_steps if hasattr(self.config.trainer, 'save_interval_steps') else None

        # Get data iterator so we can manually control iterations
        train_iterator = iter(self.train_dataloader)
    

        wandb.log({"System-core/non_emb_params": sum(p.numel() for p in self.model.model.layers.parameters()) / (1024 ** 2)}, step=global_step)

        epoch = 0
        start_time = time.time()
        while global_step < self.total_steps:
            try:
                data = next(train_iterator)
            except StopIteration:
                # Reset iterator for next epoch
                epoch += 1
                self.train_sampler.set_epoch(epoch=epoch)
                train_iterator = iter(self.train_dataloader)
                data = next(train_iterator)
                
            # Process batch    
            data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
            metric = self.training_step(data)
            
            if rank == 0:
                tracking.log(data=metric, step=global_step)
                wandb.log(get_gpu_memory_usage(rank=0), step=global_step) # only log rank 0, assume all ranks have the same memory usage
                wandb.log({"System-core/time": time.time() - start_time}, step=global_step)
            global_step += 1
            
            # Run validation if needed
            if validation_interval and global_step % validation_interval == 0:
                self._run_validation(global_step, rank, tracking)
                torch.distributed.barrier()
            
            # Save checkpoint if needed
            if save_interval and global_step % save_interval == 0:
                self.save_checkpoint(step=global_step)
            
        self._run_validation(global_step, rank, tracking)
        torch.distributed.barrier()
        
        # Final checkpoint
        self.save_checkpoint(step=global_step)


    def training_step(self, batch: TensorDict):
        rank = self.device_mesh.get_rank()

        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=False) / n_micro_batches
            loss.backward()
            step_loss += loss.item()
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)


        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3, 'train/grad_norm': grad_norm}

    def _run_validation(self, global_step, rank, tracking):
        """Helper method to run validation"""
        val_losses = []
        total_validation_count = self.config.trainer.total_validation_count
        val_count = 0
        for data in self.val_dataloader:
            data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
            val_loss = self.validation_step(data)
            val_losses.append(val_loss)
            val_count += data['input_ids'].size(0)
            if val_count >= total_validation_count:
                break
            
        if val_count < total_validation_count:
            logger.warn(f"Validation count {val_count} is less than total_validation_count {total_validation_count}")
        
        if rank == 0:
            avg_val_loss = torch.mean(torch.stack(val_losses))
            metric = {'val/loss': avg_val_loss.detach().item()}
            tracking.log(data=metric, step=global_step)
