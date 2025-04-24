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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, PreTrainedModel
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
# import wandb
import time
from codetiming import Timer
from contextlib import contextmanager
from typing import Dict

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

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

            self.model.resize_token_embeddings(len(self.tokenizer))
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
        # The total training steps in SFT is mainly for early exit
        # assert only 1 is valid
        
        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
        # Configure validation interval
        validation_interval = self.config.trainer.validation_interval_steps if hasattr(self.config.trainer, 'validation_interval_steps') else None
        # Configure checkpoint saving interval
        save_interval = self.config.trainer.save_interval_steps if hasattr(self.config.trainer, 'save_interval_steps') else None

        # Get data iterator so we can manually control iterations
        train_iterator = iter(self.train_dataloader)
    
        if rank == 0:
            tracking.log({"System-core/non_emb_params": sum(p.numel() for p in self.model.model.layers.parameters()) / (1024 ** 2)}, step=global_step)

        epoch = 0
        start_time = time.time()
        while global_step < self.total_steps:
            timing_raw = {}
            with _timer('step', timing_raw):
                try:
                    with _timer('data_loading', timing_raw):
                        data = next(train_iterator)
                except StopIteration:
                    # Reset iterator for next epoch
                    epoch += 1
                    self.train_sampler.set_epoch(epoch=epoch)
                    train_iterator = iter(self.train_dataloader)
                    with _timer('data_loading', timing_raw):
                        data = next(train_iterator)
                    
                # Process batch    
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                with _timer('train_step', timing_raw):
                    metric = self.training_step(data, timing_raw)

                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    tracking.log(get_gpu_memory_usage(rank=0), step=global_step) # only log rank 0, assume all ranks have the same memory usage
                    tracking.log({"System-core/time": time.time() - start_time}, step=global_step)
                global_step += 1
                
                # Run validation if needed
                if validation_interval and global_step % validation_interval == 0:
                    with _timer('validation', timing_raw):
                        self._run_validation(global_step, rank, tracking)
                    torch.distributed.barrier()
                
                # Save checkpoint if needed
                if save_interval and global_step % save_interval == 0:
                    with _timer('save_checkpoint', timing_raw):
                        self.save_checkpoint(step=global_step)

            if rank == 0:
                if timing_raw != {}:
                    log_timing_raw = {}
                    for k, v in timing_raw.items():
                        log_timing_raw[f'timing_s/{k}'] = v
                    tracking.log(log_timing_raw, step=global_step)

            # === TFLOPS calculation ===
            if rank == 0:
                if 'train_step' in timing_raw and timing_raw['train_step'] > 0:
                    tflops = self.estimate_tflops(self.fsdp_model, data, timing_raw['train_step'])
                    metric['train/tflops'] = tflops
                    tracking.log({'train/tflops': tflops}, step=global_step)
            
        self._run_validation(global_step, rank, tracking)
        torch.distributed.barrier()
        
        # Final checkpoint
        self.save_checkpoint(step=global_step)

    def estimate_tflops(self, model, batch, step_time_s):
        """
        Roughly estimate TFLOPS for the current step.
        """
        # Number of model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Assume input is TensorDict, get input_ids shape
        if isinstance(batch, TensorDict):
            batch_size = batch['input_ids'].shape[0]
            seq_len = batch['input_ids'].shape[1]
        else:
            batch_size = batch['input_ids'].size(0)
            seq_len = batch['input_ids'].size(1)
        # 2 * param * batch * seq (forward + backward)
        flops = 2 * num_params * batch_size * seq_len #TODO: coe may go through a param several times, this may be taken into account
        tflops = flops / step_time_s / 1e12
        return tflops

    def training_step(self, batch: TensorDict, timing_raw=None):
        if timing_raw is None:
            timing_raw = {}
        rank = self.device_mesh.get_rank()

        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            with _timer('micro_batch_forward_backward', timing_raw):
                with _timer('micro_batch_forward', timing_raw):
                    loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=False) / n_micro_batches
                with _timer('micro_batch_loss_backward', timing_raw):
                    loss.backward()
                step_loss += loss.item()
                grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)


        log_gpu_memory_usage('Before optimizer step', logger=logger)

        with _timer('micro_batch_optimizer_step', timing_raw):
            self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        with _timer('micro_batch_lr_scheduler_step', timing_raw):
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
