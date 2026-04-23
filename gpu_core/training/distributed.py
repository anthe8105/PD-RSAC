import os
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..config import DistributedConfig
from ..networks.sac import SACAgent
from ..features.replay_buffer import GPUReplayBuffer
from .trainer import SACTrainer, TrainingMetrics


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
    master_addr: str = '127.0.0.1',
    master_port: str = '29500'
):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_DEBUG'] = 'INFO'  # Debug NCCL issues
    
    print(f"[Rank {rank}] Initializing process group: {master_addr}:{master_port}")
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30)
    )
    
    print(f"[Rank {rank}] Process group initialized successfully")
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Using GPU {rank}")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device_from_rank(rank: int, gpus: Optional[List[int]] = None) -> torch.device:
    if gpus:
        device_id = gpus[rank % len(gpus)]
    else:
        device_id = rank
    return torch.device(f'cuda:{device_id}')


class DistributedTrainer(SACTrainer):
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config,
        distributed_config: DistributedConfig,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        self.rank = rank
        self.world_size = world_size
        self.distributed_config = distributed_config
        
        device = get_device_from_rank(rank, distributed_config.gpus)
        
        super().__init__(
            agent=agent,
            replay_buffer=replay_buffer,
            training_config=training_config,
            device=str(device),
            **kwargs
        )
        
        self._wrap_models_ddp()
    
    def _wrap_models_ddp(self):
        if self.world_size > 1:
            device_id = self.rank if not self.distributed_config.gpus else \
                        self.distributed_config.gpus[self.rank % len(self.distributed_config.gpus)]
            
            self.agent.actor = DDP(
                self.agent.actor,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            self.agent.critic = DDP(
                self.agent.critic,
                device_ids=[device_id],
                output_device=device_id,
                broadcast_buffers=False
            )
    
    def sync_replay_buffer(self):
        if self.world_size <= 1:
            return
        
        buffer_size = torch.tensor([len(self.replay_buffer)], device=self.device)
        dist.all_reduce(buffer_size, op=dist.ReduceOp.SUM)
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if self.world_size <= 1:
            return metrics
        
        keys = sorted(metrics.keys())
        values = torch.tensor([metrics[k] for k in keys], device=self.device)
        
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values / self.world_size
        
        return {k: v.item() for k, v in zip(keys, values)}
    
    def train_step(self) -> Dict[str, float]:
        metrics = super().train_step()
        
        if metrics and self.world_size > 1:
            metrics = self.all_reduce_metrics(metrics)
        
        return metrics
    
    def broadcast_weights(self, src: int = 0):
        if self.world_size <= 1:
            return
        
        for param in self.agent.parameters():
            dist.broadcast(param.data, src=src)
    
    def save_checkpoint(self, filename: str):
        if self.rank == 0:
            super().save_checkpoint(filename)
        
        if self.world_size > 1:
            dist.barrier()
    
    def load_checkpoint(self, filename: str):
        super().load_checkpoint(filename)
        
        if self.world_size > 1:
            self.broadcast_weights(src=0)
            dist.barrier()
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class GradientAccumulationTrainer(DistributedTrainer):
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config,
        accumulation_steps: int = 4,
        **kwargs
    ):
        super().__init__(agent, replay_buffer, training_config, **kwargs)
        self.accumulation_steps = accumulation_steps
        self.accumulated_step = 0
    
    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        transitions = self.replay_buffer.sample(self.config.batch_size)
        
        states = transitions.states
        actions = transitions.actions
        rewards = transitions.rewards
        next_states = transitions.next_states
        dones = transitions.dones
        
        critic_loss = self.agent.compute_critic_loss(
            states, actions, rewards, next_states, dones
        )
        scaled_critic_loss = critic_loss / self.accumulation_steps
        scaled_critic_loss.backward()
        
        actor_loss, log_prob, _aux = self.agent.compute_actor_loss(states)
        scaled_actor_loss = actor_loss / self.accumulation_steps
        scaled_actor_loss.backward()
        
        self.accumulated_step += 1
        
        if self.accumulated_step >= self.accumulation_steps:
            self.agent.critic_optimizer.step()
            self.agent.actor_optimizer.step()
            self.agent.critic_optimizer.zero_grad()
            self.agent.actor_optimizer.zero_grad()
            
            if self.agent.auto_alpha:
                alpha_loss = self.agent.compute_alpha_loss(log_prob)
                alpha_loss.backward()
                self.agent.alpha_optimizer.step()
                self.agent.alpha_optimizer.zero_grad()
            
            self.agent.soft_update_target()
            self.accumulated_step = 0
            self.global_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.agent.alpha.item()
        }


class MixedPrecisionTrainer(DistributedTrainer):
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config,
        use_amp: bool = True,
        **kwargs
    ):
        super().__init__(agent, replay_buffer, training_config, **kwargs)
        self.use_amp = use_amp
        
        if use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        transitions = self.replay_buffer.sample(self.config.batch_size)
        
        states = transitions.states
        actions = transitions.actions
        rewards = transitions.rewards
        next_states = transitions.next_states
        dones = transitions.dones
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                critic_loss = self.agent.compute_critic_loss(
                    states, actions, rewards, next_states, dones
                )
            
            self.agent.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.agent.critic_optimizer)
            
            with torch.amp.autocast('cuda'):
                actor_loss, log_prob, _aux = self.agent.compute_actor_loss(states)
            
            self.agent.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.agent.actor_optimizer)
            
            self.scaler.update()
        else:
            return super().train_step()
        
        if self.agent.auto_alpha:
            alpha_loss = self.agent.compute_alpha_loss(log_prob)
            self.agent.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.agent.alpha_optimizer.step()
        
        self.agent.soft_update_target()
        self.global_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.agent.alpha.item()
        }


def _distributed_worker_wrapper(rank, world_size, backend, train_fn, train_args):
    """Top-level worker function for multiprocessing (must be picklable)"""
    setup_distributed(rank, world_size, backend)
    try:
        # train_args is a tuple containing the original args
        train_fn(rank, world_size, *train_args)
    finally:
        cleanup_distributed()


def spawn_distributed_training(
    train_fn,
    world_size: int,
    args: Tuple = (),
    kwargs: Dict = None,
    backend: str = 'nccl'
):
    import torch.multiprocessing as mp
    
    # Note: kwargs not used, all args should be in args tuple
    mp.spawn(
        _distributed_worker_wrapper,
        args=(world_size, backend, train_fn, args),
        nprocs=world_size,
        join=True
    )
