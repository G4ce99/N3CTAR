import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from contextlib import contextmanager

## DDP utils ##
def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def cleanup():
  dist.destroy_process_group()

## contextmanager for enabling/disabling profiling ##
@contextmanager
def check_profile(enabled, **kwargs):
  if enabled:
    from torch.profiler import profile
    with profile(**kwargs) as prof:
      yield prof
  else:
    class DummyContext:
      def __enter__(self): return self
      def __exit__(self, *args): pass
    yield DummyContext()