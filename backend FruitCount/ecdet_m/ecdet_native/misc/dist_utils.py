"""Only the distributed query used by the upstream backbone loader."""
import torch.distributed as dist

def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()
