import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(x)

    model = nn.Linear(16, 16).to(device)
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)

    inp = torch.randn(8, 16, device=device)
    out = ddp(inp).sum()
    out.backward()

    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}, all_reduce={x.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
