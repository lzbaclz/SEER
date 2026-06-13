"""Background contender used by substrate_a7_mps_isolated.py."""
import os
import time

import torch
dev = torch.device('cuda', int(os.environ.get('SEER_DEVICE_INDEX', '0')))
buf_mb = float(os.environ.get('SEER_CONTENDER_MB', '1.0'))
nbytes = int(buf_mb * 1024 * 1024)
nelem = nbytes // 2
host = torch.empty(nelem, dtype=torch.float16, pin_memory=True)
devv = torch.empty(nelem, dtype=torch.float16, device=dev)
stream = torch.cuda.Stream(device=dev)
with torch.cuda.device(dev):
    while True:
        with torch.cuda.stream(stream):
            devv.copy_(host, non_blocking=True)
            host.copy_(devv, non_blocking=True)
        if int(time.perf_counter() * 100) % 200 == 0:
            stream.synchronize()
