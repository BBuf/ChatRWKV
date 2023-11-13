########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"{current_path}/../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

########################################################################################################

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
# MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth'
MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = []

########################################################################################################

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp32')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    logits = 0
    correct = True
    out, model_state = model.forward(src+dst, None, full_output=True)

    for i in range(len(dst)):
        probs = F.softmax(out[len(src)-1+i,:], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))


# Check LAMBADA...
# 100 ppl 27.11 acc 40.0
# 200 ppl 18.46 acc 44.5
# 300 ppl 18.04 acc 46.0
# 400 ppl 20.29 acc 43.5
# 500 ppl 21.03 acc 42.8
# 600 ppl 20.01 acc 42.5
# 700 ppl 20.6 acc 41.29
# 800 ppl 20.58 acc 40.0
# 900 ppl 20.22 acc 40.22
# 1000 ppl 20.36 acc 41.1
# 1100 ppl 20.33 acc 41.27
# 1200 ppl 20.73 acc 40.5
# 1300 ppl 20.46 acc 40.69
# 1400 ppl 20.85 acc 40.14
# 1500 ppl 20.4 acc 40.93
# 1600 ppl 19.75 acc 41.38
# 1700 ppl 19.72 acc 41.76
# 1800 ppl 19.81 acc 42.06
# 1900 ppl 19.88 acc 41.47
# 2000 ppl 19.55 acc 41.85
# 2100 ppl 19.43 acc 41.86
# 2200 ppl 19.54 acc 41.73
# 2300 ppl 19.74 acc 41.78
# 2400 ppl 19.79 acc 41.79
# 2500 ppl 19.46 acc 42.0
# 2600 ppl 19.35 acc 41.88
# 2700 ppl 19.41 acc 41.78
# 2800 ppl 19.61 acc 41.86
# 2900 ppl 19.97 acc 41.66
# 3000 ppl 19.98 acc 41.6
# 3100 ppl 20.02 acc 41.87
# 3200 ppl 20.09 acc 41.75
# 3300 ppl 20.15 acc 41.64
# 3400 ppl 20.09 acc 41.71
# 3500 ppl 20.14 acc 41.74
# 3600 ppl 20.04 acc 41.72
# 3700 ppl 19.85 acc 41.81
# 3800 ppl 19.73 acc 41.82
# 3900 ppl 19.7 acc 42.18
# 4000 ppl 19.81 acc 42.18
# 4100 ppl 19.72 acc 42.32
# 4200 ppl 19.58 acc 42.4
# 4300 ppl 19.61 acc 42.42
# 4400 ppl 19.58 acc 42.36
# 4500 ppl 19.67 acc 42.31
# 4600 ppl 19.61 acc 42.43
# 4700 ppl 19.58 acc 42.43
# 4800 ppl 19.52 acc 42.33
# 4900 ppl 19.56 acc 42.18
# 5000 ppl 19.39 acc 42.32
# 5100 ppl 19.29 acc 42.33
# 5153 ppl 19.32 acc 42.25
