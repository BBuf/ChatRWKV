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

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

MODEL_NAME = '/data/user/cangshui/tianchao/pth_models/RWKV-5-World-1B5-v2-20231025-ctx4096'

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp32') # !!! currenly World models will overflow in fp16 !!!
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # !!! update rwkv pip package to 0.7.4+ !!!

########################################################################################################

QUESTIONS = ['你是谁？']

PAD_TOKENS = [] # [] or [0] or [187] -> probably useful

print(MODEL_NAME)
for q in QUESTIONS:
    out_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    ctx = f'Question: {q.strip()}\n\nAnswer:' # !!! do not use Q/A (corrupted by a dataset) or Bob/Alice (not used in training) !!!
    print(ctx, end = '')
    for i in range(200):
        tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]
        # tokens = [1] * 10
        # tokens = torch.ones(size=(1, 1024), device='cuda', dtype=torch.long)
        
        out, state = pipeline.model.forward(tokens, state)
        for n in occurrence:
            out[n] -= (0.4 + occurrence[n] * 0.4) # repetition penalty
        
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.1)
        if token == 0: break # exit when 'endoftext'
        
        out_tokens += [token]
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        
        tmp = pipeline.decode(out_tokens[out_last:])
        if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): # only print when the string is valid utf-8 and not end with \n
            print(tmp, end = '', flush = True)
            out_str += tmp
            out_last = i + 1
        
        if '\n\n' in tmp: # exit when '\n\n'
            out_str += tmp
            out_str = out_str.strip()
            break

    print('\n' + '=' * 50)
