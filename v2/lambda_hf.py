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

PAD_SEQ = []

########################################################################################################


import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("/data/user/cangshui/bbuf/upload_hf_model/rwkv-5-world-169m/", trust_remote_code=True, torch_dtype=torch.float32).to(0)
tokenizer = AutoTokenizer.from_pretrained("/data/user/cangshui/bbuf/upload_hf_model/rwkv-5-world-169m/", trust_remote_code=True)

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0

for d in todo:
    # 使用tokenizer对数据进行编码
    src = tokenizer(d[0], return_tensors="pt").to(0)["input_ids"].squeeze(0).tolist()
    dst = tokenizer(d[1], return_tensors="pt").to(0)["input_ids"].squeeze(0).tolist()

    logits = 0
    correct = True
    # 使用模型进行前向传播
    input_tensor = torch.tensor(src + dst).unsqueeze(0).to(0)
    with torch.no_grad():
        out = model(input_tensor).logits

    out = out.squeeze(0)
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
# 100 ppl 27.13 acc 40.0
# 200 ppl 18.47 acc 44.5
# 300 ppl 18.05 acc 46.0
# 400 ppl 20.3 acc 43.5
# 500 ppl 21.04 acc 42.8
# 600 ppl 20.02 acc 42.5
# 700 ppl 20.61 acc 41.29
# 800 ppl 20.59 acc 40.0
# 900 ppl 20.23 acc 40.11
# 1000 ppl 20.37 acc 41.0
# 1100 ppl 20.34 acc 41.18
# 1200 ppl 20.74 acc 40.42
# 1300 ppl 20.46 acc 40.62
# 1400 ppl 20.86 acc 40.07
# 1500 ppl 20.4 acc 40.87
# 1600 ppl 19.76 acc 41.31
# 1700 ppl 19.72 acc 41.71
# 1800 ppl 19.81 acc 42.0
# 1900 ppl 19.88 acc 41.42
# 2000 ppl 19.56 acc 41.8
# 2100 ppl 19.44 acc 41.81
# 2200 ppl 19.55 acc 41.68
# 2300 ppl 19.75 acc 41.74
# 2400 ppl 19.8 acc 41.75
# 2500 ppl 19.47 acc 41.96
# 2600 ppl 19.36 acc 41.85
# 2700 ppl 19.41 acc 41.74
# 2800 ppl 19.61 acc 41.82
# 2900 ppl 19.97 acc 41.62
# 3000 ppl 19.99 acc 41.57
# 3100 ppl 20.02 acc 41.84
# 3200 ppl 20.09 acc 41.72
# 3300 ppl 20.16 acc 41.64
# 3400 ppl 20.09 acc 41.71
# 3500 ppl 20.15 acc 41.74
# 3600 ppl 20.04 acc 41.69
# 3700 ppl 19.86 acc 41.78
# 3800 ppl 19.74 acc 41.79
# 3900 ppl 19.7 acc 42.15
# 4000 ppl 19.82 acc 42.15
# 4100 ppl 19.72 acc 42.29
# 4200 ppl 19.58 acc 42.38
# 4300 ppl 19.61 acc 42.4
# 4400 ppl 19.58 acc 42.34
# 4500 ppl 19.67 acc 42.29
# 4600 ppl 19.61 acc 42.41
# 4700 ppl 19.58 acc 42.4
# 4800 ppl 19.52 acc 42.31
# 4900 ppl 19.57 acc 42.16
# 5000 ppl 19.4 acc 42.3
# 5100 ppl 19.29 acc 42.31
# 5153 ppl 19.33 acc 42.21