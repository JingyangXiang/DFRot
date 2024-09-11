# 不做旋转
import os

import torch

run = print
if torch.cuda.is_available():
    run = os.system

model_names = ["meta-llama/Llama-2-7b-hf", 'meta-llama/Llama-2-13b-hf', "meta-llama/Meta-Llama-3-8B",
               "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2-7B"]
model_names = model_names[-1:]
w_bits = [4, 4, 4, 4, 4, 4, 4]
a_bits = [16, 4, 4, 4, 8, 8, 8]
kv_bits = [16, 4, 8, 16, 4, 8, 16]
"""for hadamard"""
w_rtn = False
template = ("python3 main.py --model {model_name} "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit} --w_clip --v_groupsize 128 --k_groupsize 128 --a_asym --k_asym --v_asym "
            "--fp32_had")

for model_name in model_names:
    for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
        # 运行GPTQ的情况
        run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit))
if w_rtn:
    template += ' --w_rtn'
    for model_name in model_names:
        for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
            # 运行GPTQ的情况
            run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit))
