import os

import torch

run = print
if torch.cuda.is_available():
    run = os.system

model_names = ["meta-llama/Llama-2-7b-hf", 'meta-llama/Llama-2-13b-hf', "meta-llama/Meta-Llama-3-8B",
               "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2-7B"]
w_bits = [4, 4, 4, 4]
a_bits = [16, 4, 4, 4]
kv_bits = [16, 16, 8, 4]

model_names = model_names[-1:]
w_bits = w_bits[-3:]
a_bits = a_bits[-3:]
kv_bits = kv_bits[-3:]

separate_fwd = True
separate_gptq = False

"""for random"""
template = ("python3 main_separate.py --model {model_name}  --rotate "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit} --w_clip --v_groupsize 128 --k_groupsize 128 --a_asym --k_asym --v_asym "
            "--rotate_mode random --fp32_had")

if separate_fwd:
    template += ' --separate_fwd'
if separate_gptq:
    template += ' --separate_gptq'

for model_name in model_names:
    for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
        # 运行GPTQ的情况
        run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit))
"""for hadamard"""
template = ("python3 main_separate.py --model {model_name}  --rotate "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit} --w_clip --v_groupsize 128 --k_groupsize 128 --a_asym --k_asym --v_asym "
            "--rotate_mode hadamard --fp32_had")

if separate_fwd:
    template += ' --separate_fwd'
if separate_gptq:
    template += ' --separate_gptq'

for model_name in model_names:
    for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
        # 运行GPTQ的情况
        run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit))
