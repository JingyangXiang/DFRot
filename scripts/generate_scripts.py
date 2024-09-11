import os

import numpy as np
import torch

run = print
if torch.cuda.is_available():
    run = os.system

clip_ratios = np.linspace(0.6, 1.0, 21)
model_names = ["meta-llama/Llama-2-7b-hf", 'meta-llama/Llama-2-13b-hf', "meta-llama/Meta-Llama-3-8B",
               "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2-7B", "Qwen/Qwen2-1.5B"]
model_names = model_names[-1:]
"""for baseline"""
w_bits = [16, ]
a_bits = [16, ]
kv_bits = [16, ]
template = ("python3 main.py --model {model_name} "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit}")
for model_name in model_names:
    for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
        # 运行GPTQ的情况
        run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit))

"""for hadamard"""
w_bits = [16, 4, 4, 4, 4, 4, 4]
a_bits = [16, 4, 4, 4, 8, 8, 8]
kv_bits = [16, 4, 8, 16, 4, 8, 16]
run_rtn = False
disable_qk_rotation = False
rotate_modes = ['random', 'hadamard'][1:2]
template = ("python3 main.py --model {model_name}  --rotate "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit} --w_clip --v_groupsize 128 --k_groupsize 128 --a_asym --k_asym --v_asym "
            "--rotate_mode {rotate_mode} --fp32_had")
for rotate_mode in rotate_modes:
    for model_name in model_names:
        for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
            if run_rtn:
                # 运行RTN的情况
                if a_bit != 16:
                    for a_clip_ratio in clip_ratios:
                        run(template.format(rotate_mode=rotate_mode, model_name=model_name, w_bit=w_bit, a_bit=a_bit,
                                            k_bit=k_bit, v_bit=v_bit) + f" --w_rtn --a_clip_ratio {a_clip_ratio:.2f}")

            # 运行GPTQ的情况
            run(template.format(rotate_mode=rotate_mode, model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit,
                                v_bit=v_bit) + " --lm_eval")
            if disable_qk_rotation:
                run(template.format(rotate_mode=rotate_mode, model_name=model_name, w_bit=w_bit, a_bit=a_bit,
                                    k_bit=k_bit, v_bit=v_bit) + ' --disable_qk_rotation ')

"""for orthogonal_procrustes"""
import random

seed = random.randint(0, 0)
indices_files = ["LLaMA-2-7B-4.npy", "LLaMA-2-13B-4.npy",
                 "LLaMA-3-8B-4.npy", "Mistral-7B-4.npy", 'Mistral-7B-V3-4.npy', 'QWen2-7B-4.npy']
template = ("python3 main.py --model {model_name}  --rotate "
            "--w_bits {w_bit} --a_bits {a_bit} --k_bits {k_bit} "
            "--v_bits {v_bit} --w_clip --v_groupsize 128 --k_groupsize 128 --a_asym --k_asym --v_asym "
            "--rotate_mode orthogonal_procrustes --indices_path {indices_path} --fp32_had --seed {seed}")
data_principles = ['alter', 'massive']

alphas = [1, 5, 10, 20, 50, 100, 200, 500]
for rotate_mode in rotate_modes:
    for data_principle in data_principles:
        for alpha in alphas:
            if data_principle == 'alter':
                indices_paths = [os.path.join(f"rms_norm_feature_{rotate_mode}_{data_principle}", f"{alpha}", file) for
                                 file in indices_files]
            else:
                indices_paths = [os.path.join(f"rms_norm_feature_{rotate_mode}_{data_principle}", file) for file
                                 in indices_files]
            for indices_path, model_name in zip(indices_paths, model_names):
                for w_bit, a_bit, k_bit, v_bit in zip(w_bits, a_bits, kv_bits, kv_bits):
                    if run_rtn:
                        # 运行RTN的情况
                        if a_bit != 16:
                            for a_clip_ratio in clip_ratios:
                                run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit,
                                                    v_bit=v_bit,
                                                    indices_path=indices_path,
                                                    seed=seed) + f" --w_rtn --a_clip_ratio {a_clip_ratio:.2f}")
                        else:
                            run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit,
                                                v_bit=v_bit,
                                                indices_path=indices_path,
                                                seed=seed) + f" --w_rtn")
                    # 运行GPTQ的情况
                    if a_bit != 16:
                        run(template.format(model_name=model_name, w_bit=w_bit, a_bit=a_bit, k_bit=k_bit, v_bit=v_bit,
                                            indices_path=indices_path, seed=seed) + " --lm_eval")
            if data_principle == 'massive':
                break
