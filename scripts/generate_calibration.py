import os
import pickle

import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

import misc
from utils import model_utils, data_utils, rotation_utils, quant_utils
from utils.rotation_utils import QKRotationWrapper


@torch.no_grad()
def main():
    args = misc.parser_gen()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # Rotate the weights
    rotation_utils.fuse_layer_norms(model)

    # Get data from RoPE
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    k_quant_config = {'k_bits': args.k_bits, "k_groupsize": args.k_groupsize,
                      "k_sym": not args.k_asym, "k_clip_ratio": args.k_clip_ratio,
                      'disable_qk_rotation': args.disable_qk_rotation}
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_function_name,
            config=model.config,
            **k_quant_config)

    # setting model params
    model.seqlen = args.model_max_length
    model.config.use_cache = False

    # get train loader to calibrate
    train_loader = data_utils.get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=2048,
        hf_token=None,
        eval_mode=False
    )

    os.makedirs(args.output_dir, exist_ok=True)
    saved_path = os.path.join(args.output_dir, f"{os.path.basename(args.model)}_qkv.pkl")

    if not os.path.exists(saved_path):

        rms_norm_outputs = {}
        handles = []

        """RMSNorm数据特征记录"""
        # find all LlamaRMSNorm
        fulls = quant_utils.find_qlayers(model, layers=[LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm])

        def add_batch(name):
            def tmp(_, inp, out):
                if name in rms_norm_outputs:
                    rms_norm_outputs[name].append(out.data.detach().cpu().numpy())  # noqa: F821])
                else:
                    rms_norm_outputs[name] = [out.data.detach().cpu().numpy(), ]
                assert len(out.data.shape) == 3

            return tmp

        for name, layer in fulls.items():
            # we don't save RMSNorm output before lm_head
            if "layers" in name:
                handles.append(layer.register_forward_hook(add_batch(name)))

        """QK数据特征记录"""
        fulls = quant_utils.find_qlayers(model, layers=[QKRotationWrapper, ])

        def add_batch_rope(name):
            def tmp(_, inp, out):
                if name in rms_norm_outputs:
                    rms_norm_outputs[name].append(
                        [out[0].data.detach().cpu().numpy(), out[1].data.detach().cpu().numpy()])  # noqa: F821])
                else:
                    rms_norm_outputs[name] = [
                        [out[0].data.detach().cpu().numpy(), out[1].data.detach().cpu().numpy()], ]
                # assert len(out.data.shape) == 3
                assert len(out) == 2

            return tmp

        for name, layer in fulls.items():
            handles.append(layer.register_forward_hook(add_batch_rope(name)))

        """Value数据特征记录"""
        fulls = quant_utils.find_qlayers(model, layers=[torch.nn.Linear, ])

        def add_batch_value(name):
            def tmp(_, inp, out):
                if name in rms_norm_outputs:
                    rms_norm_outputs[name].append(out.data.detach().cpu().numpy())  # noqa: F821])
                else:
                    rms_norm_outputs[name] = [out.data.detach().cpu().numpy(), ]
                assert len(out.data.shape) == 3

            return tmp

        for name, layer in fulls.items():
            if "v_proj" in name:
                handles.append(layer.register_forward_hook(add_batch_value(name)))

        model.to(misc.DEV)
        for batch in train_loader:
            model(batch[0].to(misc.DEV))

        for h in handles:
            h.remove()

        # saved with pkl file
        with open(saved_path, 'wb') as f:
            pickle.dump(rms_norm_outputs, f)

    # read saved
    with open(saved_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    print(f"model {args.model}")
    print(f"saved keys: {loaded_dict.keys()}")


if __name__ == '__main__':
    if torch.cuda.is_available():
        main()
    else:
        output_dir = "calibration"
        models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B",
                  "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2-7B"]
        template = "python3 generate_calibration.py --model {model:<26} --nsamples 1 --output_dir {output_dir} --disable_qk_rotation"
        for model in models:
            print(template.format(model=model, output_dir=output_dir))
