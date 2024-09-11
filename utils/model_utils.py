import logging
import os
from typing import List

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM, MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2RMSNorm

import misc

LLAMA_MODEL = LlamaForCausalLM
LLAMA_LAYER = LlamaDecoderLayer
MISTRAL_MODEL = MistralForCausalLM
MISTRAL_LAYER = MistralDecoderLayer
QW_MODEL = Qwen2ForCausalLM
QW_LAYER = Qwen2DecoderLayer


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QW_MODEL):
        return QW_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, MISTRAL_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, QW_MODEL):
        return "apply_rotary_pos_emb"
    else:
        raise NotImplementedError


def get_layers(model):
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    elif isinstance(model, MISTRAL_MODEL):
        return model.model.layers
    elif isinstance(model, QW_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_mistral(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = MistralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_qwen(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(torch.empty_like(model.lm_head.weight))
        model.lm_head.weight.data = model.model.embed_tokens.weight.clone()
    return model


def get_model(model_name, hf_token=None):
    if 'llama' in model_name.lower():
        return get_llama(model_name, hf_token)
    elif 'mistral' in model_name.lower():
        return get_mistral(model_name, hf_token)
    elif "qwen" in model_name.lower():
        return get_qwen(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        model_type = MISTRAL_MODEL
    elif isinstance(model, QW_MODEL):
        model_type = QW_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def get_embeddings(model, model_type) -> List[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == MISTRAL_MODEL:
        return [model.model.embed_tokens]
    elif model_type == QW_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == MISTRAL_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == QW_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == MISTRAL_MODEL:
        return model.lm_head
    elif model_type == QW_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, LlamaRMSNorm)
    elif model_type == MISTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, MistralRMSNorm)
    elif model_type == QW_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, Qwen2RMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm


def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == MISTRAL_MODEL:
        return model.config.intermediate_size
    elif model_type == QW_MODEL:
        return model.config.intermediate_size
    else:
        raise ValueError(f'Unknown model type {model_type}')


def replace_modules(root: torch.nn.Module, type_to_replace, new_module_factory, replace_layers: bool) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')


def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())

        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(misc.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
