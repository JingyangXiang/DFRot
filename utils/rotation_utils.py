import functools
import math
import random
import typing

import torch
import tqdm

import misc
from utils import model_utils
from utils import monkeypatch
from utils import quant_utils
from utils.hadamard_utils import apply_exact_had_to_linear, is_pow2, random_hadamard_matrix
from utils.householder_utils import house_v2, householder

try:
    from fast_hadamard_transform import hadamard_transform
except:
    hadamard_transform = None


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    model_type = model_utils.get_model_type(model)

    kwargs = {'model': model, 'model_type': model_type}

    layers = model_utils.get_transformer_layers(**kwargs)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:

        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or \
                model_type == model_utils.QW_MODEL:
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            W_norm = layer.post_attention_layernorm.weight.data
            layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
            W_norm = layer.input_layernorm.weight.data
            layer.input_layernorm.weight.data = torch.ones_like(W_norm)
        else:
            raise ValueError(f'Unknown model type {model_type}')

    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def random_householder_matrix(size, device, indices):
    assert isinstance(indices, list), "idx needs to be a list of outlier channel idx"
    matrix = torch.eye(size).to(device)
    for idx in indices:
        matrix = matrix @ householder(
            house_v2(size, idx).to(matrix) * torch.where(torch.randn(size) > 0, 1, -1).to(matrix))
    return matrix.to(torch.float64)


def get_orthogonal_matrix(size, mode, device, **kwargs):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    elif mode == "householder":
        return random_householder_matrix(size, device, kwargs.get("indices"))
    elif mode == 'hadamard_householder':
        hadamard_matrix = random_hadamard_matrix(size, device)
        householder_matrix = householder(house_v2(size, random.randint(0, size - 1))).to(hadamard_matrix)
        return hadamard_matrix @ householder_matrix
    elif mode == 'orthogonal_procrustes':
        assert kwargs.get('indices') is not None
        orthogonal_procrustes_matrix = kwargs.get('indices')
        return orthogonal_procrustes_matrix
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=misc.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=misc.DEV, dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=misc.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=misc.DEV, dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QW_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=misc.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=misc.DEV, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=misc.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=misc.DEV, dtype=dtype)


def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QW_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=misc.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=misc.DEV, dtype=dtype)


def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QW_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=misc.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=misc.DEV, dtype=dtype)
    apply_exact_had_to_linear(W, had_dim=-1,
                              output=False)  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=misc.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=misc.DEV, dtype=dtype)


def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1 / math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(
        X.shape)


def rotate_faster_down_proj(layer, model_type, hardK):
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QW_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')

    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device=misc.DEV, dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=misc.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device=misc.DEV, dtype=dtype)


def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QW_MODEL:
        o_proj = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


@torch.inference_mode()
def rotate_model(model, Q):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    misc.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]


def register_online_rotation(module, Q: torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        self.head_dim = head_dim
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        self.disable_qk_rotation = kwargs['disable_qk_rotation']
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], \
                f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1,
                                       sym=self.k_sym, clip_ratio=self.k_clip_ratio)

        # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        assert self.head_dim == q.shape[-1]
        dtype = q.dtype
        if not self.disable_qk_rotation:
            q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
            k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)

        (bsz, num_heads, seq_len, head_dim) = k.shape
        if self.k_bits < 16:
            if self.k_groupsize == -1:  # token-wise quantization
                token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
                self.k_quantizer.find_params(token_wise_k)
                k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
            else:  # head-wise quantization
                per_head_k = k.reshape(-1, head_dim)
                self.k_quantizer.find_params(per_head_k)
                k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name,
                                                                    functools.partial(QKRotationWrapper, *args,
                                                                                      **kwargs))
    setattr(module, attr_name, wrapper)
