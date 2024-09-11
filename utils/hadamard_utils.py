import os

import numpy as np
import torch

from utils.hadamard_matrix import get_had12, get_had108, get_had140, get_had156, get_had172, get_had36, get_had52, \
    get_had60, get_had20, get_had40, get_had28

try:
    import fast_hadamard_transform
except:
    fast_hadamard_transform = None


# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
def largest_power_of_2(n):
    if n <= 0:
        raise ValueError("The input must be a positive integer.")

    power = 0
    while n % 2 == 0:
        n //= 2
        power += 1

    return 2 ** power


def read_and_process_file(file_path):
    matrix = []

    # 读取文件内容
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉首尾的空白字符
            line = line.strip()

            # 跳过分隔行（假设分隔行的第一个字符是方括号或破折号）
            if '-----' in line:
                continue

            # 分隔数值
            values = line.lstrip("[").rstrip("]").replace('|', ' ').split()

            # 将数值从字符串转换为整数，并过滤掉其他符号
            row = [int(x) for x in values if x in ['1', '-1']]

            # 添加到矩阵中
            matrix.append(row)

    # 转化为numpy矩阵
    return np.array(matrix)


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert (is_pow2(n // 172))
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert (is_pow2(n // 156))
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate 
        assert (is_pow2(n // 140))
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate 
        assert (is_pow2(n // 108))
        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert (is_pow2(n // 60))
        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert (is_pow2(n // 52))
        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert (is_pow2(n // 36))
        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:  # llama-3 up
        assert (is_pow2(n // 28))
        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        assert (is_pow2(n // 40))
        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert (is_pow2(n // 20))
        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert (is_pow2(n // 12))
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    elif n % 148 == 0:
        assert (is_pow2(n // 148))
        K = 148
        hadK = torch.from_numpy(read_and_process_file(os.path.join(os.path.dirname(__file__), "./hadamard_148.txt")))
    else:
        assert (is_pow2(n))
        K = 1

    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def matmul_hadU_cuda(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0 / torch.tensor(n).sqrt())
        # if transpose:
    #     hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)
    input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0 / torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    return matmul_hadU(Q).to(device)


def repeat_rotate(hadK, temp):
    if hadK.shape[0] != temp.shape[1]:
        assert hadK.shape[0] < temp.shape[1]
        hadK = hadK.repeat_interleave(temp.shape[1] // hadK.shape[0], dim=0)
    return hadK


def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    bias_ = module.bias.data.float().cuda() if module.bias is not None else None
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
            if bias_ is not None:
                raise NotImplementedError
                # bias_ = matmul_hadU_cuda(bias_.reshape(1, -1), had_K, K).flatten()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
            if bias_ is not None:
                raise NotImplementedError
                # bias_ = matmul_hadU_cuda(bias_.reshape(1, -1), had_K, K).flatten()
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float32)
        if len(hadK.shape) == 2:
            hadK = hadK.unsqueeze(0)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            hadK = repeat_rotate(hadK, temp)
            temp = torch.einsum('b h c, h c i -> b h i', temp.to(torch.float32), hadK.to(torch.float32))
            W_ = temp.reshape(transposed_shape).t()
            if bias_ is not None:
                temp = bias_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                bias_ = torch.einsum('b h c, h c i -> b h i', temp.to(torch.float32), hadK.to(torch.float32))
                bias_ = bias_.flatten()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            hadK = repeat_rotate(hadK, temp)
            temp = torch.einsum('b h c, h c i -> b h i', temp.to(torch.float32), hadK.to(torch.float32))
            W_ = temp.reshape(init_shape)
            if bias_ is not None:
                raise NotImplementedError
                # temp = bias_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                # bias_ = torch.einsum('b h c, h c i -> b h i', temp.to(torch.float32), hadK.to(torch.float32))
                # bias_ = bias_.flatten()
    module.weight.data = W_.to(device=dev, dtype=dtype)
    if bias_ is not None:
        module.bias.data = bias_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


if __name__ == "__main__":
    # 调用函数处理文件
    matrix = get_hadK(148)[0]
    print(torch.diag(matrix @ matrix.t()), torch.sum(matrix @ matrix.t()) / matrix.shape[0])
