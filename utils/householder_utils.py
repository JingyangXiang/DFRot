import glob
import math

import torch


def householder(x):
    return torch.eye(x.shape[0]).to(x) - 2 * torch.einsum('i,j->ij', x, x)


def house_v1(shape, idx):
    other = 1 / math.sqrt(2 * (shape + math.sqrt(shape)))
    data = torch.ones(shape) * other
    main = math.sqrt(0.5 * (1 + 1 / math.sqrt(shape)))
    data[idx] = main
    return data


def house_v2(shape, idx):
    other = 1 / math.sqrt(2 * (shape - math.sqrt(shape)))
    data = torch.ones(shape) * other
    main = math.sqrt(0.5 * (1 - 1 / math.sqrt(shape)))
    data[idx] = main
    return data


def get_householder_indices(indices_path):
    with open(indices_path, 'r') as f:
        contents = f.readlines()
    content = contents[-1].strip()
    outliers_indices = list(eval(content[content.find(" "):]))
    return outliers_indices


if __name__ == '__main__':
    shape = 12
    idx = 3
    unit_vector1 = house_v1(shape, idx) * torch.sign(torch.randn(shape))
    unit_vector2 = house_v2(shape, idx) * torch.sign(torch.randn(shape))

    house1 = householder(unit_vector1)
    house2 = householder(unit_vector2)

    print("unit_vector1: ", unit_vector1, torch.norm(unit_vector1))
    print("unit_vector2: ", unit_vector2, torch.norm(unit_vector2))

    print("house1 @ house1      : ", torch.diag(house1 @ house1).mean())
    print("house1 @ house1.t()  : ", torch.diag(house1 @ house1.t()).mean())
    print("house2 @ house2      : ", torch.diag(house2 @ house2).mean())
    print("house2 @ house2.t()  : ", torch.diag(house2 @ house2.t()).mean())

    """
    It seems that house_v2 is more uniform
    """
    size = 12
    indices = [2, 5, 7, 9]
    matrix = torch.eye(size).to('cpu')
    for idx in indices:
        matrix = matrix @ householder(house_v2(size, idx) * torch.where(torch.randn(size) > 0, 1, -1))
    # It should take care that householder need to transpose when random +1 and -1
    print("output: ", (matrix @ matrix.t()).diag().mean(), (matrix @ matrix).diag().mean())

    size = 12
    indices = [2, 5, 7, 9]
    matrix = torch.eye(size).to('cpu')
    for idx in indices:
        matrix = matrix @ householder(house_v2(size, idx))
    # It should take care that householder need to transpose when multiple matmul
    print("output: ", (matrix @ matrix.t()).diag().mean(), (matrix @ matrix).diag().mean())

    device = 'mps'
    sizes = [4096, 5120, 4096]
    indices_paths = glob.glob("../outlier/*.log")
    for indices_path, size in zip(indices_paths, sizes):
        indices = get_householder_indices(indices_path)
        matrix = torch.eye(size).to(device)
        for idx in indices:
            matrix = matrix @ householder(house_v2(size, idx).to(device))
        # It should take care that householder need to transpose when multiple matmul
        print(f"{indices_path:<20} {(matrix @ matrix.t()).diag().mean():.2f}, {(matrix @ matrix).diag().mean():.2f}")
