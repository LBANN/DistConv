import torch


def fp32_allclose(a, b, rtol=1e-3, atol=1e-5):
    return torch.allclose(a, b, rtol=rtol, atol=atol)
