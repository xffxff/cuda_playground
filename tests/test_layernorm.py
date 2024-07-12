import torch
import torch.testing as tt
from cuda_playground.triton import layernorm as triton_layernorm
from cuda_playground_backend import layernorm_forward as cuda_layernorm


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = triton_layernorm.LayerNorm.apply(x, w_shape, weight, bias, eps)
    y_cuda = cuda_layernorm(x, weight, bias)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    print(y_tri)
    print(y_cuda)
    # compare

    tt.assert_close(y_tri, y_ref, rtol=0, atol=1e-2)
    tt.assert_close(y_cuda, y_ref, rtol=0, atol=1e-2)

test_layer_norm(1151, 8192, torch.float32)