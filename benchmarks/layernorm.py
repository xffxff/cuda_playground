import triton
from cuda_playground.triton import layernorm as triton_layernorm
from cuda_playground_backend import layernorm_forward as cuda_layernorm
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'cuda'],
        line_names=['Triton', 'Torch', 'Cuda'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float32},
    ))
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return triton_layernorm.LayerNorm.apply(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704
        
        if provider == "cuda":
            return cuda_layernorm(x, weight, bias)


    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_layer_norm.run(save_path='.', print_data=True)