import numpy as np
from typing import Tuple, Optional
import torch
import triton
import triton.language as tl

from deep_gemm import fp8_gemm_nt

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)


def check_sf_layout(sf: torch.Tensor,
                    mn: int,
                    k: int,
                    gran: Tuple[int, int],
                    num_groups: Optional[int],
                    tma_stride_check: bool = False,
                    type_check: Optional[torch.dtype] = None) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(
        k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


@triton.jit
def _per_token_quant_and_transform_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    output_ptr,
    stride_output_0,
    stride_output_1,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    token_num,
    size_k,
    fp8_max,
    fp8_min,
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num = tl.num_programs(1)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    k_ptr_offs = hidden_dim_block_index * BLOCK + tl.arange(0, BLOCK)
    output_scale_offs = hidden_dim_block_index * 4 + tl.arange(0, 4)
    # input_ptr_offs = input_ptr + k_ptr_offs
    # output_ptr_offs = output_ptr + k_ptr_offs
    # output_scale_offs = output_scale_ptr + hidden_dim_block_index

    for token_index in tl.range(token_id,
                                token_num,
                                block_num,
                                num_stages=NUM_STAGE):

        act = tl.load(
            input_ptr + token_index * stride_input_0 + k_ptr_offs,
            mask= k_ptr_offs < size_k,
            other=0.0,
        )
        # act = tl.view(act, (4, 128))
        _absmax = tl.maximum(tl.max(tl.abs(act)), 1e-10)
        output_s = _absmax / fp8_max
        if hidden_dim_block_index == 0 and token_index == 0:
            tl.device_print("output_s:", output_s)
        # if SCALE_UE8M0:
        output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(act / output_s, fp8_min,
                            fp8_max).to(output_ptr.dtype.element_ty)

        # output_q = tl.view(output_q, (BLOCK))
        # output_s = tl.view(output_s, (4))
        tl.store(
            output_ptr + token_index * stride_output_0 + k_ptr_offs,
            output_q,
            mask=k_ptr_offs < size_k,
        )
        tl.store(
            output_scale_ptr + token_index * stride_output_scale_0 + hidden_dim_block_index,
            output_s,
        )



@triton.jit
def _per_token_quant_and_transform_kernel_1(
    input_ptr,
    stride_input_0,
    stride_input_1,
    output_ptr,
    stride_output_0,
    stride_output_1,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    m,
    k,
    fp8_max,
    fp8_min,
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):

    k_block_index = tl.program_id(0)
    m_block_index = tl.program_id(1)

    k_block_num = tl.num_programs(0)
    # m_block_num = tl.num_programs(1)


    offs_m = (m_block_index * m_block_size + tl.arange(0, m_block_size))
    offs_k = (k_block_index * k_block_size + tl.arange(0, k_block_size))

    input_ptrs = input_ptr + (offs_m[:, None] * stride_input_0 + offs_k [None, :] * stride_input_1)
    output_ptrs = output_ptr + (offs_m[:, None] * stride_output_0 + offs_k [None, :] * stride_output_1)

    offs_scale_m = (m_block_index * m_block_size + tl.arange(0, m_block_size))
    # offs_scale_k = (k_block_index + tl.arange(0, 1))
    # output_scale_ptrs = output_scale_ptr + (offs_scale_k[:, None] * stride_output_scale_0 + offs_scale_m [None, :] * stride_output_scale_1)
    output_scale_ptrs = output_scale_ptr + k_block_index * stride_output_scale_0 + offs_scale_m * stride_output_scale_1
   
    act = tl.load(input_ptrs, mask=(offs_k[None, :] < k) & (offs_m[:, None] < m), other=0.0).to(tl.float32)
    
    _absmax = tl.maximum(tl.max(tl.abs(act), axis=1), 1e-10)
    output_s = _absmax / fp8_max

    # if SCALE_UE8M0:
    output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
    output_s_1 = output_s.expand_dims(1)

    output_q = tl.clamp(act / output_s_1, fp8_min,
                        fp8_max).to(output_ptr.dtype.element_ty)

    tl.store(
        output_ptrs,
        output_q,
        mask=(offs_k[None, :] < k) & (offs_m[:, None] < m),
    )
    tl.store(
        output_scale_ptrs,
        output_s,
        mask=(offs_scale_m < m),
    )


def per_token_quant_and_transform(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    """

    assert input.is_contiguous()
    assert len(input.shape) == 2
    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    m, k = input.shape

    # Create output
    output = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")

    # Create output scale
    scale_k = ceil_div(k, quant_group_size)  # scale_k = k // 128
    # output_scale = torch.empty((m, scale_k),
    #                            dtype=torch.float32,
    #                            device='cuda')    # output_scale shape [k // 4, m]

    output_scale = torch.empty((scale_k, m),
                               dtype=torch.float32,
                               device='cuda')    # output_scale shape [k // 128, m]

    # Get block/grid/stage/warp
    # BLOCK_NUM = 4096

    k_block_size = quant_group_size # BLOCK = 128
    k_block_num = triton.cdiv(k, k_block_size)
    m_block_size = 128
    m_block_num = triton.cdiv(m, m_block_size)

    
    num_warps = 8
    NUM_STAGES = 1
    
    grid = (
        k_block_num,
        m_block_num,
        1,
    )
    _per_token_quant_and_transform_kernel_1[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        m,
        k,
        fp8_max,
        fp8_min,
        m_block_size=m_block_size,
        k_block_size=k_block_size,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    output_scale = output_scale.transpose(0, 1)
    return output, output_scale


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, 128)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / torch.tensor(448.0, dtype=torch.float32, device=x.device)
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape

    x_padded = torch.zeros((align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


if __name__ == "__main__":


    shape = (37800, 5120)
    
    # input = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 5

    # Benchmark
    times = []
    for _ in range(10):
        
        input = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 5
        # weight = torch.randn((5120, 5120), device='cuda', dtype=torch.bfloat16) * 5

        # weight_fp8, weight_sf = per_block_cast_to_fp8(weight, use_ue8m0=True)

        # gemm_output = torch.empty((37800, 5120), device='cuda', dtype=torch.bfloat16)

        output_ref, output_scale_ref = per_token_cast_to_fp8(input, use_ue8m0=True)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output, output_scale = per_token_quant_and_transform(input)
        # fp8_gemm_nt(
        #         (output, output_scale),
        #         (weight_fp8, weight_sf),       
        #         gemm_output,
        #         None, # bias
        #         disable_ue8m0_cast=False,
        #     )
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
        print("Elapsed time: ", elapsed_time)
        times.append(elapsed_time)


    print("diff: ", torch.max(torch.abs(output.to(torch.float32) - output_ref.to(torch.float32))))
    print("diff: ", torch.max(torch.abs(output_scale - output_scale_ref)))

    # print("output_ref:", output_ref)
    # print("output:", output)

    # print("output_scale_ref:", output_scale_ref)
    # print("output_scale:", output_scale)

    # times = np.array(times)
    # print("Mean time: ", np.mean(times),  "Min time: ", np.min(times))
