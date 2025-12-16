# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime, read_xcd_id

import sys
import os

import iris


# Layout mapping tile/GEMM WG IDs to the output grid to improve L2/LLC cache locality.
Layout_GEMM

# Layout encoding the dependencies between COMM and GEMM workgroups. 
# This is a simple column major layout of height S where S is the ratio of GEMM WGs to COMM WGs.
Layout_COMM_to_GEMM

@triton.jit()
def persistent_gemm_tessera(
    A,
    B,
    C,
    bias_ptr,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
    SHOW_MAP: tl.constexpr = False,
    gemm_map_xcd: tl.tensor = None,
    gemm_xcd_flag: tl.tensor = None,
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # Chiplet transform
    pid = (pid % NUM_XCDS) * (GEMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    if COLLECT_TIMESTAMPS:
        timestamp = read_realtime()
        tl.atomic_min(mm_begin_timestamp_ptr + pid, timestamp)

    # Tiling remap.
    pid_m, pid_n = Layout_GEMM(pid)
    if SHOW_MAP:
        tl.store(gemm_map_xcd + (pid_m * num_pid_n + pid_n), xcd_id)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
        b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b)

    # Accumulator registers with C results
    c = acc.to(C.type.element_ty)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # Add compiler hints
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
    sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Calculate the "global" offset of C based on the rank.
    # Note how the N-dimension is being multiplied by current rank.
    # This is because each rank is computing a portion of the N-dimension
    # locally and then scattering it to all other ranks to complete
    # the global N-dimension.
    global_offset = rm[:, None] * stride_cm + (rn[None, :] + cur_rank * N) * stride_cn

    # Timestamp for GEMM before store
    if COLLECT_TIMESTAMPS:
        timestamp = read_realtime()
        tl.atomic_max(mm_end_timestamp_ptr + pid, timestamp)

    # tl.store(C + global_offset, c, mask=sub_mask, cache_modifier=".wt")
    tl.store(C + global_offset, c, mask=sub_mask)
    tl.debug_barrier()
    # tl.store(locks + tile_id, 1, cache_modifier=".wt")
    tl.store(locks + pid, 1)


@triton.jit()
def persistent_all_scatter_tessera(
    C,
    locks,
    M,
    N,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
    SHOW_MAP: tl.constexpr = False,
    comm_map_xcd: tl.tensor = None,
    gemm_xcd_flag: tl.tensor = None,
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # This is a hard-coded transform to ensure that pid 0 of both kernels start on the same XCD.
    # Right now this is hard-coded assuming PID 0 of the GEMM kernel starts on XCD 1 and we start on XCD 7,
    # but we can dynmically determine this modular shift if needed via a flag communicating between the kernels.
    pid = ((pid - 1) + COMM_SMS) % COMM_SMS

    # Identical chiplet transform as in the GEMM kernel, but with a different total number of WGs.
    pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Each COMM pid is dependent on the output of multiple GEMM pids.
    # The dependnecy layout encodes this relationship.
    for gemm_id in Layout_COMM_to_GEMM(pid): # Returns a sub-layout that we can iterate over.

        # gemm_id can be treated simply as a pid from the GEMM kernel,
        # therefore we can re-use the identical layouts from the GEMM kernel here
        # to get the mappings we need.
        pid_m, pid_n = Layout_GEMM(gemm_id)

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        if SHOW_MAP:
            tl.store(comm_map_xcd + (pid_m * num_pid_n + pid_n), xcd_id)

        # Begin: See the if segment for explanation:
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)
        global_offset = rm[:, None] * stride_cm_global + (rn[None, :] + cur_rank * N) * stride_cn_global
        # End: masks/offset calculations.

        # while tl.load(locks + tile_id, cache_modifier=".cv", volatile=True) != 1:
        #     pass
        while tl.load(locks + tile_id) != 1:
            pass

        for remote_rank in range(world_size):
            if remote_rank != cur_rank:
                iris.put(C + global_offset, C + global_offset, cur_rank, remote_rank, heap_bases, mask=sub_mask)
