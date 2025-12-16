import triton
import triton.language as tl
from examples.common.utils import read_xcd_id


@triton.jit()
def producer_kernel(
    buffer,
    lock,
    producer_xcd,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_SPATIAL_FUSION: tl.constexpr,
    barrier, 
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # Enable spatial fusion
    if not ENABLE_SPATIAL_FUSION:
        pid = ((pid - 1) + 8) % 8
        # pid = pid

    if pid > 0:
        # Early exit all other workgroups
        return
    

    tl.store(producer_xcd, xcd_id)

    # Wait for both kernels to be ready
    tl.atomic_add(barrier, 1)
    tl.debug_barrier()
    # while tl.atomic_cas(barrier, 2, 2) != 2:
    #     pass
    while tl.load(barrier, volatile=True) < 2:
        pass
    tl.debug_barrier()

    num_tiles = tl.cdiv(NUM_ELEMENTS, BLOCK_SIZE)
    for tile_id in range(num_tiles):
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        offsets = tl.multiple_of(offsets, BLOCK_SIZE)
        mask = offsets < NUM_ELEMENTS 

        x = tl.load(buffer + offsets, mask=mask)

        # tl.store(buffer + offsets, x+x, mask=mask, cache_modifier=".wt")
        tl.store(buffer + offsets, x, mask=mask, cache_modifier=".cg", eviction_policy="evict_last")
        # tl.store(buffer + offsets, x+x, mask=mask)

        tl.debug_barrier()

        # tl.store(lock + tile_id, 1, cache_modifier=".wt")
        # tl.store(lock + tile_id, 1, cache_modifier=".cg")
        # tl.store(lock + tile_id, 1)
        tl.atomic_xchg(lock + tile_id, 1)

@triton.jit()
def consumer_kernel(
    buffer,
    output_buffer,
    lock,
    consumer_xcd,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_SPATIAL_FUSION: tl.constexpr,
    barrier
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # Enable spatial fusion
    # if ENABLE_SPATIAL_FUSION:
    #     pid = ((pid - 1) + 8) % 8

    if pid > 0:
        # Early exit all other workgroups
        return

    tl.store(consumer_xcd, xcd_id)


    # Wait for both kernels to be ready
    tl.atomic_add(barrier, 1)
    tl.debug_barrier()
    while tl.atomic_cas(barrier, 2, 2) != 2:
        pass
    tl.debug_barrier()

    num_tiles = tl.cdiv(NUM_ELEMENTS, BLOCK_SIZE)

    for tile_id in range(num_tiles):
        # while tl.load(lock + tile_id, cache_modifier=".cv", volatile=True) != 1:
        #     pass

        while tl.atomic_cas(lock + tile_id, 1, 1) != 1:
            pass

        # Lock acquired, read tile.
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        offsets = tl.multiple_of(offsets, BLOCK_SIZE)
        mask = offsets < NUM_ELEMENTS 

        x = tl.load(buffer + offsets, mask=mask)
        tl.store(output_buffer + offsets, x, mask=mask)


@triton.jit()
def workgroup_specialized_kernel(
    buffer,
    output_buffer,
    lock,
    producer_xcd,
    consumer_xcd,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_SPATIAL_FUSION: tl.constexpr,
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # Determine producer and consumer pids based on spatial fusion flag
    if ENABLE_SPATIAL_FUSION:
        producer_pid = 0
        consumer_pid = 8
    else:
        producer_pid = 0
        consumer_pid = 1

    # Early exit for all pids except producer and consumer
    if pid != producer_pid and pid != consumer_pid:
        return

    # Producer logic
    if pid == producer_pid:
        tl.store(producer_xcd, xcd_id)

        num_tiles = tl.cdiv(NUM_ELEMENTS, BLOCK_SIZE)
        for tile_id in range(num_tiles):
            block_start = tile_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            offsets = tl.multiple_of(offsets, BLOCK_SIZE)
            mask = offsets < NUM_ELEMENTS 

            x = tl.load(buffer + offsets, mask=mask)

            tl.store(buffer + offsets, x, mask=mask, cache_modifier=".cg")

            tl.debug_barrier()

            tl.atomic_xchg(lock + tile_id, 1)

    # Consumer logic
    elif pid == consumer_pid:
        tl.store(consumer_xcd, xcd_id)

        num_tiles = tl.cdiv(NUM_ELEMENTS, BLOCK_SIZE)

        for tile_id in range(num_tiles):
            while tl.atomic_cas(lock + tile_id, 1, 1) != 1:
                pass

            # Lock acquired, read tile.
            block_start = tile_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            offsets = tl.multiple_of(offsets, BLOCK_SIZE)
            mask = offsets < NUM_ELEMENTS 

            x = tl.load(buffer + offsets, mask=mask)
            tl.store(output_buffer + offsets, x, mask=mask)