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
):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    # Enable spatial fusion
    # pid = ((pid - 1) + 8) % 8

    if pid > 0:
        # Early exit all other workgroups
        return

    tl.store(producer_xcd, xcd_id)

    offsets = tl.arange(0, NUM_ELEMENTS)
    mask = offsets < NUM_ELEMENTS 

    x = tl.load(buffer + offsets, mask=mask)

    for _ in range(500_000):
        # Do busy work
        x += x

    # tl.store(buffer + offsets, x+x, mask=mask, cache_modifier=".wt")
    tl.store(buffer + offsets, x, mask=mask, cache_modifier=".cg")

    # All work has been done, signal consumer.
    tl.debug_barrier()
    # tl.store(lock, 1, cache_modifier=".wt")
    tl.store(lock, 1, cache_modifier=".cg")

@triton.jit()
def consumer_kernel(
    buffer,
    output_buffer,
    lock,
    consumer_xcd,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, 

):
    pid = tl.program_id(0)
    xcd_id = read_xcd_id()

    if pid > 0:
        # Early exit all other workgroups
        return

    tl.store(consumer_xcd, xcd_id)

    while tl.load(lock, cache_modifier=".cv", volatile=True) != 1:
        pass

    # Lock acquired, start reading in tiles from producer.
    offsets = tl.arange(0, NUM_ELEMENTS)
    mask = offsets < NUM_ELEMENTS 

    x = tl.load(buffer + offsets, mask=mask)
    tl.store(output_buffer + offsets, x, mask=mask)