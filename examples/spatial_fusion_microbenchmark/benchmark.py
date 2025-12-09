import torch
import torch.multiprocessing as mp
import triton
from spatial_fusion import producer_kernel, consumer_kernel

def main():
    datatype = torch.bfloat16
    block_size = 256
    num_tiles = 1
    buffer_size = block_size * num_tiles
    num_workgroups = 8
    num_tiles_per_wg = triton.cdiv(num_tiles, num_workgroups)
    num_trials = 10
    print(f"Buffer size: {buffer_size}, Num workgroups: {num_workgroups}, Tiles per WG: {num_tiles_per_wg}")
    print(f"Number of trials: {num_trials}")

    buffer = torch.rand(buffer_size, device="cuda", dtype=datatype)
    output_buffer = torch.zeros_like(buffer)
    flag = torch.zeros(1, device="cuda", dtype=torch.int32)
    flag.fill_(-1)
    producer_xcd = torch.zeros(1, device="cuda", dtype=torch.int32)
    producer_xcd.fill_(-1)
    consumer_xcd = torch.zeros(1, device="cuda", dtype=torch.int32)
    consumer_xcd.fill_(-1)

    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    reference_event = torch.cuda.Event(enable_timing=True)
    producer_start = torch.cuda.Event(enable_timing=True)
    producer_end = torch.cuda.Event(enable_timing=True)
    consumer_start = torch.cuda.Event(enable_timing=True)
    consumer_end = torch.cuda.Event(enable_timing=True)

    # Arrays to store timing results
    producer_durations = []
    consumer_durations = []
    overlap_durations = []
    total_durations = []

    def run_experiment():
        # Reset buffers
        flag.fill_(-1)
        producer_xcd.fill_(-1)
        consumer_xcd.fill_(-1)
        torch.cuda.synchronize()

        # Record reference time
        reference_event.record()

        # Launch producer kernel
        torch.cuda.nvtx.range_push("Producer")
        with torch.cuda.stream(producer_stream):
            producer_start.record()
            producer_kernel[(num_workgroups,)](
                buffer=buffer,
                lock=flag,
                producer_xcd=producer_xcd,
                NUM_ELEMENTS=buffer_size,
                BLOCK_SIZE=block_size
            )
            producer_end.record()
        torch.cuda.nvtx.range_pop()

        # Launch consumer kernel
        torch.cuda.nvtx.range_push("Consumer")
        with torch.cuda.stream(consumer_stream):
            consumer_start.record()
            consumer_kernel[(num_workgroups,)](
                buffer=buffer,
                output_buffer=output_buffer,
                lock=flag,
                consumer_xcd=consumer_xcd,
                NUM_ELEMENTS=buffer_size,
                BLOCK_SIZE=block_size
            )
            consumer_end.record()
        torch.cuda.nvtx.range_pop()

        # Wait for both kernels to complete
        torch.cuda.synchronize()

        # Calculate absolute times relative to reference
        prod_start_ms = reference_event.elapsed_time(producer_start)
        prod_end_ms = reference_event.elapsed_time(producer_end)
        cons_start_ms = reference_event.elapsed_time(consumer_start)
        cons_end_ms = reference_event.elapsed_time(consumer_end)

        prod_xcd = producer_xcd.item()
        cons_xcd = consumer_xcd.item()
        print(f"Producer XCD: {prod_xcd}, Consumer XCD: {cons_xcd}")

        # Print timing information
        print(f"\n{'='*60}")
        print(f"Kernel Timing Information")
        print(f"{'='*60}")
        print(f"Producer:  start={prod_start_ms:.3f}ms, end={prod_end_ms:.3f}ms, duration={prod_end_ms-prod_start_ms:.3f}ms")
        print(f"Consumer:  start={cons_start_ms:.3f}ms, end={cons_end_ms:.3f}ms, duration={cons_end_ms-cons_start_ms:.3f}ms")

        # Check for overlap
        overlap_start = max(prod_start_ms, cons_start_ms)
        overlap_end = min(prod_end_ms, cons_end_ms)
        overlap_ms = max(0, overlap_end - overlap_start)
        total_time = max(prod_end_ms, cons_end_ms) - min(prod_start_ms, cons_start_ms)

        # Store timing results
        producer_durations.append(prod_end_ms - prod_start_ms)
        consumer_durations.append(cons_end_ms - cons_start_ms)
        overlap_durations.append(overlap_ms)
        total_durations.append(total_time)

        print(f"\nOverlap:   {overlap_ms:.3f}ms")
        print(f"Total:     {total_time:.3f}ms")
        if prod_end_ms - prod_start_ms > 0:
            print(f"Overlap %: {overlap_ms/(prod_end_ms-prod_start_ms)*100:.1f}% of producer")
        print(f"{'='*60}\n")

    # Warmup
    print("Warmup...")
    run_experiment()

    # Clear warmup results from timing arrays
    producer_durations.clear()
    consumer_durations.clear()
    overlap_durations.clear()
    total_durations.clear()

    # Benchmark trials
    print(f"\nRunning {num_trials} benchmark trials...")
    for trial in range(num_trials):
        # if (trial + 1) % 10 == 0:
        #     print(f"Completed {trial + 1}/{num_trials} trials")
        run_experiment()

    # Calculate and print average statistics
    import statistics
    
    avg_producer = statistics.mean(producer_durations)
    avg_consumer = statistics.mean(consumer_durations)
    avg_overlap = statistics.mean(overlap_durations)
    avg_total = statistics.mean(total_durations)
    
    std_producer = statistics.stdev(producer_durations) if len(producer_durations) > 1 else 0
    std_consumer = statistics.stdev(consumer_durations) if len(consumer_durations) > 1 else 0
    std_overlap = statistics.stdev(overlap_durations) if len(overlap_durations) > 1 else 0
    std_total = statistics.stdev(total_durations) if len(total_durations) > 1 else 0

    print(f"\n{'='*60}")
    print(f"Average Results over {num_trials} trials")
    print(f"{'='*60}")
    print(f"Producer:  {avg_producer:.3f} ± {std_producer:.3f} ms")
    print(f"Consumer:  {avg_consumer:.3f} ± {std_consumer:.3f} ms")
    print(f"Overlap:   {avg_overlap:.3f} ± {std_overlap:.3f} ms")
    print(f"Total:     {avg_total:.3f} ± {std_total:.3f} ms")
    if avg_producer > 0:
        print(f"Overlap %: {avg_overlap/avg_producer*100:.1f}% of producer")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()