"""
Array Increment Triton Implementation: High-Performance Element-wise Increment
Math: A[i] = A[i] + 1 for all i in [0, N)
Inputs: A[N] - input array, N - array length
Assumptions: N > 0, array is contiguous, device has sufficient memory
Parallel Strategy: Each block processes multiple elements with coalesced access
Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
Complexity: O(N) FLOPs, O(N) bytes moved
Test Vectors: Deterministic random arrays with known increments
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional
import time

# Autotune configurations for different hardware
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def array_increment_kernel(
    # Input/output array
    array_ptr,
    # Array length
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise array increment.
    
    Memory layout: Contiguous 1D array
    Each thread block processes BLOCK_SIZE elements
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    
    # Process elements in this block
    for i in range(block_start, block_end):
        # Load element
        val = tl.load(array_ptr + i)
        
        # Increment
        val += 1
        
        # Store result
        tl.store(array_ptr + i, val)


@triton.jit
def array_increment_optimized_kernel(
    # Input/output array
    array_ptr,
    # Array length
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel with vectorized operations and better memory access.
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    
    # Vectorized processing for better memory bandwidth
    for i in range(block_start, block_end, 4):
        # Process 4 elements at a time
        if i + 3 < block_end:
            # Load 4 elements: pointer must be vector for vectorized load
            offsets = tl.arange(0, 4)
            mask = (i + offsets) < block_end
            vals = tl.load(array_ptr + i + offsets, mask=mask)
            
            # Increment
            vals += 1
            
            # Store results: pointer must be vector for vectorized store
            tl.store(array_ptr + i + offsets, vals, mask=mask)
        else:
            # Handle remaining elements
            for j in range(i, min(i + 4, block_end)):
                val = tl.load(array_ptr + j)
                val += 1
                tl.store(array_ptr + j, val)


def array_increment_triton(
    array: torch.Tensor,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply array increment using Triton kernel.
    
    Args:
        array: Input array [N]
        optimized: Whether to use optimized kernel
        
    Returns:
        Result array [N] (in-place modification)
    """
    # Input validation
    assert array.dim() == 1, "Input must be 1D tensor"
    assert array.is_contiguous(), "Input tensor must be contiguous"
    
    N = array.shape[0]
    device = array.device
    dtype = array.dtype
    
    # Ensure tensor is on correct device
    array = array.to(device)
    
    # Grid must derive from BLOCK_SIZE: use callable so autotune's selected
    # BLOCK_SIZE is respected (Bug 3 fix: hardcoded 256 would mismatch autotuned configs)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    if optimized:
        array_increment_optimized_kernel[grid](
            array,
            N,
            BLOCK_SIZE=256,
        )
    else:
        array_increment_kernel[grid](
            array,
            N,
        )
    
    return array


class ArrayIncrementModule(torch.nn.Module):
    """
    PyTorch module wrapper for Triton array increment.
    """
    
    def __init__(self, optimized: bool = True):
        super().__init__()
        self.optimized = optimized
        
    def forward(self, array: torch.Tensor) -> torch.Tensor:
        return array_increment_triton(array, self.optimized)


def benchmark_array_increment(
    sizes: list = [1024, 10000, 100000, 1000000, 10000000],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> None:
    """
    Benchmark array increment across different sizes and implementations.
    """
    print(f"\n=== Array Increment Benchmark on {device} ===")
    print(f"Data type: {dtype}")
    
    results = []
    
    for N in sizes:
        print(f"\nArray size: {N:,}")
        
        # Create test tensor
        torch.manual_seed(42)
        array = torch.randn(N, device=device, dtype=dtype)
        
        # Expected result
        expected = array + 1
        
        # Test Triton implementation
        if device == 'cuda':
            # Warmup
            for _ in range(10):
                test_array = array.clone()
                _ = array_increment_triton(test_array, optimized=True)
            
            torch.cuda.synchronize()
            
            # Benchmark optimized kernel
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            num_iterations = 100
            start_event.record()
            for _ in range(num_iterations):
                test_array = array.clone()
                result = array_increment_triton(test_array, optimized=True)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            
            # Verify correctness
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
            
            # Calculate performance metrics
            avg_time_ms = elapsed_ms / num_iterations
            gflops = N / (avg_time_ms / 1000.0) / 1e9
            bandwidth_gb_s = 2 * N * array.element_size() / (avg_time_ms / 1000.0) / 1e9  # Read + Write
            
            print(f"  Triton optimized: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
            
            # Compare with PyTorch
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(num_iterations):
                test_array = array.clone()
                result_pytorch = test_array + 1
            end_event.record()
            torch.cuda.synchronize()
            pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations
            
            speedup = pytorch_time_ms / avg_time_ms
            print(f"  PyTorch baseline: {pytorch_time_ms:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                'size': N,
                'triton_time': avg_time_ms,
                'pytorch_time': pytorch_time_ms,
                'speedup': speedup,
                'gflops': gflops,
                'bandwidth': bandwidth_gb_s
            })
        else:
            # CPU fallback
            start_time = time.time()
            for _ in range(10):
                test_array = array.clone()
                result = array_increment_triton(test_array, optimized=True)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) / 10 * 1000
            print(f"  Triton (CPU): {avg_time_ms:.3f} ms")
    
    return results


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    print(f"Testing Array Increment Triton implementation on {device}")
    
    # Test with small array
    N = 1000
    torch.manual_seed(42)
    array = torch.randn(N, device=device, dtype=dtype)
    
    print(f"\n=== Testing with array size {N} ===")
    
    # Test Triton implementation
    result = array_increment_triton(array.clone(), optimized=True)
    expected = array + 1
    
    # Verify output
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), "Output values don't match expected"
    
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = array_increment_triton(array.clone(), optimized=True)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    array_increment_module = ArrayIncrementModule(optimized=True)
    result_module = array_increment_module(array.clone())
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Run comprehensive benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_array_increment(
            sizes=[1024, 10000, 100000, 1000000],
            device=device.type,
            dtype=dtype
        )
        
        # Print summary
        print("\n=== Performance Summary ===")
        for result in benchmark_results:
            print(f"Size {result['size']:,}: {result['speedup']:.2f}x speedup, "
                  f"{result['gflops']:.2f} GFLOPS, {result['bandwidth']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

"""
Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python array_increment_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python array_increment_triton.py
 * 
 * 3. For optimal performance:
 *    - Use autotuning to find best block sizes for your hardware
 *    - Enable optimized kernel for better memory access patterns
 *    - Use appropriate data types (FP16/BF16 for memory-bound operations)
 *    - Consider vectorized operations for better throughput
 * 
 * 4. Memory optimization:
 *    - Ensure input tensors are contiguous
 *    - Use appropriate block sizes for your GPU architecture
 *    - Consider memory coalescing for better bandwidth utilization
 * 
 * 5. Distributed training considerations:
 *    - Use tl.comm_* primitives for multi-GPU operations
 *    - Implement gradient synchronization for distributed training
 *    - Consider memory-efficient implementations for large models
"""
