import numpy as np
from numba import cuda
import math
import time

# CUDA kernel to compute Pi
@cuda.jit
def compute_pi_kernel(n, chunk_size, result):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    h = 1.0 / n
    local_sum = 0.0
    
    for i in range(idx * chunk_size + 1, (idx + 1) * chunk_size + 1):
        if i > n:
            break
        x = h * (i - 0.5)
        local_sum += 4.0 / (1.0 + x * x)
    
    # Write the local sum to the result array
    if idx < result.size:
        result[idx] = h * local_sum

def compute_pi_cuda(n, G, L):
    # Number of elements processed by each thread
    chunk_size = math.ceil(n / G)
    
    # Allocate memory for partial results
    result = np.zeros(G, dtype=np.float64)
    d_result = cuda.to_device(result)
    
    compute_pi_kernel[G, L](n, chunk_size, d_result)
    cuda.synchronize()
    
    # Copy results back to host
    result = d_result.copy_to_host()
    
    # Compute the final value of Pi by summing the partial results
    pi = np.sum(result)
    return pi

def main():
    N = 1 << 26  
    PI25DT = 3.141592653589793238462643
    
    # Grid and block sizes to test
    G_sizes = [2**i for i in range(10, 15)]  # global sizes
    L_sizes = [2**i for i in range(5, 8)]   # local sizes
    
    best_time = float('inf')
    best_G = None
    best_L = None
    
    for G in G_sizes:
        for L in L_sizes:
            print(f"Testing G: {G}, L: {L}")
            try:
                start_time = time.time()
                pi = compute_pi_cuda(N, G, L)
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if elapsed_time < best_time:
                    best_time = elapsed_time
                    best_G = G
                    best_L = L
                
                print(f"G: {G}, L: {L}, Elapsed time: {elapsed_time:.5f} seconds, Pi: {pi:.16f}, Error: {abs(pi - PI25DT):.16f}")
            except cuda.CudaAPIError as e:
                print(f"CUDA error with G: {G}, L: {L} - {e}")
    
    # Output for best configuration
    print(f"Best configuration: G: {best_G}, L: {best_L}, Best time: {best_time:.5f} seconds")
    
    # Calculate Pi using the best configuration
    pi = compute_pi_cuda(N, best_G, best_L)
    print(f"Calculated Pi: {pi:.16f}, Error: {abs(pi - PI25DT):.16f}")

if __name__ == "__main__":
    main()
