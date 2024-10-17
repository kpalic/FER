import math
import timeit
from compute_pi import compute_pi_cuda
import numba.cuda as cuda

def compute_pi_sequential(n):
    h = 1.0 / n
    sum = 0.0
    for i in range(1, n + 1):
        x = h * (i - 0.5)
        sum += 4.0 / (1.0 + x * x)
    pi = h * sum
    return pi

def main():
    N = 1 << 25  # 2^25
    PI25DT = 3.141592653589793238462643
    
    # Measure time for the sequential version
    start_time = timeit.default_timer()
    pi_sequential = compute_pi_sequential(N)
    end_time = timeit.default_timer()
    sequential_time = end_time - start_time
    print(f"Sequential Pi: {pi_sequential:.16f}, Error: {abs(pi_sequential - PI25DT):.16f}, Time: {sequential_time:.5f} seconds")
    
    # CUDA configuration
    G_sizes = [2**i for i in range(10, 15)]  # global sizes
    L_sizes = [2**i for i in range(5, 8)]    # local sizes
    
    best_time = float('inf')
    best_G = None
    best_L = None
    
    for G in G_sizes:
        for L in L_sizes:
            # print(f"Testing G: {G}, L: {L}")
            try:
                start_time = timeit.default_timer()
                pi = compute_pi_cuda(N, G, L)
                end_time = timeit.default_timer()
                elapsed_time = end_time - start_time
                
                if elapsed_time < best_time:
                    # print(f"New best time: {elapsed_time:.5f} seconds")
                    best_time = elapsed_time
                    best_G = G
                    best_L = L
                
                # Output for current configuration
                # print(f"G: {G}, L: {L}, Elapsed time: {elapsed_time:.5f} seconds, Pi: {pi:.16f}, Error: {abs(pi - PI25DT):.16f}")
            except cuda.CudaAPIError as e:
                print(f"CUDA error with G: {G}, L: {L} - {e}")
            except ValueError as e:
                print(f"Value error with G: {G}, L: {L} - {e}")
    
    # Output for best configuration
    print(f"Best configuration: G: {best_G}, L: {best_L}, Best time: {best_time:.5f} seconds")
    
    # Calculate Pi using the best configuration
    if best_G is not None and best_L is not None:
        pi = compute_pi_cuda(N, best_G, best_L)
        print(f"Calculated Pi: {pi:.16f}, Error: {abs(pi - PI25DT):.16f}")
        
        # Calculate speedup
        speedup = sequential_time / best_time
        print(f"Speedup: {speedup:.2f}")
    else:
        print("No valid configuration found.")

if __name__ == "__main__":
    main()
