import numpy as np
from numba import cuda
from math import sqrt
import time

@cuda.jit
def count_primes_atomic(numbers, result, chunk_size):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start * chunk_size, min((start + 1) * chunk_size, numbers.size)):
        num = numbers[i]
        is_prime = 1
        if num <= 1:
            is_prime = 0
        else:
            sqrt_num = int(sqrt(num))
            for j in range(2, sqrt_num + 1):
                if num % j == 0:
                    is_prime = 0
                    break
        if is_prime:
            cuda.atomic.add(result, 0, 1)

def main():
    # Variables
    N = 1 << 25  
    numbers = np.arange(1, N + 1, dtype=np.int32)
    result = np.zeros(1, dtype=np.int32)
    
    d_numbers = cuda.to_device(numbers)
    d_result = cuda.to_device(result)
    
    start_time = time.time()
    
    G = 1024  # global work size
    L = 128   # local work size
    chunk_size = N // G  # Size of the chunk each thread will process
    
    count_primes_atomic[G, L](d_numbers, d_result, chunk_size)
    cuda.synchronize()
    
    end_time = time.time()
    
    result = d_result.copy_to_host()
    
    print(f"Total number of prime numbers: {result[0]}")
    print(f"Elapsed time: {end_time - start_time:.5f} seconds")

if __name__ == "__main__":
    main()
