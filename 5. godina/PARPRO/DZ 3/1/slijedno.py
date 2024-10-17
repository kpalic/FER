import numpy as np
from math import sqrt
import time

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def main():
    N = 1 << 23 
    numbers = np.arange(1, N + 1, dtype=np.int32)
    
    start_time = time.time()
    
    total_primes = 0
    for num in numbers:
        if is_prime(num):
            total_primes += 1
    
    end_time = time.time()
    
    # Print results
    print(f"Total number of prime numbers: {total_primes}")
    print(f"Elapsed time: {end_time - start_time:.5f} seconds")

if __name__ == "__main__":
    main()
