def is_prime(n: int) -> bool:
    """
    Return True if n is a prime number, else False.
    0 and 1 are *not* prime.
    """
    # TODO: replace pass with your implementation
    return isprime(n)
 
# Quick self-check
from sympy import *
print([x for x in range(10) if is_prime(x)])   # Expected: [2, 3, 5, 7]
