# Here's Python code to calculate Q and R using Gram-Schmidt:

# ```python
import numpy as np

# Define matrix A
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

# Gram-Schmidt process
def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            # Calculate R[i,j] using dot product
            R[i,j] = np.dot(Q[:, i], A[:, j])
            # Subtract projection
            v -= R[i,j] * Q[:, i]
        
        # Calculate R[j,j]
        R[j,j] = np.linalg.norm(v)
        # Normalize to get Q[:, j]
        Q[:, j] = v / R[j,j]
    
    return Q, R

Q, R = gram_schmidt(A)

print("Q = ")
print(Q)
print("\nR = ")
print(R)