import numpy as np
from sympy import Matrix, N
import sympy

n = 3
k = 128
a = 574389759345345

A = np.eye(n, dtype=int).tolist()

for i in range(n):
    A[i][i] = 2 ** k
A[0][0] = 1
for i in range(1, n):
    A[0][i] = A[0][i - 1] * a

print(A)

m = Matrix(A)
print(m.eigenvals())
for k, v in m.eigenvals().items():
    print(k)
    print(N(sympy.log(k, 2)))
    break
