
import numpy as np
from MCCA import MCCA
from CCA import linCCA

N=100

D1=10
X1=np.random.rand(N,D1)

D2=15
X2=np.random.rand(N,D2)

# Number of components.
K=5

# Regularization parameters.
rcov1=0.0
rcov2=0.0

# 2-view case,
CORR, UU, MM, X_PROJ = MCCA(2, 1, [X1, X2], [rcov1, rcov2])
print("The canonical correlation + 1 from multi-view CCA:")
print(CORR)

# Benchmark with CCA.
A, B, m1, m2, E = linCCA(X1, X2, K, rcov1, rcov2)
print("The canonical correlation from 2-view CCA:")
print(E)

# The correlations shall be consistent, in the sense that CORR=E+1.
diffE = np.max(np.fabs(CORR[:K] - 1 - E[:K]))
print("Difference in canonical correlations: %f" % diffE)

# The projection matrices shall be consistent, up to a flip of sign.
for k in range(K):
    diffA = np.max(np.fabs(A[:, 0]) - np.fabs(UU[0][:, 0]))
    print("Difference in projection vector in dim %d: %f" % (k, diffA))

P1 = np.matmul(np.subtract(X1, m1),  A[:, :K])
P2 = np.matmul(np.subtract(X2, m2),  B[:, :K])
np.matmul(P1.transpose(), P2) / (N-1)

# More than 2 views.
D3=5
X3=np.random.rand(N,D3)

CORR, UU, MM, X_PROJ = MCCA(3, 1, [X1, X2, X3], [0.0, 0.0, 0.0])
print("\n3-view canonical correlations:")
print(CORR[:K]-1)