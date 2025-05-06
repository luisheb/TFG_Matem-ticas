import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def compute_solution(n):
    h = 1 / n
    e = np.ones(n - 1)
    B = sp.diags([-e, 4 * e, -e], [-1, 0, 1], shape=(n - 1, n - 1))
    I = sp.eye(n - 1)
    I1 = sp.diags([-e, -e], [-1, 1], shape=(n - 1, n - 1))
    A = sp.kron(I, B) + sp.kron(I1, I)
    A /= h**2

    f = np.ones((n - 1) ** 2)
    y = spla.spsolve(A, f)

    val = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(n - 1):
            val[i, j] = y[j + (n - 1) * i]

    valnn = np.zeros((n + 1, n + 1))
    valnn[1:n, 1:n] = val
    return valnn

val_coarse = compute_solution(50)
val_fine = compute_solution(100)

x_coarse = np.linspace(0, 1, 51)
x_fine = np.linspace(0, 1, 101)

interp = RectBivariateSpline(x_fine, x_fine, val_fine)
val_fine_on_coarse = interp(x_coarse, x_coarse)

diff = val_fine_on_coarse - val_coarse
norm_diff = np.sqrt(np.sum(diff**2) / diff.size)

print("L2 norm of the difference:", norm_diff)
