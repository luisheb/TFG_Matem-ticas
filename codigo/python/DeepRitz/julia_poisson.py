import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

n = 50
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

xx = np.linspace(0, 1, n + 1)
yy = xx
X, Y = np.meshgrid(xx, yy)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, valnn, cmap="viridis")
ax1.set_title("Solution Mesh")

ax2 = fig.add_subplot(122)
c = ax2.contourf(X, Y, valnn, cmap="viridis")
plt.colorbar(c, ax=ax2)
ax2.set_title("Contour Plot")

plt.show()
