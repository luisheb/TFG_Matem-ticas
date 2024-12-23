import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

def pde(x, u):
    A = 0.5 * tf.cast(tf.less(x, 0), tf.float32) + 1 * tf.cast(tf.greater_equal(x, 0), tf.float32) 
    du_x = dde.grad.jacobian(u, x) 
    dA_du_x = dde.grad.jacobian(A * du_x, x) 
    f = 0 * tf.cast(tf.less(x, 0), tf.float32) + (-2) * tf.cast(tf.greater_equal(x, 0), tf.float32) 
    return -dA_du_x - f

def weak_solution(x):
    return (-2/3 * x - 2/3) * (x < 0) + (x**2 - (1/3) * x - (2/3)) * (x >= 0)

def boundary(x, on_boundary):
    return on_boundary

geom = dde.geometry.Interval(-1, 1)
bc = dde.DirichletBC(geom, weak_solution, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=2, num_test=100, solution=weak_solution)

net = dde.maps.FNN([1] + [256] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=0.001,metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Residuo de la EDP")
plt.show()


