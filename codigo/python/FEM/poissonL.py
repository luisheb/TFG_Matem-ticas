from dolfin import *
import mshr  # Import mshr for geometry operations
import matplotlib.pyplot as plt

# Define the main square domain (-1,1) x (-1,1)
square = mshr.Rectangle(Point(-1, -1), Point(1, 1))

# Define the subdomain to remove: [0,1) x (-1,0]
cutout = mshr.Rectangle(Point(0, -1), Point(1, 0))

# Define the final domain by subtracting the cutout
domain = square - cutout

# Generate the mesh
mesh = mshr.generate_mesh(domain, 50)

# Define function space
V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition (u = 0 on the boundary)
bc = DirichletBC(V, Constant(0), "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Solve the problem
u = Function(V)
solve(a == L, u, bc)

# Plot solution
import matplotlib.tri as tri
import numpy as np

# Extract mesh coordinates
coordinates = mesh.coordinates()
triangles = np.array([cell.entities(0) for cell in cells(mesh)])

# Evaluate solution at mesh points
u_values = np.array([u(Point(x, y)) for x, y in coordinates])

# Create a triangulation
triang = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)

# Plot the solution using tricontourf (which supports colorbars)
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(triang, u_values, cmap="coolwarm")
plt.colorbar(contour, label="Solution u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution to Poisson Equation on Custom Domain")
plt.show()

