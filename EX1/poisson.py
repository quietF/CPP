import numpy as np
import matplotlib.pyplot as plt

# Grid division
N  = 32
a  = 0.
b  = 2.*np.pi
L  = b - a
x  = np.linspace(a, b, N)

# initial conditions for density
def rho_0(x):
  return 2.*np.sin(x) + x*np.cos(x)
rho = rho_0(x)
h  = x[1] - x[0]
h2 = h*h

# initial conditions for potential 
alpha = 0.
beta  = 2.*np.pi

Rho               = np.zeros(np.size(x))
Rho[:]            = rho[:]
Rho[0]            = rho[0] + alpha / h2
Rho[np.size(x)-1] = rho[np.size(x)-1] + beta / h2

# analytical solution for potential
def phi_true(x):
  return x*np.cos(x)

# matrix A_ij phi(x_i) = rho(x_i)
a = np.zeros((N,N));         b = np.zeros((N,N))
np.fill_diagonal(a, -1./h2); np.fill_diagonal(b, -1./h2)
a = np.roll(a, 1);           b = np.roll(b, -1)
A_ij = a + b
np.fill_diagonal(A_ij, 2./h2)
#print(np.linalg.eigvals(T))

A_ij_inv   = np.linalg.inv(A_ij)
newPhi = np.dot(A_ij_inv, Rho)

plt.plot(x, newPhi, '-r', x, phi_true(x), '--g', x, rho, '-b')
plt.show()
