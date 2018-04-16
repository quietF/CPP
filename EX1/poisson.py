import numpy as np
import matplotlib.pyplot as plt
import sys

# Grid division
N  = 8
a  = 0.
b  = 2.*np.pi
L  = b - a
x  = np.linspace(a, b, N)

# Type of Boundary Conditions
boundaryConditions = ['Dirichlet','Mixed','Periodic']
boundaryCondition  = 'Dirichlet'
if boundaryCondition not in boundaryConditions:
  print("{0} not a suitable boundary condition.".format(boundaryCondition))
  sys.exit(1)

# initial conditions for density
def rho_0(x):
  return 2.*np.sin(x) + x*np.cos(x)
rho = rho_0(x)
h  = x[1] - x[0]
h2 = h*h

# analytical solution for potential
def phi_true(x):
  return x*np.cos(x)

# initial conditions for potential 
if boundaryCondition == 'Dirichlet':
  alpha             = 0.
  beta              = 2.*np.pi
  Nunknowns         = N-2
  RHS               = np.zeros(Nunknowns)
  RHS[:]            = rho[1:Nunknowns+1]
  RHS[0]            = rho[0] + alpha / h2
  RHS[Nunknowns-1]  = rho[Nunknowns+1] + beta / h2
  matrixSize        = N-2
elif boundaryCondition == 'Mixed':
  alpha             = 0.
  gamma             = 1.
  Nunknowns         = N-2
  RHS               = np.zeros(Nunknowns)
  RHS[:]            = rho[1:Nunknowns+1]
  RHS[0]            = rho[0] + alpha / (2.*h)
  RHS[np.size(x)-2] = gamma
  matrixSize        = N-1

# matrix A_ij phi(x_i) = rho(x_i)
a = np.zeros((matrixSize,matrixSize));     b = np.zeros((matrixSize,matrixSize))
np.fill_diagonal(a, -1./h2);               np.fill_diagonal(b, -1./h2)
a = np.roll(a, 1);                         b = np.roll(b, -1)
A_ij = a + b
np.fill_diagonal(A_ij, 2./h2)
np.set_printoptions(precision=3)
print(A_ij*h2)
print(np.linalg.eigvals(A_ij))

A_ij_inv   = np.linalg.inv(A_ij)
newPhi = np.dot(A_ij_inv, RHS)
newPhi = np.append((alpha), newPhi); newPhi = np.append(newPhi, (beta))

phiAnalytic = phi_true(x)

error = phiAnalytic[:]-newPhi[:]

plt.plot(x, newPhi, '-r', x, phi_true(x), '--g')#, x, rho, '-b')
plt.show()

errors_file = str("out/errors_{0}.dat".format(N))
output_err  = np.transpose(np.array((x, newPhi, phiAnalytic, error)))
np.savetxt(errors_file, output_err, fmt="%g")
