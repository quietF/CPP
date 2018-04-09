import numpy as np
import matplotlib.pyplot as plt

# Grid division
N  = 8
a  = 0.
b  = 2.*np.pi
L  = b - a
x  = np.linspace(a, b, N)

# initial conditions for density
def rho_0(x):
  return 2.*np.sin(x) + x*np.cos(x)
rho = rho_0(x)

# initial conditions for potential 
phi                 = np.zeros(np.size(x))
phi[0]              = 0.
phi[np.size(phi)-1] = 2.*np.pi

#plt.plot(x, rho, '-r', x, phi, '-b')
#plt.show()

# analytical solution for potential
def phi_true(x):
  return x*np.cos(x)

a = np.zeros((N,N));     b = np.zeros((N,N))
np.fill_diagonal(a, -1); np.fill_diagonal(b, -1)
a = np.roll(a, 1);       b = np.roll(b, -1)

T = a + b
np.fill_diagonal(T, 2)
print(np.linalg.eigvals(T))


#dphi                   = np.zeros(np.size(x))
#dphi[1:np.size(phi)-1] = (phi[2:np.size(phi)] - 2.*phi[1:np.size(phi)-1] + phi[0:np.size(phi)-2]) / (x[1] - x[0])
#phi                    = phi + dphi

#print(phi)
