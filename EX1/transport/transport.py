import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

# Define r_minor range (axis to separatrix)
r_axis       = 0.001
r_separatrix = 0.622

# Grid division
N            = 2**10
r            = np.linspace(r_axis, r_separatrix, N+1)
h            = r[1] - r[0]

# Initial conditions
chi_0        = 3.6e-7 
dchiN_dr     = 0.

# Import profiles
def extract_profile_from_file(profile_file): 
  ''' Turns input profiles to usable profiles.
        
        profile_file -- string with profile path (profile 
                        must be in a file with two columns
                        r_minor vs. profile.
        
        return       -- Usable profile.
                     -- Profile's gradient w.r.t r_minor'''
  try:
    profile_data = np.transpose(np.loadtxt(profile_file))
  except FileNotFoundError:
    print("{0} file not found.".format(profile_file))
  r_initial       = profile_data[0]
  profile_initial = profile_data[1]
  tck_profile     = splrep(r_initial, profile_initial, s=0)
  profile_new     = splev(r, tck_profile, der=0)
  dprofile_newdr  = np.gradient(profile_new, h)

  return profile_new, dprofile_newdr, np.gradient(dprofile_newdr, h)

T_dTdr     = extract_profile_from_file("T_preELM.dat")
T          = T_dTdr[0];   dT_dr      = T_dTdr[1];   d2T_dr2    = T_dTdr[2]

ne_dnedr   = extract_profile_from_file("ne_preELM.dat")
ne         = ne_dnedr[0]; dne_dr     = ne_dnedr[1]; d2ne_dr2   = ne_dnedr[2]

ST_dSTdr   = extract_profile_from_file("ST_preELM.dat")
ST         = ST_dSTdr[0]

Sne_dSnedr = extract_profile_from_file("Sne_preELM.dat")
Sne        = Sne_dSnedr[0]

# Define the A_ij Matrix that solves A_ij chi_j = - ST_i'
X_j = np.zeros(N); X_j = dT_dr[1:]*dne_dr[1:] + ne[1:]*dT_dr[1:]/r[1:] + ne[1:]*d2T_dr2[1:]
X_0 = dT_dr[0]*dne_dr[0] + ne[0]*dT_dr[0]/r[0] + ne[0]*d2T_dr2[0]
print(X_0)
Y_j = np.zeros(N); Y_j = ne[1:]*dT_dr[1:]/(2.*h)
delta_ij   = np.zeros((N,N)); np.fill_diagonal(delta_ij, 1)
delta_NN   = np.zeros((N,N)); delta_NN[N-1][N-1]   = 1
X_ij       = (delta_ij-delta_NN)*X_j
Y_ij_1     = (delta_ij-delta_NN)*Y_j; Y_ij_1 = np.roll(Y_ij_1, -1); Y_ij_1[N-1][N-1] = 0
Y_ij_2     = (delta_ij-delta_NN)*Y_j; Y_ij_2 = np.roll(Y_ij_2, 1)
Y_ij       = Y_ij_1-Y_ij_2 
A_ij       = X_ij + Y_ij; A_ij[N-1][N-2] = -1./h; A_ij[N-1][N-1] = 1./h

# Define the ST_i' vector
ST_i_prime      = np.zeros(N)
ST_i_prime      = ST[1:]
ST_i_prime[0]   = ST_i_prime[0] - ne[1]*dT_dr[1]*chi_0/(2.*h)
ST_i_prime[N-1] = dchiN_dr

A_ij_inv   = np.linalg.inv(A_ij)
chi_j      = np.dot(A_ij_inv, -ST_i_prime)

#chi        = np.append((chi_0), chi_j)
#plt.plot(r, chi)
plt.plot(r[1:], chi_j)
plt.show()
