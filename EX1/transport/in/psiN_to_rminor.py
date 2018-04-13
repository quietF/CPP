import numpy as np
from scipy.interpolate import splev, splrep

central_mass    = 2.0
central_density = 0.76464
mass_proton     = 1.673e-27
n_0             = central_density * 1.e20

averaged0      = np.transpose(np.loadtxt("exprs_averaged_s00000.dat"))
input_profiles = np.transpose(np.loadtxt("input_profiles.dat"))

Psi_N_ip   = input_profiles[0]
Sne_ip     = input_profiles[7]
ST_ip      = input_profiles[8]

Psi_N_av   = averaged0[0]#[:len(averaged0[0])-10]
r_minor_av = averaged0[14]#[:len(averaged0[0])-10]

tck_Sne_ip = splrep(Psi_N_ip, Sne_ip, s=0)
tck_ST_ip  = splrep(Psi_N_ip, ST_ip,  s=0)

Sne_av     = splev(Psi_N_av, tck_Sne_ip, der=0)
ST_av      = splev(Psi_N_av, tck_ST_ip,  der=0)

outputSne = np.transpose(np.array((r_minor_av, Sne_av)))
outputST  = np.transpose(np.array((r_minor_av, ST_av )))

np.savetxt('Sne.dat', outputSne, fmt="%g")
np.savetxt('ST.dat',  outputST,  fmt="%g")

