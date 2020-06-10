#
# FORM FACTOR
# CROSS SECTION CALCULATION
# NUMBER OF EVENTS
#
# created 2020 June
# Leire Larizgoitia
#

import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 20})
import matplotlib.colors as mcolors
import numpy as np
import math
import scipy.integrate as integrate
import scipy.integrate as quad
import scipy.constants as constants
import pandas as pd
import random
from scipy.stats import norm
import matplotlib.mlab as mlab



"Constants"
c = constants.c # speed of light in vacuum
e = constants.e
fs = constants.femto # 1.e-15

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2
#Gf = Gf0 * (hbar*c)**3 * 1E-9 # Fermi constant in GeV.m^3

"Masses"
m_pion = 139.57018 #MeV/c^2
m_muon = 105.6583755 #MeV/c^2

"Xenon 132 isotope"
Z= 54
N = 78
M = 131.9041535 * constants.u # mass of Xenon 132 in kg

"RMS of Xenon"
Rn2 = (4.8864)**2
Rn4 = (5.2064)**4

"Recoil energy range of Xenon in KeV"
T_min= 0.9  # recoil energy of the mass nucluest
T_max = 27.
T_thres = 0.9 #THRESOLD RECOIL ENERGY FOR DETECTION

sigma0 = 0.4

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 16
Enu_max = 53

"Approximation values"
sin_theta_w_square = 0.23867 #zero momnetum transfer data from paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

nsteps = 100

"FUNCTIONS"
def F(Q2): # from factor of the nucleus evaluated at Q^2 = 2E^2TM/(E^2 − ET)
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    Fq = Fn / Qw
    return (Fq)

def F2(Q2): # from factor of the nucleus evaluated at Q^2 = 2E^2TM/(E^2 − ET)
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2) #approximation
    Fq = Fn / Qw
    return (Fq)

def QQ():
    return (4*(Z*gv_p + N*gv_n)**2)

def cross_section(T,Enu):
    Q2= 2 *Enu**2 * M/e  *c**2 * T *1E-9 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = (Gf*0.0389379)**2 /(2*np.pi) * QQ()**2 / 4 * F(Q2)**2 * M *c**2 * 1E-38/ e / hbar_c**2  * (2 - M * c**2 * T * 1E-9 / (e*Enu**2) )
    return dsigmadT

def flux(E,alpha):  #Fluxes following a continuous distribution. Normalized
    if (alpha == 0): #muon
        dirac = (m_pion**2 - m_muon**2) / (2*m_pion) #~29.7MeV
        if (E == dirac):
            f = 1.0
        else:
            f = 0.0
    if (alpha == 1): #anti-muon
        f = 64 / m_muon/c**2 * ((E / m_muon/c**2)**2 * (3/4 - E/m_muon/c**2))
    if (alpha == 2): #electron
        f = 192 / m_muon/c**2 * ((E / m_muon/c**2)**2 * (1/2 - E/m_muon/c**2))
    return f

int_mu =  np.zeros((nsteps+1),float)
int_antimu =  np.zeros((nsteps+1),float)
int_e = np.zeros((nsteps+1),float)
EE=   np.zeros((nsteps+1),float)

def differential_events(T,a):
    """Integral Bounds"""
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M*c**2/e *1E-3)) * 1E-3 #MeV
    Emax = m_muon*c**2/2
    if (a==0):
        return (cross_section(T, (m_pion**2 - m_muon**2) / (2*m_pion) ))
    if (a==1):
        for i in range (0,nsteps+1):
            EE[i] = Emin + (Emax - Emin)/nsteps * i
            int_antimu[i] = (cross_section(T,EE[i]) * flux(EE[i],1))
        return (np.trapz(int_antimu, x=EE))
    if (a==2):
        for i in range (0,nsteps+1):
            EE[i] = Emin + (Emax - Emin)/nsteps * i
            int_e[i] = (cross_section(T,EE[i]) * flux(EE[i],2))
        return (np.trapz(int_e, x=EE))

    #return (cross_section(T, (m_pion**2 - m_muon**2) / (2*m_pion) ) + np.trapz(int_antimu, x=EE) + np.trapz(int_e, x=EE)) *1e39

def variance(T):
    return sigma0 * np.sqrt(T/T_thres)

def minimum_value(sequence):
    """return the minimum element of sequence"""
    low = sequence[0] # need to start with some value
    for i in sequence:
        if i < low:
            low = i
    return low

def Plot_ff(x,f0,f1,f2):
    plt.plot(x,f0, color="cornflowerblue", linestyle='dashdot', label='Zero order approximation')
    plt.plot(x,f1, color="mediumseagreen", label='Forth order approximation')
    plt.plot(x,f2, color="tomato",  linestyle='dashdot', label='Second order approximation')
    plt.ylim((minimum_value(f2)))
    plt.xlim((minimum_value(x)))
    plt.xlabel(r'$Q (MeV)$')
    plt.ylabel(r'$F(Q)$')
    plt.legend()
    #plt.savefig("Plot form_factor.pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_cs(x,f):
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x,f, color="orange", label='Xe')
    plt.ylim((minimum_value(f)))
    plt.xlim((minimum_value(x)))
    plt.xlabel(r'$T(keV)$')
    plt.ylabel(r'$\frac{d\sigma}{dT} (\frac{cm^2}{keV})$')
    plt.legend()
    #plt.savefig("Plot cross section 40MeV.pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def best_fit(x,bins):
    # best fit of data
    (mu, sigma) = norm.fit(x)
    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    return plt.plot(bins, y, 'r--', linewidth=2)

#%%
"Main part"

T = []
T_rad = []

T_real = []
T_real_mu = []
T_real_antimu = []
T_real_e = []

dNdT =  []
dNdT_mu =  []
dNdT_antimu =  []
dNdT_e =  []

T_resolution = []
T_resolution_mu = []
T_resolution_anti = []
T_resolution_e = []


for i in range(0,nsteps+1):
    T.append(T_min + (T_max - T_min)/nsteps * i)
    dNdT_mu.append(1e6*differential_events(T_min + (T_max - T_min)/nsteps * i,0))
    dNdT_antimu.append(1e6*differential_events(T_min + (T_max - T_min)/nsteps * i,1))
    dNdT_e.append(1e6*differential_events(T_min + (T_max - T_min)/nsteps * i,2))

dNdT = dNdT_mu + dNdT_antimu + dNdT_e

for i in range(int(1e6)):
    T_random = random.uniform(T_min,T_max)
    T_rad.append(T_random)

    dNdT_random_mu = 1e6*differential_events(T_random,0)
    dNdT_random_antimu = 1e6* differential_events(T_random,1)
    dNdT_random_e = 1e6*differential_events(T_random,2)
    dNdT_random = dNdT_random_mu + dNdT_random_antimu + dNdT_random_e

    r = random.uniform(dNdT[0],dNdT[nsteps])
    r_mu = random.uniform(dNdT_mu[0],dNdT_mu[nsteps])
    r_antimu = random.uniform(dNdT_antimu[0],dNdT_antimu[nsteps])
    r_e = random.uniform(dNdT_e[0],dNdT_e[nsteps])

    if (dNdT_random_mu > r_mu):
        T_real_mu.append(T_random)
    if (dNdT_random_antimu > r_antimu):
        T_real_antimu.append(T_random)
    if (dNdT_random_e > r_e):
        T_real_e.append(T_random)
    if (dNdT_random > r):
        T_real.append(T_random)

for i in range(len(T_real)):
    T_resolution.append(random.gauss(T_real[i], sigma0 * np.sqrt(T_real[i]/T_thres)))

for i in range(len(T_real_mu)):
    T_resolution_mu.append(random.gauss(T_real_mu[i], sigma0 * np.sqrt(T_real_mu[i]/T_thres)))

for i in range(len(T_real_antimu)):
    T_resolution_anti.append(random.gauss(T_real_antimu[i], sigma0 * np.sqrt(T_real_antimu[i]/T_thres)))

for i in range(len(T_real_e)):
    T_resolution_e.append(random.gauss(T_real_e[i], sigma0 * np.sqrt(T_real_e[i]/T_thres)))


# the histogram of the data
n, bins, patches = plt.hist(T_resolution_mu, 50, density=1, label=r'$\nu_{\mu}$', color='olivedrab')# edgecolor='cornflowerblue')
n, bins, patches = plt.hist(T_resolution_anti, 50, density=1, label= r'$\bar{\nu}_{\mu}$', color='yellowgreen')#, edgecolor='orange')
n, bins, patches = plt.hist(T_resolution_e, 50, density=1, label=r'$\nu_{e}$', color='lightgreen')#, edgecolor='orange')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the data
n, bins, patches = plt.hist(T_resolution_mu, 50, density=1, label=r'$\nu_{\mu}$', color='olivedrab')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the data
n, bins, patches = plt.hist(T_resolution_anti, 50, density=1, label= r'$\bar{\nu}_{\mu}$', color='yellowgreen')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the data
n, bins, patches = plt.hist(T_resolution_e, 50, density=1, label=r'$\nu_{e}$', color='lightgreen')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

"PLOT MAKING"

"Events plot"



"Form factor plot"

#Plot_ff(xx,ff1,ff,ff2)

"Cross section plot"

#Plot_cs(tt,dodT)
