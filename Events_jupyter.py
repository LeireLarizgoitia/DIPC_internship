#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
e = constants.e #elementary charge
fs = constants.femto # 1.e-15

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2

"Mass"
m_pion = 139.57018 #MeV/c^2
m_muon = 105.6583755 #MeV/c^2

"Xenon 132 isotope"
Z= 54
N = 78
M = 131.9041535 * constants.u # mass of Xenon 132 in kg

"RMS of Xenon"
Rn2 = (4.8864)**2
Rn4 = (5.2064)**4

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = m_muon/2 #~52.8MeV
Enu_mu = (m_pion**2 - m_muon**2) / (2*m_pion) #~29.8MeV

"Recoil energy range of Xenon in KeV"
T_min= 0.9
T_max = 2*(Enu_max)**2*e*1E9/M/c**2
T_thres = 0.9 #THRESOLD RECOIL ENERGY FOR DETECTION

sigma0 = 0.4 #corresponds to T_thresold

"Approximation values"
sin_theta_w_square = 0.23867 #zero momnetum transfer data from the paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

nsteps = 100

def F(Q2):  # from factor of the nucleus
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    Fq = Fn / Qw
    return (Fq)

def F2(Q2): # from factor of the nucleus
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
    if (alpha == 1): #anti-muon
        f = 64 / m_muon * ((E / m_muon)**2 * (3/4 - E/m_muon))
    if (alpha == 2): #electron
        f = 192 / m_muon * ((E / m_muon)**2 * (1/2 - E/m_muon))
    return f


# In[23]:


int_mu =  np.zeros((nsteps+1),float)
int_antimu =  np.zeros((nsteps+1),float)
int_e = np.zeros((nsteps+1),float)
EE_antimu= np.zeros((nsteps+1),float)
EE_e= np.zeros((nsteps+1),float)

def differential_events(T,a):
    """Integral Bounds"""
    Emin = np.sqrt(T*M*c**2/e /2 *1E-9) #1/2 * (T + np.sqrt(T**2 + 2*T*M*c**2/e *1E-3)) * 1E-3 #MeV
    if (a==0):
        if (T<2*(Enu_mu)**2*e*1E9/M/c**2):
            return (cross_section(T,Enu_mu)) #~29.8MeV
        else:
            return 0.0
    if (a==1):
        for i in range (0,nsteps+1):
            EE_antimu[i] = Emin + (Enu_max - Emin)/nsteps * i
            int_antimu[i] = (cross_section(T,EE_antimu[i]) * flux(EE_antimu[i],1))
        return (np.trapz(int_antimu, x=EE_antimu))
    if (a==2):
        for i in range (0,nsteps+1):
            EE_e[i] = Emin + (Enu_max - Emin)/nsteps * i
            int_e[i] = (cross_section(T,EE_e[i]) * flux(EE_e[i],2))
        return (np.trapz(int_e, x=EE_e))


#%%
"MAIN PART"

"Defining lists"
T = []
T_rad = []

T_real_mu = []
T_real_antimu = []
T_real_e = []

dNdT_mu =  []
dNdT_antimu =  []
dNdT_e =  []

T_resolution_mu = []
T_resolution_antimu = []
T_resolution_e = []

T.clear()
T_rad.clear()

T_real_mu.clear()
T_real_antimu.clear()
T_real_e.clear()

dNdT_mu.clear()
dNdT_antimu.clear()
dNdT_e.clear()

T_resolution_mu.clear()
T_resolution_antimu.clear()
T_resolution_e.clear()


# In[24]:


"Events analisys, thresold at 0.9KeV and QF of 10%"
for i in range(0,nsteps+1):
    T.append(T_min + (T_max - T_min)/nsteps * i)
    dNdT_mu.append(differential_events(T_min + (T_max - T_min)/nsteps * i,0))
    dNdT_antimu.append(differential_events(T_min + (T_max - T_min)/nsteps * i,1))
    dNdT_e.append(differential_events(T_min + (T_max - T_min)/nsteps * i,2))

nn=int(1E5)
for i in range(nn):
    T_random = random.uniform(T_min,T_max)
    T_rad.append(T_random)

    dNdT_random_mu = differential_events(T_random,0)
    dNdT_random_antimu = differential_events(T_random,1)
    dNdT_random_e = differential_events(T_random,2)

    r_mu = random.uniform(dNdT_mu[0],dNdT_mu[nsteps])
    r_antimu = random.uniform(dNdT_antimu[0],dNdT_antimu[nsteps])
    r_e = random.uniform(dNdT_e[0],dNdT_e[nsteps])

    if dNdT_random_mu > r_mu:
        #if (T_random>= 0.9):
        T_real_mu.append(T_random)
    if dNdT_random_antimu > r_antimu:
        #if (T_random>= 0.9):
        T_real_antimu.append(T_random)
    if dNdT_random_e > r_e:
        #if (T_random>= 0.9):
        T_real_e.append(T_random)

for t in (T_real_mu):
    gauss = random.gauss(t, sigma0 * np.sqrt(t/T_thres)) 
    #if gauss> 0.9:
    T_resolution_mu.append(gauss)

for t in (T_real_antimu):
    gauss = random.gauss(t, sigma0 * np.sqrt(t/T_thres)) 
    #if gauss> 0.9:
    T_resolution_antimu.append(gauss)

for t in (T_real_e):
    gauss = random.gauss(t, sigma0 * np.sqrt(t/T_thres)) 
    #if gauss> 0.9:
    T_resolution_e.append(gauss)


"PLOT MAKING"

"Events plot"

print('Total long antimu ', len(T_resolution_antimu))
print('Total long e ',len(T_resolution_e))
print('Total long mu ',len(T_resolution_mu))

print('expected length antimu + e: ', len(T_resolution_antimu) + len(T_resolution_e))
print('expected length antimu + e + mu: ', len(T_resolution_antimu) + len(T_resolution_e) + len(T_resolution_mu))

T_total1 = []
T_total2 = []
T_total1.clear()
T_total2.clear()

for y in (T_resolution_antimu):
    T_total1.append(y)

for y in (T_resolution_e):
    T_total1.append(y)

print('Total1 long after append antimu + e ', len(T_total1))

for y in (T_total1):
    T_total2.append(y)

for y in (T_resolution_mu):
    T_total2.append(y)

n, bins, patches = plt.hist([T_resolution_antimu,T_resolution_e, T_resolution_mu], 22, density=False, stacked=True, label= [r'$$\bar{\nu}_{\mu}$$',r'$$\nu_{e}$$',r'$$\nu_{\mu}$$'], color = ['lightgreen','yellowgreen','olivedrab'])

plt.xlabel(r'$$T (keV)$$')
plt.ylabel(r'$$Counts/bin $$')
plt.legend()
#plt.savefig("Plot counts with no condition.png", format='png', dpi=1200,bbox_inches = 'tight')
plt.show()


# In[ ]:




