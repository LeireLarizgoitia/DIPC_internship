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
e = constants.e #elementary charge
fs = constants.femto # 1.e-15

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2

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

"FUNCTIONS"

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

def Plot_cross_section():
    cross = []
    x = []
    Enu = 40 #MeV
    Tmin = 0.0
    Tmax = 2*(Enu*1e3)**2*e*1e3/M/c**2
    for i in range(0,nsteps+1):
        x.append(Tmin + (Tmax - Tmin)/nsteps * i)
        cross.append(cross_section(x[i],40)) #cross section for 40MeV neutrino energy
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x,cross, color="orange", label=r'Xe for $E_{\nu}=40MeV$')
    plt.ylim(top=1e-38)
    plt.xlabel(r'$T(keV)$')
    plt.ylabel(r'$\frac{d\sigma}{dT} (\frac{cm^2}{keV})$')
    plt.legend()
    #plt.savefig("Plot cross section 40MeV.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_flux():
    Enu = []
    f_antimu = []
    f_e = []
    for i in range(0,nsteps+1):
        Enu.append(Enu_min + (Enu_max - Enu_min)/nsteps * i)
        f_antimu.append(flux(Enu[i],1))
        f_e.append(flux(Enu[i],2))
    plt.axvline(x=(m_pion**2 - m_muon**2) / (2*m_pion), label=r'$\nu_{\mu}$', color='red')
    plt.plot(Enu, f_antimu, label= r'$\bar{\nu}_{\mu}$', color='green', linestyle='dashdot')
    plt.plot(Enu, f_e, label=r'$\nu_{e}$', color='blue', linestyle='dashed')
    plt.xlim((minimum_value(Enu)))
    plt.ylim((minimum_value(f_antimu)))
    plt.xlabel(r'$E_{\nu}(MeV)$')
    plt.ylabel(r'$\frac{d\phi}{dE_{\nu}} (a.u.)$')
    plt.legend()
    #plt.savefig("Plot neutrino flux.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_thresold():
    T_thres_max = Enu_mu
    T_thres_min = 0.9
    T_thresold = []
    TT = []
    dNdT = []
    events = []
    for j in range(0,nsteps+1):
        T_thresold.append(T_thres_min + (T_thres_max - T_thres_min)/nsteps * j )
        for i in range(0,nsteps+1):
            TT.append(T_thresold[j] + (T_max - T_thresold[j])/nsteps * i)
            dNdT.append(1e38*(differential_events(TT[i],0) + differential_events(TT[i],1)+ differential_events(TT[i],2)))
        events.append(np.trapz(dNdT, x=TT))
        TT.clear()
        dNdT.clear()

    plt.yscale('log')
    plt.plot(T_thresold, events, label='Xe', color='orange')
    plt.xlabel(r'$T_{thresold}(keV)$')
    plt.ylabel(r'$CE\nu NS $ nuclear recoils')
    plt.legend()
    #plt.savefig("Plot thresold change till Enu_mu.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

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

"Neutrino flux plot"
#Plot_flux()
"Form factor plot"
#Plot_ff(xx,ff1,ff,ff2)
"Cross section plot"
#Plot_cross_section()
"Detector Thresold plot"
#Plot_thresold()


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

    if (dNdT_random_mu > r_mu):
        if (T_random*0.1>= 0.9):
            T_real_mu.append(T_random*0.1)
    if (dNdT_random_antimu > r_antimu):
        if (T_random*0.1>= 0.9):
            T_real_antimu.append(T_random*0.1)
    if (dNdT_random_e > r_e):
        if (T_random*0.1>= 0.9):
            T_real_e.append(T_random*0.1)

for t in (T_real_mu):
    T_resolution_mu.append(random.gauss(t, sigma0 * np.sqrt(t/T_thres)))

for t in (T_real_antimu):
    T_resolution_antimu.append(random.gauss(t, sigma0 * np.sqrt(t/T_thres)))

for t in (T_real_e):
    T_resolution_e.append(random.gauss(t, sigma0 * np.sqrt(t/T_thres)))

"PLOT MAKING"

"Events plot"
Tsum1 = T_resolution_antimu + T_resolution_e
Tsum2 = T_resolution_antimu + T_resolution_e + T_resolution_mu

# the histogram of all the data
n, bins, patches = plt.hist(Tsum2, 22, density=1, label=r'$\nu_{\mu}$', color='olivedrab') #all flavours
n, bins, patches = plt.hist(Tsum1, 22, density=1, label= r'$\nu_{e}$', color='yellowgreen') #only muon antineutrino and electron neutrino
n, bins, patches = plt.hist(T_resolution_antimu, 22, density=1, label=r'$\bar{\nu}_{\mu}$', color='lightgreen') #only muon antineutrino

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
#plt.savefig("Events"+str(nn)+"bins22+QF.png", format='png', dpi=1200,bbox_inches = 'tight')
plt.show()

# the histogram of the MUON NEUTRINO data
n, bins, patches = plt.hist(T_resolution_mu, 22, density=1, label=r'$\nu_{\mu}$', color='cornflowerblue')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the ELECTRON NEUTRINO data
n, bins, patches = plt.hist(T_resolution_e, 22, density=1, label=r'$\nu_{e}$', color='black')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the MUON ANITNEUTRINO data
n, bins, patches = plt.hist(T_resolution_antimu, 22, density=1, label=r'$\bar{\nu}_{\mu}$', color='orange')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()

# the histogram of the MUON ANITNEUTRINO data
n, bins, patches = plt.hist(T_resolution_antimu + T_resolution_e, 22, density=1, label=r'$\bar{\nu}_{\mu} + \nu_{e}$', color='yellowgreen')

plt.xlabel(r'$T (keV)$')
plt.ylabel(r'$ $')
plt.legend()
plt.show()
