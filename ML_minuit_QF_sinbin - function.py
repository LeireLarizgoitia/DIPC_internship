#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

from __future__ import division

import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 20})
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
import math
import scipy.integrate as integrate
import scipy.integrate as quad
import scipy.constants as constants
import pandas as pd
import random
from scipy import stats

from scipy.stats import norm
from scipy import special
from scipy.special import gamma, factorial
import matplotlib.mlab as mlab
import statistics
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.special import gammaln # x! = Gamma(x+1)
from time import time

import seaborn as sns
from statsmodels.base.model import GenericLikelihoodModel
from iminuit import Minuit

"Constants"
c = constants.c # speed of light in vacuum
e = constants.e #elementary charge
Na = constants.Avogadro #Avogadro's number
fs = constants.femto # 1.e-15
year = constants.year #one year in seconds
bar = constants.bar
kpc = 1e3 * constants.parsec #m
kpc_cm = kpc * 1e2 #cm
ton = constants.long_ton #one long ton in kg

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
hbar_c_ke = 6.58211899*1e-17 * c #KeV cm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2
"1GeV-2 = 0.389379mb" "10mb = 1fm^2"
Gf = Gf * 1e-12 * (hbar_c_ke)**3 #keV cm^3


"Xenon 132 isotope"
Z =  54
N =  78
A = Z+N
M =  131.9041535 * constants.u # mass of Xenon 132 in kg
M = M*c**2/e *1e-3 #mass of Xenon 132 in keV
M_u = 131.9041535 #in u

"RMS of Xenon"
Rn2 = (4.8864)**2
Rn4 = (5.2064)**4

"SUPERNOVA NEUTIRNOS"

erg_MeV = 624.15 * 1e3 # MeV

"SUPERNOVA MODEL"
alpha = 2.3
A_true =  3.9*1e11 #cm-2
Eav_true = 14 #MeV

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = 50.

"Recoil energy range of Xenon in KeV"
T_min= 1.
T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))
T_thres = 1.  #THRESOLD RECOIL ENERGY FOR DETECTION

#sigma0 = 0.4 #corresponds to T_thresold for Xe

sigma0 = 0.4 #corresponds to AT/T for SN progenitor

QF = 1. #QF in Xe , 20%

"Approximation values"
sin_theta_w_square = 0.231 #zero momnetum transfer data from the paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

"NORMALIZATION CONSTANT"
Mass_detector =  10* 1e3 #kg
Dist = 10 #kpc
Distance = Dist * kpc_cm # Supernova at the Galactic centre in cm

print('Mass of the detector in kg : ', Mass_detector)
print('Distance to the source in kpc : ', Dist)

Area = 4*np.pi* Distance**2

Dist = Distance / kpc_cm
print('Distance to the source in kpc : ', Dist)
print('Solid angle : ', Area)
print('Solid angle reverse : ', 1/Area)

efficiency = 0.80

normalization =  Mass_detector * 1e3 * Na / (M_u * Area)

print('Normalization constant for a detector of mass ', Mass_detector, ' kg and a distance to the source of ', Dist, ' kpc : '  , normalization)

nsteps = 100

"FUNCTIONS"

def F(Q2,N,Rn2,Rn4):  # form factor of the nucleus
    Fn = N* (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    return (Fn)

def cross_section(T,Enu, N,M,Rn2,Rn4):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 /(2*np.pi)  / 4 * F(Q2,N,Rn2,Rn4)**2 * M / (hbar_c_ke)**4 * (2 - 2*T *1E-3 / Enu  + (T *1E-3 / Enu)**2 - M *T * 1E-6 / Enu**2 ) #cm^2/keV
    return dsigmadT

def flux(E,A,E_av):  #Fluxes following a continuous distribution. Normalized
    NN = (alpha + 1)**(alpha + 1) /(E_av * gamma(alpha + 1))
    phi =  NN * (E/E_av)**alpha * np.exp(-(alpha + 1)*(E/E_av)) #energy spectrum
    f = A * phi
    return f

int_e =  np.zeros((nsteps+1),float)
int_antie =  np.zeros((nsteps+1),float)
int_x = np.zeros((nsteps+1),float)
EE_antie = np.zeros((nsteps+1),float)
EE_e = np.zeros((nsteps+1),float)
EE_x = np.zeros((nsteps+1),float)

def differential_events_flux(T,A, Eav,N,M,Rn2,Rn4):
    nsteps = 20
    int=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Emax= Enu_max
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Emax - Emin)/nsteps * i
        int[i] = (cross_section(T,EE[i],N,M,Rn2,Rn4) * flux(EE[i],A,Eav))
    return (np.trapz(int, x=EE))

def constant_usefull():
    c = constants.c # speed of light in vacuum
    e = constants.e #elementary charge
    "RMS of Xenon"
    Rn2 = (4.8864)**2
    Rn4 = (5.2064)**4
    Z =  54
    N =  78
    Na = constants.Avogadro #Avogadro's number
    #Mass_detector =  10* 1e3 #kg
    M_u = 131.9041535 #in u
    #M =  131.9041535 * constants.u # mass of Xenon 132 in kg
    M = 131.9041535 * constants.u*c**2/e *1e-3 #mass of Xenon 132 in keV
    Enu_min = 0.0
    Enu_max = 50.
    T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))
    T_thres = 1.  #THRESOLD RECOIL ENERGY FOR DETECTION

"Binning   -  Return: array of T in bins (bin size dependent)"
def binning():
    T_thres = 1.
    t = 1.487929405
    sigma = sigma0 * T_thres * np.sqrt(t/T_thres)
    x0 = t-sigma

    def x1(x0, sigma0, T_thres):
        a = 1
        b = -2*(x0+(sigma0*T_thres)**2./T_thres)
        c = (x0-2*(sigma0*T_thres)**2./T_thres)*x0
        x = (-b+np.sqrt(b*b-4*a*c))/(2*a)
        sigma = sigma0 * T_thres *np.sqrt((x0+x)/(2*T_thres))

        return [x, sigma]

    bins = []
    while (x0 <=T_max) :
        bin0 = x1(x0, sigma0, T_thres)
        x0 = bin0[0]
        bins.append(bin0)

    binss = []
    centres = []

    for bint in bins:
        centres.append(bint[0]-bint[1])

    bins.insert(0, [t-sigma, sigma0])

    for bint in bins:
        binss.append(bint[0])

    return binss,centres
def fnc_QF(E): #for XENON
    k = 0.133* Z**(2/3) * A**(-1/2) # k=0.166 for Xe
    epsilon = 11.5 * E * Z**(-7/3)
    g = 3 *epsilon**(0.15) + 0.7*epsilon**(0.6) + epsilon
    L = k*g / (1 + k*g)

    lambdaa = 0.5
    eta = 3.55
    fl = 1 / (1 + eta * epsilon**(lambdaa))

    alpha = 1.240
    xi = 0.0472
    beta = 239
    gamma = 0.01385
    delta = 0.0620

    W = 13.7 #eV (Dahl's thesis)

    F = 530 #V/m electric field

    NexNi = alpha * F**(-xi) * (1 - np.exp(-beta*epsilon))

    phi = gamma * F**(-delta)
    Ni = E * L / W  * (1/ (1 + NexNi))
    r = 1 - (np.log(1 + Ni * phi) / (Ni*phi))
    #r = 1 - (np.log(1 + phi) / (phi))


    ne = L * E/W * (1/(1 + NexNi)) * (1-r) * fl

    Qy = ne/E
    return Qy



#%%
"MAIN PART"

TT=[]
dNdT = []

nsteps = 100
for i in range(0,nsteps+1):
    TT.append(T_thres + (T_max - T_thres)/nsteps * i)
    dNdT.append( Mass_detector *1e3* Na /M_u * (differential_events_flux(TT[i],A_true, Eav_true,N, M,Rn2,Rn4))) #normlization factor m_detector in g

total = np.trapz(dNdT, x=TT)
dNdT.clear()

print('Total number of events EXPECTED - NO BINNING: ', total)


"Events on intervals after Resolution"
def fnc_events_interval_obs(A,E):
    constant_usefull()
    binss,centres = binning()
    T_bins = []
    T_bins_plot = []
    dNdT = []
    dNdT_plot = []
    events_interval= []
    events_interval_obs= []
    dNdT_res_s = []
    dNdT_res_obs=[]

    t_obs_plot=[]
    dNdT_res_obs_plot=[]

    t_obs_plot1=[]
    dNdT_res_obs_plot1=[]

    T_true1=[]
    dNdT1 = []

    T_true=[]
    dNdT = []


    nsteps = 100
    #T_max=15.

    for i in range(0,nsteps+1):
        T_true.append(T_thres + (T_max - T_thres)/nsteps * i)
        dNdT.append( Mass_detector *1e3* Na /M_u * (differential_events_flux(T_true[i],A,E,N, M,Rn2,Rn4))) #normlization factor m_detector in g

    #total = np.trapz(dNdT, x=T_true)

    for i in range(0,nsteps+1):
        t = T_thres + (T_max - T_thres)/nsteps * i
        x = (1-fnc_QF(t) )* t
        #x = 0.2 * t
        if x >= T_thres:
            T_true1.append(x)
            dNdT1.append( Mass_detector *1e3* Na /M_u * (differential_events_flux(T_true[i],A,E,N, M,Rn2,Rn4))) #normlization factor m_detector in g


    "Sample T -> Tobs , in the way we want"
    TT_obs=[]
    nsteps_obs=1000

    'apply QF to distribution with resolution'

    tqf=[]
    lqf=[]

    for i in range(0,nsteps_obs+1):
        t = T_true[0] + (T_true[len(T_true)-1] - T_true[0])/nsteps_obs * i
        x = (1-fnc_QF(t) )* t
        tqf.append(x)
        lqf.append(fnc_QF(t))
        #x = 0.2 * t
        if x >= T_thres:
            TT_obs.append(x)

    T_bins=[]
    #T_bins.append(0.0)
    for e in binss:
        T_bins.append(e)

    tbin=[]
    for j in range(0,len(T_bins)-1):
        for i in range(0,len(TT_obs)):
            if TT_obs[i]<=T_bins[j+1] and TT_obs[i]>=T_bins[j]:
                tbin.append(TT_obs[i])

        for tobs in tbin: #T observed
            for i in range(0,len(T_true1)):
                sigma = sigma0 * T_thres * np.sqrt(T_true1[i]/T_thres)
                #sigma = np.sqrt(sigma0 * ttrue)
                gauss_res = 1/(sigma* np.sqrt(2*np.pi)) * np.exp(-(T_true1[i]-tobs)**2 / (2*sigma**2))
                dNdT_res_s.append(gauss_res * dNdT1[i])

            dNdT_res_obs.append(np.trapz(dNdT_res_s, x=T_true1))
            dNdT_res_obs_plot.append(np.trapz(dNdT_res_s, x=T_true1))

            t_obs_plot.append(tobs)
            dNdT_res_s.clear()


        events_interval_obs.append(np.trapz(dNdT_res_obs, x=tbin))

        dNdT_res_obs.clear()
        tbin.clear()

    events_interval_obs_simple=[]
    for e in events_interval_obs:
        if e!=0.0:
            events_interval_obs_simple.append(e)

    plt.plot(tqf,lqf,label=r'$ \rm{expected} $')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    "Compare two pdf-s"
    #plt.plot(T_true,dNdT,label=r'$ \rm{expected} $')
    #plt.plot(T_true1,dNdT1,label=r'$ \rm{expected } \, \rm{after } \, \rm{QF} $')
    #plt.plot(t_obs_plot1,dNdT_res_obs_plot1,label=r'$ \rm{resolution } 40\% $')
    #plt.plot(t_obs_plot,dNdT_res_obs_plot,label=r'$\rm{resolution } 40\% \, \rm{and } \, \rm{QF} $')
    #plt.vlines(x=T_thres, ymin=0, ymax=30, colors = 'black', linestyle='dashed', label = 'Threshold line')
    #plt.xlim(0,15)
    #plt.ylim(0,30)
    #plt.xlabel(r'$ \rm{T}(keV) $')
    #plt.ylabel(r'$ \frac{dN}{dT} \, (\frac{\rm{events}}{keV})$')
    #plt.legend()
    #plt.savefig("dNdT_res40_QFfunction_distribution.png", format='png', dpi=1200,bbox_inches = 'tight')
    #plt.show()

    return events_interval_obs_simple

#events_interval_obs = fnc_events_interval_obs(A_true,Eav_true)  #n obs
events_interval_obs = fnc_events_interval_obs(A_true,Eav_true)  #n obs

#print('TOTAL', total)

#print('Total number of events EXPECTED: ', sum(events_interval_exp))
print('Total number of events after resolution and QF: ', sum(events_interval_obs))
print('Total number of events after resolution and QF per BIN: ',events_interval_obs)

"""

"MINUIT"

def fcn_np(par):
    constant_usefull()
    at=par[0]
    ev=par[1]
    #print(at)
    #print(ev)
    n_obs = fnc_events_interval_obs(A_true,Eav_true) #events_obs
    mu = fnc_events_interval_obs(at,ev) #events_est
    #print(mu)

    sum_tot=0
    for i in range(0,len(n_obs)): #sum over bins
        #if mu[i]<=0.0:
    #        chi2 =100000
    #    else:
        sum_tot = sum_tot + (mu[i] - n_obs[i] + n_obs[i]*np.log(n_obs[i] / mu[i])) # this is lnL / lnL_max
        chi2 = 2*sum_tot
    #print(chi2)
    return chi2


fcn_np.errordef = 1 #Minuit.LIKELIHOOD

#A_true = 3.9*1e11 #cm-2
#Eav_true = 14 #MeV
at_start = 3.8*1e11
ev_start = 15

m = Minuit(fcn_np, (at_start,ev_start),name=("a", "b")) #

m.limits['a'] = (1, None)
m.limits['b'] = (1, None)

m.migrad()  # run optimiser
#m.simplex().migrad()  # run optimiser
print(m.values)

a_ML = m.values[0] #ESTIMATED PARAMETERS
e_ML = m.values[1]

#m.hesse()   # assumes gaussian distribution, not adecuate, ours POISSON
m.minos()   # run covariance estimator
print(m.errors)

"CONTOUR PLOT"
#points = contour(4, 5)
grid1 = m.mncontour('a','b', cl=0.6827)  #1SIGMA
#grid = m.contour('a','b', cl=0.95)    #2SIGMA


def plot_contour(grid1,Eav_true,A_true,e_ML,a_ML):
    plt.plot(grid1[:,1],grid1[:,0] ,label=r'$1 \sigma \, \rm{contour}$')
    #plt.plot(grid2[:,1],grid2[:,0] ,label=r'$2 \sigma \, \rm{contour}$')
    plt.scatter(Eav_true,A_true, label='True values', color='black', marker=',', s=1)
    plt.scatter(e_ML,a_ML, label='MLE values', color='green', marker='1', s=1)
    plt.xlabel(r'$ \rm{Average \, Neutrino \, Energy} \, <E_T> \, [MeV]$')
    plt.ylabel(r'$ \rm{Neutrino \, Flux \, Amplitude \,} \, A_T \, [cm^{-2}]$')
    plt.xlim(5,30)
    plt.ylim(0.01*1e11,10*1e11)
    plt.legend()
    #plt.savefig("E 14 A 3.9e11_res20.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

#plot_contour(grid1,Eav_true,A_true,e_ML,a_ML)



"Save contour data"

with open("Contour_res40_QFfunction1.txt", "w") as txt_file:
    for line in grid1:
        content = str(line)
        txt_file.write(" ".join(content) + "\n") #AT, Eav

txt_file.close()
"""

"Prove that contour gives 2.3 if 2D or 1 if 1D for example"

#T_obs=[]
#dNdT_est=[]
#dNdT_obs=[]
#n_obs = []
#mu=[]
#chi2=[]

#n_true = fnc_events_interval(A_true,Eav_true)
#binss = binning()

#for aa in range(0,len(grid1[:,1])):
#    a_obs = grid1[aa,0]
#    e_obs = grid1[aa,1]
#    mu = fnc_events_interval(a_obs,e_obs)
#    sum_tot=0
#    for ii in range(0, len(binss)-1): #sum over bins
#        sum_tot = sum_tot + (mu[ii] - n_true[ii] + n_true[ii]*np.log(n_true[ii] / mu[ii])) # this is lnL / lnL_max
#    chi2.append(2*sum_tot)
#    mu.clear()
#print(chi2)





"END OF CODE"
