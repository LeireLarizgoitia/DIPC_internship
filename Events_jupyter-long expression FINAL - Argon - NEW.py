#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

"""ANALYSIS FOR 132 Xe ISOTOPE"""

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
from scipy.stats import norm
import matplotlib.mlab as mlab


"Constants"
c = constants.c # speed of light in vacuum
e = constants.e #elementary charge
fs = constants.femto # 1.e-15
year = constants.year #one year in seconds
bar = constants.bar

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
hbar_c_ke = 6.58211899*1e-17 * c #KeV cm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2
#1GeV-2 = 0.389379mb
#10mb = 1fm^2
Gf = Gf * 1e-12 * (hbar_c_ke)**3

"Masses"
m_pion = 139.57018 #MeV/c^2
m_muon = 105.6583755 #MeV/c^2


#NO QF in Ar

"Argon 40 isotope"
Z = 18
N = 22
M = 39.9623831238 *  constants.u # Ar40
M = M*c**2/e *1e-3 #mass  in keV

"RMS of Argon"
Rn2 = (3.4168)**2
Rn4 = (3.7233)**4

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = m_muon/2 #~52.8MeV
Enu_mu = (m_pion**2 - m_muon**2) / (2*m_pion) #~29.8MeV

"Recoil energy range of Xenon in KeV"
T_min= 0.9
T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))
T_thres = 0.9 #THRESOLD RECOIL ENERGY FOR DETECTION

sigma0 = 0.4 #corresponds to T_thresold for Xe

"Approximation values"
sin_theta_w_square = 0.23867 #zero momnetum transfer data from the paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

"NEUTRINOS PER YEAR IN 3 YEARS"
"ESS operateS on a scheduled 5000 hours of beam delivery per year "
Power = 5e6 #Watt
E_proton = 2e9 * e # Joule
nu_per_flavour_per_proton = 0.3
in_operation = 5000 * 3600 * 3 #s for 3 years

nu_per_flavour_per_year  =  nu_per_flavour_per_proton * Power/E_proton * in_operation #TOTAL OF 3 YEARS

"NORMALIZATION CONSTANT"
Mass_detector = 6 #20 #kg
Pressure = 20 #bar
Distance = 2000 #cm
Area =  75*50 #cm^2

"Preasure interval 10 to 50 bar"
""" mass of detector = volume of detector * pressure """
mass_detector = []
for i in range(10,55,5): #every 5bars
    mass_detector.append(i) #kg

#Mass_detector = mass_detector[8]

print('Mass of the detector in kg : ', Mass_detector)

"Distance interval 10 to 30 m"
distance = []
for i in range(10,35,5): #every 5m
    distance.append(i*100) #cm

#Distance = distance[4]

print('Distance to the source in cm : ', Distance)

#Solid_angle = 2*np.pi *(1 - np.cos(math.atan(0.25/20))) / (4*np.pi)
Solid_angle =  Area / (4*np.pi* Distance**2)

print('Solid angle : ', Solid_angle)

nucleus = Mass_detector/(M *e/c**2*1e3)
nucleus_per_area = nucleus / Area
Nu_on_target = nucleus_per_area * Solid_angle

efficiency = 0.80

normalization = Nu_on_target * nu_per_flavour_per_year * efficiency

print('Normalization constant for a detector of mass ', Mass_detector, ' kg and a distance to the source of ', Distance, ' m : '  , normalization)

nsteps = 100

"FUNCTIONS"

def F(Q2):  # from factor of the nucleus
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    return (Fn)

def F2(Q2): # second order approximation
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2) #approximation
    return (Fn)

def cross_section(T,Enu):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 /(2*np.pi) / 4 * F(Q2)**2 * M / (hbar_c_ke)**4 * (2 - 2*T *1E-3 / Enu  + (T *1E-3 / Enu)**2 - M *T * 1E-6 / Enu**2 ) #cm^2/keV
    return dsigmadT

def flux(E,alpha):  #Fluxes following a continuous distribution. Normalized
    if (alpha == 1): #muon - antineutrino
        f = 64 / m_muon * ((E / m_muon)**2 * (3/4 - E/m_muon))
    if (alpha == 2): #electron neutrino
        f = 192 / m_muon * ((E / m_muon)**2 * (1/2 - E/m_muon))
    return f


int_mu =  np.zeros((nsteps+1),float)
int_antimu =  np.zeros((nsteps+1),float)
int_e = np.zeros((nsteps+1),float)
EE_antimu= np.zeros((nsteps+1),float)
EE_e= np.zeros((nsteps+1),float)

def differential_events(T,a):
    """Integral Bounds"""
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Tnu_mu = 1/2 * (M + 2*Enu_mu*1e3 - np.sqrt(M**2 + 4*Enu_mu*1e3*(M - Enu_mu*1e3)))
    if (a==0):
        if (T<Tnu_mu):
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

"PLOT FUNCTIONS"
def minimum_value(sequence):
    """return the minimum element of sequence"""
    low = sequence[0] # need to start with some value
    for i in sequence:
        if i < low:
            low = i
    return low

def Plot_form_factor():
    Q=[]
    FF4=[]
    FF2=[]
    FF0 = N * np.ones(nsteps+1,float)

    for i in range(0,nsteps+1):
        Q.append(0.0 + 100/nsteps * i) #MeV
        #Q2= 2 *Enu**2 * M/e  *c**2 * T *1E-9 /(Enu**2 - Enu*T*1E-3)
        FF4.append(F((0.0 + 100/nsteps * i)**2))
        FF2.append(F2((0.0 + 100/nsteps * i)**2))

    plt.plot(Q,FF0, label=r'$F(Q^2) = N/Q_w$', color='blue', linestyle='dashdot')
    plt.plot(Q,FF2, label=r'$F(Q^2) = N/Q_w [1 - \frac{Q^2}{3!} <R_n^2>]$' , color='red', linestyle='dashdot')
    plt.plot(Q,FF4, label=r'$F(Q^2) = N/Q_w [1 - \frac{Q^2}{3!} <R_n^2> + \frac{Q^4}{5!} <R_n^4>]$', color='green')
    plt.xlabel(r'$Q (MeV)$')
    plt.ylabel(r'$F(Q^2)$')
    plt.xlim(0.0, 85)
    plt.ylim(bottom=0.35)
    plt.legend()
    #plt.savefig("Plot from factor from Q.png", format='png', dpi=1200,bbox_inches = 'tight')
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

def Plot_cross_section():
    cross = []
    x = []
    Enu = 40. #MeV
    Tmin = 0.0
    Tmax = T_max
    for i in range(0,nsteps+1):
        x.append(Tmin + (Tmax - Tmin)/nsteps * i)
        cross.append(cross_section(x[i],40)) #cross section for 40MeV neutrino energy

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x,cross, color="orange", label=r'$$\mathrm{Xe \, for \, } E_{\nu}=40MeV$$')
    plt.ylim(top=1e-38)
    plt.xlabel(r'$$T(keV)$$')
    plt.ylabel(r'$$\frac{d\sigma}{dT} (\frac{cm^2}{keV})$$')
    plt.legend()
    #plt.savefig("Plot cross section 40MeV_long exp.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_thresold():
    T_thres_max = Enu_mu #T_max
    T_thres_min = 1e-4
    T_thresold = []
    TT = []
    dNdT = []
    events = []
    for j in range(0,nsteps+1):
        T_thresold.append(T_thres_min + (T_thres_max - T_thres_min)/nsteps * j )
        for i in range(0,nsteps+1):
            TT.append(T_thresold[j] + (T_max - T_thresold[j])/nsteps * i)
            dNdT.append(normalization/Mass_detector/3*(differential_events(TT[i],0) + differential_events(TT[i],1)+ differential_events(TT[i],2)))
        events.append(np.trapz(dNdT, x=TT))
        TT.clear()
        dNdT.clear()

    plt.yscale('log')
    plt.plot(T_thresold, events, label='Xe', color='orange')
    plt.xlim(left=0)
    plt.xlabel(r'$$T_{thresold}(keV)$$')
    plt.ylabel(r'$$CE\nu NS \, \mathrm{nuclear \, recoils} \, / (kg \cdot yr)$$')
    plt.legend()
    #.savefig("Plot thresold change till Enu_mu_Xe_per kg year.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_bin_size_proof(binss,T_thres,centres):
    difference = []
    centre_double = []
    width = []
    for i in range(len(binss)-1):
        width.append(binss[i+1]-binss[i])
        centre_double.append( 2 * sigma0 * T_thres * np.sqrt(centres[i]/T_thres) )
        difference.append(np.abs(centre_double[i] - width[i]))

    plt.plot(width, centre_double, label='Difference', color='orange')
    plt.xlabel('Bin width (keV)')
    plt.ylabel('Twice resolution of centre of bin (keV)')
    plt.legend()
    #plt.savefig("Defining bin size with QF FINAL.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

    plt.plot(difference, label='Difference', color='orange')
    plt.ylabel('Twice resolution of centre of bin - Bin width (keV)')
    plt.ylim(bottom=-1. ,top=1.)
    plt.legend()
    #plt.savefig("Defining difference with QF FINAL.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_backgroung_events_histogram(T_bg, binss):
    n, bins, patches = plt.hist(T_bg, binss, density=False, label= r'$$N_{bg}$$', color = 'cornflowerblue')
    plt.xlabel(r'$$T (keV)$$')
    plt.ylabel(r'$$Counts/bin $$')
    plt.legend()
    #plt.savefig('Plot counts with QF of Nbg.png', format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_counts_all_flavours_histogram(T_resol_mu, T_resol_antimu, T_resol_e, binss, T_thres,centres,Tbg,Nbg):

    n, bins, patches = plt.hist([T_resol_antimu,T_resol_e, T_resol_mu], binss, density=False, stacked=True, label= [r'$$\bar{\nu}_{\mu}$$',r'$$\nu_{e}$$',r'$$\nu_{\mu}$$'], color = ['lightgreen','yellowgreen','olivedrab']) #, 'Full spectra' ,'cornflowerblue'
    Nbg.insert(0,0.0)
    plt.step(binss, np.sqrt(Nbg), linestyle='dashed' , label=r'$$\sqrt{N_{bckg}}$$')
    #plt.axvline(x=QF*(m_pion**2 - m_muon**2) / (2*m_pion), color='black', linestyle='dashed')
    #plt.text((QF*(m_pion**2 - m_muon**2) / (2*m_pion)) + 0.4, 2000, r'$$ Maximum \, recoil \, for \, \nu_{\mu}$$', {'color': 'black'},
    #        horizontalalignment='center',
    #        verticalalignment='center',
    #        rotation=90,
    #        clip_on=False)
    plt.xlabel(r'$$T (keV)$$')
    plt.ylabel(r'$$Counts/bin $$')
    plt.legend()
    plt.savefig('Plot counts with QF with nbg with sqrt FINAL_ARGON_NEW.png', format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Plot_counts_per_flavours_histogram(events_interval_antimu,events_step_mu,events_interval_e, T_resol_antimu, T_resol_mu, T_resol_e, binss, T_thres):

    events_step_mu = []
    events_step_mu.append(0.0)

    for e in events_interval_mu:
        events_step_mu.append(e)

    events_step_antimu =[]
    events_step_antimu.append(0.0)

    for e in events_interval_antimu:
        events_step_antimu.append(e)

    events_step_e =[]
    events_step_e.append(0.0)

    for e in events_interval_e:
        events_step_e.append(e)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.step(binss, events_step_mu, label= r'$$\nu_{\mu} \, \mathrm{before} $$')
    ax.hist(T_resol_mu, bins=binss,alpha = 0.5, color = 'lightgreen', label= r'$$\nu_{\mu} \, \mathrm{after} $$')

    ax.set_xlim(right = 5.)
    ax.set(xlabel=r'$$T (keV)$$', ylabel=r'$$Counts/bin $$')
    ax.legend()
    #plt.savefig('Plot counts enu_mu with QF T_thres' + str(T_thres) + '.png', format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()

    "Plot HISTOGRAM of Counts/bin per flavour BEFORE AND AFTER SMEARING"
    fig , ax = plt.subplots(3,1 ,sharex=True)

    fig.subplots_adjust(hspace=0)

    ax[0].step(binss, events_step_mu, label= r'$$\nu_{\mu} \, \mathrm{before} $$')
    ax[0].hist(T_resol_mu, bins=binss,alpha = 0.5, color = 'lightgreen', label= r'$$\nu_{\mu} \, \mathrm{after} $$')
    ax[1].step(binss, events_step_e, label= r'$$\nu_{e} \, \mathrm{before} $$')
    ax[1].hist(T_resol_e, bins=binss,alpha = 0.5, color = 'lightgreen', label= r'$$\nu_{e} \, \mathrm{after} $$')
    ax[2].step(binss, events_step_antimu, label=r'$$\bar{\nu}_{\mu} \, \mathrm{before} $$')
    ax[2].hist(T_resol_antimu, bins=binss,alpha = 0.5, color = 'lightgreen', label= r'$$\bar{\nu}_{\mu} \, \mathrm{after} $$')
    ax[0].set(xlabel=(r'$$T (keV)$$'),ylabel=(r'$$Counts/bin $$'))
    ax[1].set(xlabel=(r'$$T (keV)$$'),ylabel=(r'$$Counts/bin $$'))
    ax[2].set(xlabel=(r'$$T (keV)$$'),ylabel=(r'$$Counts/bin $$'))
    ax[0].set_xlim(right = 5.)
    ax[1].set_xlim(right = 5.)
    ax[2].set_xlim(right = 5.)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    #plt.savefig("Plot counts enu with QF T_thres' + str(T_thres) + '.png", format='png', dpi=1200,bbox_inches = 'tight')
    plt.show()


#%%
"MAIN PART"

"Neutrino flux plot"
#Plot_flux()
"Form factor plot"
#Plot_form_factor()
"Cross section plot"
#Plot_cross_section()
"Detector Thresold plot"
#Plot_thresold()

"Bins definition for Xe"
#MANUALLY SET
#centres = [1.339129405, 2.361388215 , 3.671647025, 5.269905835 , 7.156164645, 9.330423455, 11.79268226, 14.54294107, 17.58119988, 20.90745869, 24.5217175, 28.42397631, 32.61423512, 37.09249393, 41.85875274]

#binss = []
#binss.append(T_thres) #starts at T_thresold
#for t in centres:
#    sigma = sigma0 * T_thres * np.sqrt(t/T_thres)
#    binss.append(t + sigma)

#T_thres = 7 #Thresold change

#ANOTHER METHOD
t = 1.339129405
sigma = sigma0 * T_thres * np.sqrt(t/T_thres)
x0 = t-sigma

#print('sigma', sigma)

def x1(x0, sigma0, T_thres):
    a = 1
    b = -2*(x0+(sigma0*T_thres)**2./T_thres)
    c = (x0-2*(sigma0*T_thres)**2./T_thres)*x0
    x = (-b+np.sqrt(b*b-4*a*c))/(2*a)
    sigma = sigma0 * T_thres *np.sqrt((x0+x)/(2*T_thres))

    return [x, sigma]

bins = []
while (x0 < T_max) :
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

"Events on intervals"
T_bins = []
events_interval_mu= []
dNdT_mu = []

events_interval_e= []
dNdT_e = []

events_interval_antimu= []
dNdT_antimu = []


for j in range(0,len(binss)-1):
    for i in range(0,nsteps+1):
            T_bins.append(binss[j] + ( binss[j+1] - binss[j])/nsteps * i)
            dNdT_mu.append(normalization*(differential_events(T_bins[i],0)))
            dNdT_antimu.append(normalization*(differential_events(T_bins[i],1)))
            dNdT_e.append(normalization*(differential_events(T_bins[i],2)))


    events_interval_mu.append(np.trapz(dNdT_mu, x=T_bins))
    events_interval_antimu.append(np.trapz(dNdT_antimu, x=T_bins))
    events_interval_e.append(np.trapz(dNdT_e, x=T_bins))

    T_bins.clear()
    dNdT_mu.clear()
    dNdT_antimu.clear()
    dNdT_e.clear()

print('Number of events with nu_mu before smearing: ', sum(events_interval_mu))
print('Number of events with antinu_mu before smearing: ', sum(events_interval_antimu))
print('Number of events with nu_e before smearing: ', sum(events_interval_e))
print('Total number of events before smearing: ', sum(events_interval_mu) + sum(events_interval_antimu) + sum(events_interval_e))


T_resol_mu = []
T_resol_antimu = []
T_resol_e = []

sum_mu = 0
sum_antimu = 0
sum_e = 0
sum_bg = 0

for i in range(len(events_interval_mu)):
    num = int(events_interval_mu[i])
    for j in range(num):
        gauss = random.gauss(centres[i], sigma0 * T_thres * np.sqrt(centres[i]/T_thres))
        T_resol_mu.append(gauss)
        if gauss >= T_thres:
            sum_mu = sum_mu + 1

for i in range(len(events_interval_antimu)):
    num = int(events_interval_antimu[i])
    for j in range(num):
        gauss = random.gauss(centres[i], sigma0 * T_thres * np.sqrt(centres[i]/T_thres))
        T_resol_antimu.append(gauss)
        if gauss >= T_thres:
            sum_antimu = sum_antimu + 1

for i in range(len(events_interval_e)):
    num = int(events_interval_e[i])
    for j in range(num):
        gauss = random.gauss(centres[i], sigma0 * T_thres * np.sqrt(centres[i]/T_thres))
        T_resol_e.append(gauss)
        if gauss >= T_thres:
            sum_e = sum_e + 1

"10 counts per kg per keV per day"
factor = 14 * 2.8*1e-3 #factor of running timing
on = 5000 * factor #per year

bg_thres = 10 * Mass_detector /24 * on *3 #in 3 years per keV
width = []
N_bg = []

for i in range(len(binss)-1):
    width.append(binss[i+1]-binss[i])

for w in width:
    N_bg.append(bg_thres * w)

T_bg = []
for i in range(len(N_bg)):
    num = int(N_bg[i])
    for n in range(num):
        T_bg.append(centres[i])
        sum_bg = sum_bg + 1

print('Number of events with nu_mu after smearing: ', sum_mu)
print('Number of events with antinu_mu after smearing:', sum_antimu)
print('Number of events with nu_e after smearing:', sum_e)
print('Total number of events after smearing: ', sum_mu + sum_antimu + sum_e)
print('Total number of background events: ', sum_bg)

"Bin width is twice the energy resolution at its centre"
#Plot_bin_size_proof(binss,T_thres,centres)

"Plot HISTOGRAM of Counts/bin for background events"
#Plot_backgroung_events_histogram(T_bg, binss)

"Plot HISTOGRAM of Counts/bin per flavour, background events taken into account"
Plot_counts_all_flavours_histogram(T_resol_mu,T_resol_antimu,T_resol_e, binss, T_thres, centres, T_bg, N_bg )

"Enu_mu Plot HISTOGRAM of Counts/bin BEFORE AND AFTER SMEARING"
#Plot_counts_per_flavours_histogram(events_interval_antimu,events_interval_mu,events_interval_e, T_resol_antimu, T_resol_mu, T_resol_e, binss, T_thres)

"END OF CODE"
