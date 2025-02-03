"""
Kenzie Nimmo 2021
Modified by Laura Spitler 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

mpl.rcParams['font.size'] = 6
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.pad']='4'
mpl.rcParams['ytick.major.pad']='4'

#* NB. 1 W.Hz^{-1} == 1.05026*10^{-11} Jy.kpc^2    * 
#* --> L = T_B*y**2*(2.761e-23)    Watts/Hz        *      
#*      = [ ]*(1.05025e-13)  Jy,kpc^2              *
# note you need to multiply by 1e9**2 to convert to GHz 

### Plotting parameters
plot_Tb = True  #Set true if you want lines of constant brightness temperature

##For general plots
#Symbol, color, alpha
plt_repeater = ['x', 'lightcoral', 1.0]
plt_oneoff = ['x', 'tab:red', 1.0]

plt_pulsar = ['.', 'tab:blue', 0.7]
plt_rrat = ['.', 'lightblue', 0.7]
plt_crab = ['.', 'darkblue', 0.2]
plt_crabns = ['.', 'darkblue', 0.5]
plt_sgr = ['o', 'tab:blue', 1.0]

plt_gw = ['s', 'darkgreen', 1.0]
plt_sn = ['s', 'tab:green', 1.0]
plt_grb = ['s', 'lightgreen', 0.5]

plt_sun = ['p', 'tab:olive', 1.0]
plt_ss = ['h', 'tan', 1.0]

plt_flare = ['d', 'sandybrown', 1.0]
plt_bd = ['d', 'tab:brown', 1.0]

plt_xrb = ['>', 'indigo', 1.0]
plt_lpt = ['<', 'indigo', 0.5]

##Source label size
plt_size = 8 #size of source class labels

#lines of constant brightness temperature
if plot_Tb:
    TB = [1e0,1e4,1e8,1e12,1e16,1e20,1e24,1e28,1e32,1e36,1e40,1e44,1e48]
    x=np.linspace(1e-10,1e20,100)

    for tb in TB:
        plt.plot(x,tb*(x**2)*2.761e-5*1.05025e-13,color='k',lw=0.3,alpha=0.2,linestyle='--')

    # add TB labels
    tb_rot=45
    plt.text(5e-9,1e11,r'$10^{44}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(1.5e-8,1e8,r'$10^{40}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e-8,1e5,r'$10^{36}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(1.5e-7,1e2,r'$10^{32}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e-7,1e-1,r'$10^{28}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(1.5e-6,1e-4,r'$10^{24}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e-6,1e-7,r'$10^{20}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(1.5e7,1e14,r'$10^{16}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(1.5e9,1e14,r'$10^{12}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e10,1e13,r'$10^{8}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e10,1e9,r'$10^{4}$ K',rotation=tb_rot,color='k',alpha=0.3)
    plt.text(5e10,1e5,r'0 K',rotation=tb_rot,color='k',alpha=0.3)

#plt.grid()

### Add FRB redshift labels
#frb_z = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
#frb_z_L = np.array([1e6, 1e11, 10**11.9, 10**13.5, 10**15.7])
#frb_z_T = np.repeat(0.1, len(frb_z))
#frb_z_Td = np.logspace(-5, -1, 5)
#z2plot = [1,2,3,4]

#for ii,zz in enumerate(frb_z):
#for ii in z2plot:
    #plt.plot(frb_z_T[ii], frb_z_L[ii], marker=0, color=plt_repeater[1], alpha=plt_repeater[2])
#    frb_z_Ld = frb_z_L[ii]*(frb_z_Td/1e-3)**-0.5
#    plt.plot(frb_z_Td, frb_z_Ld, '--', color='lightcoral')

#    if ii==np.min(z2plot) or ii==np.max(z2plot):
#        plt.text(frb_z_T[ii]+0.1, frb_z_Ld[-1], 'z=%1.1f' % frb_z[ii], color='lightcoral', size=plt_size, va='center')

# Pulsars general (psrcat)
psr=open('pulsars.txt','r')
lines=psr.readlines()
psrx=[]
psry=[]
for n in lines:
    psrx.append(float(n.split()[4])) 
    psry.append(float(n.split()[5]))


plt.scatter(psrx,psry,color=plt_pulsar[1],marker=plt_pulsar[0],alpha=plt_pulsar[2])
plt.text(0.5e-7,1e-3,'Pulsars',color=plt_pulsar[1], size=plt_size)

# RRATs general
rrat=open('rrats.txt','r')
lines=rrat.readlines()
rratx=[]
rraty=[]
for n in lines:
    rratx.append(float(n.split()[4]))
    rraty.append(float(n.split()[5]))


plt.scatter(rratx,rraty,color=plt_rrat[1],marker=plt_rrat[0],alpha=plt_rrat[2])
plt.text(0.8e-2,2e2,'Rotating Radio\nTransients',color=plt_rrat[1], size=plt_size)

# Crab giant pulses
crabgrp=np.loadtxt('crab_giant.txt')
crabip=np.loadtxt('crab_giant.ip.txt')
crabgrpx=[]
crabgrpy=[]
for n in range(len(crabgrp)):
    crabgrpx.append(crabgrp[n][0]*crabgrp[n][1]*1e-3)
    crabgrpy.append(crabgrp[n][2]/crabgrp[n][1] * (2)**2)

for n in range(len(crabip)):
    crabgrpx.append(crabip[n][0]*crabip[n][1]*1e-3)
    crabgrpy.append(crabip[n][2]/crabip[n][1] * (2)**2)

plt.scatter(crabgrpx,crabgrpy,color=plt_crab[1],marker=plt_crab[0],alpha=plt_crab[2])
plt.text(1e-8,0.6e2,'Crab giant\npulses',color=plt_crab[1], size=plt_size)

# Crab nano-shots (Hankins+2003 and Jessner+2010)
cnano=np.loadtxt('crab_nano.txt')
for n in range(len(cnano)):
    plt.scatter(cnano[n][0],cnano[n][1],s=8,color=plt_crabns[1],marker=plt_crabns[0], alpha=plt_crabns[2])
#plt.text(5e-9,1e2,'Crab nanoshots',color=plt_crabns[1], size=plt_size)

# The other nano-shots

# FRB 121102 range
#spitler+2016, Scholz+2017, law+2017, michilli+2018, hessels+2018, gourdji+2018, gajjar+2018,hardy+2017, Houben+2019, Majid+2020, Josephy+2020, Rajwade+2020, Caleb+2020

# order is freq in GHz, width in ms and Fluence in Jy ms
frb121102=np.loadtxt('frb121102.txt')
frb121102x=[]
frb121102y=[]
for n in range(len(frb121102)):
    frb121102x.append(frb121102[n][0]*frb121102[n][1]*1e-3)
    frb121102y.append(frb121102[n][2]/frb121102[n][1] * (972e3)**2)


plt.scatter(frb121102x,frb121102y,color=plt_repeater[1],marker=plt_repeater[0],alpha=plt_repeater[2])
#plt.text(1e-1,5e11,'FRB 20121102A',color='lightblue')

# R3 range
# CHIME/FRB discovery paper, Marcote+2020, Chawla+2020, CHIME/FRB periodicity, Pleunis+2021
frb180916=np.loadtxt('frb180916.txt')
frb180916x=[]
frb180916y=[]
for n in range(len(frb180916)):
    frb180916x.append(frb180916[n][0]*frb180916[n][1]*1e-3)
    frb180916y.append(frb180916[n][2]/frb180916[n][1] * (149e3)**2)


plt.scatter(frb180916x,frb180916y,color=plt_repeater[1],marker=plt_repeater[0],alpha=plt_repeater[2])
##Use if you want to differentiate repeater and non repeaters
#plt.text(1e-8,1e12,'Repeating\nFast Radio Bursts',color=plt_repeater[1], size=plt_size, alpha=plt_repeater[2])

# SGR 1935+2154 range
sgr = np.loadtxt('sgr1935.txt')
sgrx=[]
sgry=[]
for n in range(len(sgr)):
    sgrx = np.append(sgrx,sgr[n][1])
    sgry = np.append(sgry,sgr[n][0])

plt.scatter(sgrx,sgry,color=plt_sgr[1], marker=plt_sgr[0], alpha=plt_sgr[2], fc='none')
plt.text(2e-3,2e5,'Magnetar\nSGR 1935+2154',color=plt_sgr[1], size=plt_size)
#plt.fill_between([np.min(sgrx),np.max(sgrx)],[np.min(sgry),np.min(sgry)],[np.max(sgry),np.max(sgry)],alpha=0.7,color='pink')

# FRB 190711
# Macquart+2020, Kumar+2020
frb190711=np.loadtxt('frb190711.txt')
frb190711x=[]
frb190711y=[]
for n in range(len(frb190711)):
    frb190711x.append(frb190711[n][0]*frb190711[n][1]*1e-3)
    frb190711y.append(frb190711[n][2]/frb190711[n][1] * (2700e3)**2)


plt.scatter(frb190711x,frb190711y,color=plt_repeater[1],marker=plt_repeater[0],alpha=plt_repeater[2])
#plt.text(1e-1,5e13,'FRB 20190711A',color='green', size=plt_size)

# R3 Nimmo et al. 2021
frb180916_micro = np.loadtxt('frb180916_micro.txt')
frb180916_microx=[]
frb180916_microy=[]
for n in range(len(frb180916_micro)):
    frb180916_microx.append(frb180916_micro[n][0]*frb180916_micro[n][1]*1e-3)
    frb180916_microy.append(frb180916_micro[n][2]/frb180916_micro[n][1] * (149e3)**2)


plt.scatter(frb180916_microx,frb180916_microy,color=plt_repeater[1],marker=plt_repeater[0],alpha=plt_repeater[2])
#plt.text(1e-7,1e13,'FRB 20180916B (Nimmo et al. 2021)', color='springgreen')

# M81R Nimmo et al. 2021b
m81nano=np.loadtxt('m81_nano.txt')
m81nanox = []
m81nanoy = []
for n in range(len(m81nano)):
    m81nanox.append(m81nano[n][0]*m81nano[n][1]*1e-3)
    m81nanoy.append(m81nano[n][2]/m81nano[n][1] * (3.6e3)**2)


plt.scatter(m81nanox,m81nanoy,color=plt_repeater[1],marker=plt_repeater[0], alpha=plt_repeater[2])
#plt.text(3e-7,1e7,'FRB 20200120E (this work)', color='black')

### One off FRBs (only localized)
#ASKAP
askap=np.loadtxt('askap_z_tables.txt', unpack=True, usecols=(1,2,3,4), skiprows=47)

askapx=1e-3*askap[2] #convert from ms to seconds
askapy=askap[1]/askap[2]*askap[3]**2*1e6

plt.scatter(askapx, askapy, color=plt_oneoff[1], marker=plt_oneoff[0], alpha=plt_repeater[2])

plt.text(1e-8,0.5e15,'Non-repeating\nFast Radio Bursts', color=plt_oneoff[1], size=plt_size)
plt.text(1e-8,1e12,'Fast Radio\nBursts', color=plt_oneoff[1], size=plt_size)

#DSA-110
dsa110=np.loadtxt('dsa110_law_clean.txt', unpack=True, usecols=(2,3,4))

dsa110x=1e-3*dsa110[1]
dsa110y=dsa110[0]/dsa110[1]*1e6*dsa110[2]**2

plt.scatter(dsa110x, dsa110y, color=plt_oneoff[1], marker=plt_oneoff[0], alpha=plt_oneoff[2])

### Long duration transients
#GW170817

gw170817=np.loadtxt('gw170817.txt', unpack=True)
plt.scatter(gw170817[0], gw170817[1], s=20, color=plt_gw[1], marker=plt_gw[0], alpha=plt_gw[2], fc='none')
plt.text(1e8,2e4,'GW170817', color=plt_gw[1], size=plt_size)

#SNe
sne=np.loadtxt('Gosia_SN2.txt', unpack=True, usecols=(1,6,8))
plt.scatter(sne[0]*86400*sne[2], 1e-20*sne[1], s=20, color=plt_sn[1], marker=plt_sn[0], alpha=plt_sn[2])
plt.text(1e8,3e6,'Supernovae', color=plt_sn[1], size=plt_size)

grb=np.loadtxt('Gosia_GRB2.txt', unpack=True, usecols=(1,6,8))
plt.scatter(grb[0]*86400*grb[2], 1e-20*grb[1], s=20, color=plt_grb[1], marker=plt_grb[0], alpha=plt_grb[2])
plt.text(2e7,2e11,'Gamma-ray\nbursts', color=plt_grb[1], size=plt_size)

#Solar bursts
solar=np.loadtxt('solar_bursts.txt', unpack=True, usecols=(4,5))
plt.scatter(solar[0], solar[1], color=plt_sun[1], marker=plt_sun[0], alpha=0.5, fc='none')
plt.text(0.1,0.5e-6,'Solar bursts', color=plt_sun[1], size=plt_size)

#Flare / brown dwarfs
flare=np.loadtxt('Gosia_flare_stars2.txt', unpack=True, usecols=(1,6,8))
plt.scatter(flare[0]*86400*flare[2], 1e-20*flare[1], color=plt_flare[1], marker=plt_flare[0])
plt.text(0.5e4,5e-8,'Flare stars', color=plt_flare[1], size=plt_size)

bd=np.loadtxt('brown_dwarf.txt', unpack=True, usecols=(0,1))
plt.scatter(bd[0], bd[1], color=plt_bd[1], marker=plt_bd[0])
plt.text(1e3,0.2e-8,'Brown dwarfs', color=plt_bd[1], size=plt_size)

#Solar system
ss=np.loadtxt('solar_system.txt', unpack=True, usecols=(0,1,2))
plt.scatter(ss[0]*ss[1], ss[2], color=plt_ss[1], marker=plt_ss[0])
plt.text(1e-4,7e-10,'Jupiter\nS-burst', color=plt_ss[1], size=plt_size)

#X-ray binaries
xrb=np.loadtxt('Gosia_XRB2.txt', unpack=True, usecols=(1,6,8))
plt.scatter(xrb[0]*86400*xrb[2], 1e-20*xrb[1], color=plt_xrb[1], marker=plt_xrb[0], s=20, alpha=plt_xrb[2])
plt.text(2e7,30,'X-ray binaries', color=plt_xrb[1], size=plt_size)

#LPTs
lpt=np.loadtxt('LPT.txt', unpack=True, usecols=(1,2,3,4))
plt.scatter(lpt[0]*lpt[1], lpt[2]*lpt[3]**2, color=plt_lpt[1], marker=plt_lpt[0], alpha=plt_lpt[2])
plt.text(30, 70, 'Long Period\nTransients', color=plt_lpt[1], size=plt_size)

###Global plot comments
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-9,1e12)
plt.ylim(1e-10,1e17)


plt.xlabel(r'Variability time scale (GHz s)', size=12)
plt.ylabel(r'Radio Pseudo-luminosity (Jy kpc$^2$)', size=12)
#plt.ylabel(r'Spectral Luminosity [erg s$^{-1}$ Hz$^{-1}$]')
#plt.xlabel(r'Transient Duration ($\nu\ W$) [GHz s]', size=14)


# ticks
ax=plt.gcf().get_axes()
ax[0].tick_params(labelsize=10)
#plt.axes().tick_params(axis = 'both', which = 'minor', labelsize = 0, labelcolor='white')
#plt.axes().set_xticks([1e-8,1e-6,1e-4,1e-2,1], minor=True)
#plt.axes().set_yticks([1e-9,1e-8,1e-6,1e-5,1e-3,1e-2,1,10,1e3,1e4,1e6,1e7,1e9,1e10,1e12,1e13,1e15], minor=True)

#plt.axes().set_xticks([1e-9,1e-7,1e-5,1e-3,1e-1,10])
#plt.axes().set_xticklabels([r'$10^{-9}$',r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$'])

#plt.axes().set_yticks([1e-10,1e-7,1e-4,1e-1,1e2,1e5,1e8,1e11,1e14])
#plt.axes().set_yticklabels([r'$10^{10}$',r'$10^{13}$',r'$10^{16}$',r'$10^{19}$',r'$10^{22}$',r'$10^{25}$',r'$10^{28}$',r'$10^{31}$',r'$10^{34}$'])

plt.savefig('TPS.pdf',format='pdf',dpi=300)
plt.show()
