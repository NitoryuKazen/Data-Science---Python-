from math import sin,cos,tan,pi,floor,log10,sqrt,atan,degrees
import numpy as np
#import pyfits
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
from scipy.stats import anderson_ksamp
from scipy.stats import ks_2samp
from scipy import stats
from scipy import optimize
import scipy.optimize as sso
import sys

def objective(limit, target):
    w = np.where(H>limit)
    count = H[w]
    return count.sum() - target

def mklevels(H,xedges,yedges):
	norm=H.sum()
#	contour1=0.99
#	contour2=0.95
#	contour3=0.68
	contour1=0.95
	contour2=0.75
	contour3=0.50
#
#	contour1=0.75
#	contour2=0.50
#	contour3=0.25
	target1 = norm*contour1
	target2 = norm*contour2
	target3 = norm*contour3
	level1= sso.bisect(objective, H.min(), H.max(), args=(target1,))
	level2= sso.bisect(objective, H.min(), H.max(), args=(target2,))
	level3= sso.bisect(objective, H.min(), H.max(), args=(target3,))
	level4=H.max()
	return [level1,level2,level3,level4]


def mklevels2(H,xedges,yedges):
	norm=H.sum()
	contour1=0.6
	contour2=0.4
	contour3=0.2
	target1 = norm*contour1
	target2 = norm*contour2
	target3 = norm*contour3
	level1= sso.bisect(objective, H.min(), H.max(), args=(target1,))
	level2= sso.bisect(objective, H.min(), H.max(), args=(target2,))
	level3= sso.bisect(objective, H.min(), H.max(), args=(target3,))
	level4=H.max()
	return [level1,level2,level3,level4]


def calcul(z):

# if no values, assume Benchmark Model, input is z
  H0 = 70                         # Hubble constant
  WM = 0.3                        # Omega(matter)
  WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda
  WR = 0.        # Omega(radiation)
  WK = 0.        # Omega curvaturve = 1-Omega(total)
  c = 299792.458 # velocity of light in km/sec
  Tyr = 977.8    # coefficent for converting 1/H into Gyr
  DTT = 0.5      # time from z to now in units of 1/H0
  DTT_Gyr = 0.0  # value of DTT in Gyr
  age = 0.5      # age of Universe in units of 1/H0
  age_Gyr = 0.0  # value of age in Gyr
  zage = 0.1     # age of Universe at redshift z in units of 1/H0
  zage_Gyr = 0.0 # value of zage in Gyr
  DCMR = 0.0     # comoving radial distance in units of c/H0
  DCMR_Mpc = 0.0 
  DCMR_Gyr = 0.0
  DA = 0.0       # angular size distance
  DA_Mpc = 0.0
  DA_Gyr = 0.0
  kpc_DA = 0.0
  DL = 0.0       # luminosity distance
  DL_Mpc = 0.0
  DL_Gyr = 0.0   # DL in units of billions of light years
  V_Gpc = 0.0
  a = 1.0        # 1/(1+z), the scale factor of the Universe
  az = 0.5       # 1/(1+z(object))
  h = H0/100.
  WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
  WK = 1-WM-WR-WV
  az = 1.0/(1+1.0*z)
  age = 0.
  n=1000         # number of points in integrals
  for i in range(n):
    a = az*(i+0.5)/n
    adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
    age = age + 1./adot

  zage = az*age/n
  zage_Gyr = (Tyr/H0)*zage
  DTT = 0.0
  DCMR = 0.0
  for i in range(n):
    a = az+(1-az)*(i+0.5)/n
    adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
    DTT = DTT + 1./adot
    DCMR = DCMR + 1./(a*adot)
  DTT = (1.-az)*DTT/n
  DCMR = (1.-az)*DCMR/n
  age = DTT+zage
  age_Gyr = age*(Tyr/H0)
  DTT_Gyr = (Tyr/H0)*DTT
  DCMR_Gyr = (Tyr/H0)*DCMR
  DCMR_Mpc = (c/H0)*DCMR
  ratio = 1.00
  x = sqrt(abs(WK))*DCMR
  if x > 0.1:
    if WK > 0:
      ratio =  0.5*(exp(x)-exp(-x))/x 
    else:
      ratio = sin(x)/x
  else:
    y = x*x
    if WK < 0: y = -y
    ratio = 1. + y/6. + y*y/120.
  DCMT = ratio*DCMR
  DA = az*DCMT
  DA_Mpc = (c/H0)*DA
  kpc_DA = DA_Mpc/206.264806
  DA_Gyr = (Tyr/H0)*DA
  DL = DA/(az*az)
  DL_Mpc = (c/H0)*DL
  DL_Gyr = (Tyr/H0)*DL
  return DL_Mpc,DA_Mpc
###############################################
infocasjobs=[[] for i in range(5)]
infocasjobs_eli=[[] for i in range(5)]
infocasjobs_cd=[[] for i in range(5)]
infocasjobs_ass=[[] for i in range(5)]
rff=[[] for i in range(3)]
eta=[[] for i in range(3)]

#(cluster,chi2,re,mag,n,axis,pa,float(rff),float(A0),float(A1),delta,gama,flagdelta,flagmask,chi2_exp,re_exp,mag_exp,n_exp,axis_exp,pa_exp,rs_exp,magd_exp,axisd_exp,pad_exp))
############################################
with open('pargal_L07_compact_casjobs.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_casjobs.dat','r')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	if [int(float(ll1[12])),int(float(ll1[13]))] == [0,0]:
		infocasjobs[0].append(float(ll1[24]))
		infocasjobs[1].append(float(ll1[25]))
		infocasjobs[2].append(float(ll1[26]))
		infocasjobs[3].append(float(ll1[27]))
		infocasjobs[4].append(float(ll1[28])/float(ll1[29]))
		if (1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			infocasjobs_cd[0].append(float(ll1[24]))
			infocasjobs_cd[1].append(float(ll1[25]))
			infocasjobs_cd[2].append(float(ll1[26]))
			infocasjobs_cd[3].append(float(ll1[27]))
			infocasjobs_cd[4].append(float(ll1[28])/float(ll1[29]))
			rff[0].append(np.log10(float(ll1[7])))
			eta[0].append(float(ll1[7])-float(ll1[9]))
		elif (float(ll1[7]) - float(ll1[9]))/float(ll1[7]) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			infocasjobs_ass[0].append(float(ll1[24]))
			infocasjobs_ass[1].append(float(ll1[25]))
			infocasjobs_ass[2].append(float(ll1[26]))
			infocasjobs_ass[3].append(float(ll1[27]))
			infocasjobs_ass[4].append(float(ll1[28])/float(ll1[29]))
			rff[1].append(np.log10(float(ll1[7])))
			eta[1].append(float(ll1[7])-float(ll1[9]))
	
		else:
			infocasjobs_eli[0].append(float(ll1[24]))
			infocasjobs_eli[1].append(float(ll1[25]))
			infocasjobs_eli[2].append(float(ll1[26]))
			infocasjobs_eli[3].append(float(ll1[27]))
			infocasjobs_eli[4].append(float(ll1[28])/float(ll1[29]))
			rff[2].append(np.log10(float(ll1[7])))
			eta[2].append(float(ll1[7])-float(ll1[9]))

#cModelAbsMag_r,logMass,age,metallicity,petroR50_r,petroR90_r
########################################################
# RFXETA: COM LINHAS E SEM LINHAS

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.scatter(rff[0],eta[0],s=20,c='blue',edgecolors='black')
plt.scatter(rff[1],eta[1],s=20,c='red',edgecolors='black')
plt.scatter(rff[2],eta[2],s=20,c='green',edgecolors='black')
plt.plot([np.log10(0.0198),np.log10(0.0198)],[-0.01,0.1],c='black',linestyle='--')
plt.legend()
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/rffxeta_curvlines_casjobs.png')
plt.close(fig1)

#########################################
#HISTOGRAMAS
#print infocasjobs[3],max(infocasjobs[3]),min(infocasjobs[3])
magbins=np.arange(min(infocasjobs[0]),-19,0.3)
cbins=np.arange(0.2,0.45,0.01)
agebins=np.arange(min(infocasjobs[2]),max(infocasjobs[2]),1)
metalbins=np.arange(np.log10(min(infocasjobs[3])),np.log10(max(infocasjobs[3])),0.11)
starmassbins=np.arange(10.,max(infocasjobs[1]),0.1)
#rebins=np.arange(min(refparsecs),3.5,0.3)
#axbins=np.arange(min(axis),max(axis),0.05)
#whabins=np.arange(min(wha),5.,0.2)
#ksval1=ks_2samp(,)[1]
#xbins=np.arange()

#MAGNITUDE ABSOLUTA

fig3=plt.figure()
plt.hist(infocasjobs_eli[0],bins=magbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(infocasjobs_cd[0],bins=magbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$Mag_r$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig3.savefig('graficos/magabs_zhao.png')
plt.close(fig3)

average0 = np.average(infocasjobs_eli[0])
ESM0 = np.std(infocasjobs_eli[0])/sqrt(len(infocasjobs_eli[0]))
ksval0a=ks_2samp(infocasjobs_eli[0],infocasjobs_eli[0])[1]
ksval0c=ks_2samp(infocasjobs_eli[0],infocasjobs_cd[0])[1]
std0 = np.std(infocasjobs_eli[0])

average01 = np.average(infocasjobs_cd[0])
ESM01 = np.std(infocasjobs_cd[0])/sqrt(len(infocasjobs_cd[0]))
ksval01a=ks_2samp(infocasjobs_cd[0],infocasjobs_eli[0])[1]
ksval01c=ks_2samp(infocasjobs_cd[0],infocasjobs_cd[0])[1]
std01 = np.std(infocasjobs_cd[0])

print '                     MEDIA   ESM STD  KS KS KS (ORDEM DO TCC)'
print '$Mag_r$', average01, ESM01,std01,ksval01a,ksval01c
print '$Mag_r$', average0, ESM0,std0,ksval0a,ksval0c


#CONCENTRATION

fig6=plt.figure()
plt.hist(infocasjobs_eli[4],bins=cbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(infocasjobs_cd[4],bins=cbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$C$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig6.savefig('graficos/conc_zhao.png')
plt.close(fig6)

average0 = np.average(infocasjobs_eli[4])
ESM0 = np.std(infocasjobs_eli[4])/sqrt(len(infocasjobs_eli[4]))
ksval0a=ks_2samp(infocasjobs_eli[4],infocasjobs_eli[4])[1]
ksval0c=ks_2samp(infocasjobs_eli[4],infocasjobs_cd[4])[1]
std0 = np.std(infocasjobs_eli[4])

average02 = np.average(infocasjobs_cd[4])
ESM02 = np.std(infocasjobs_cd[4])/sqrt(len(infocasjobs_cd[4]))
ksval02a=ks_2samp(infocasjobs_cd[4],infocasjobs_eli[4])[1]
ksval02c=ks_2samp(infocasjobs_cd[4],infocasjobs_cd[4])[1]
std02 = np.std(infocasjobs_cd[4])

print '$C$', average02, ESM02,std02,ksval02a,ksval02c
print '$C$', average0, ESM0,std0,ksval0a,ksval0c

#AGE

fig9=plt.figure()
plt.hist(infocasjobs_eli[2],bins=agebins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(infocasjobs_cd[2],bins=agebins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$ \tau$ (Gyr)', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/age_zhao.png')
plt.close(fig9)

average0 = np.average(infocasjobs_eli[2])
ESM0 = np.std(infocasjobs_eli[2])/sqrt(len(infocasjobs_eli[2]))
ksval0a=ks_2samp(infocasjobs_eli[2],infocasjobs_eli[2])[1]
ksval0c=ks_2samp(infocasjobs_eli[2],infocasjobs_cd[2])[1]
std0 = np.std(infocasjobs_eli[2])

average02 = np.average(infocasjobs_cd[2])
ESM02 = np.std(infocasjobs_cd[2])/sqrt(len(infocasjobs_cd[2]))
ksval02a=ks_2samp(infocasjobs_cd[2],infocasjobs_eli[2])[1]
ksval02c=ks_2samp(infocasjobs_cd[2],infocasjobs_cd[2])[1]
std02 = np.std(infocasjobs_cd[2])

print 'age', average02, ESM02,std02,ksval02a,ksval02c
print 'age', average0, ESM0,std0,ksval0a,ksval0c

#METALICIDADE

fig9=plt.figure()
plt.hist(np.log10(infocasjobs_eli[3]),bins=metalbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(np.log10(infocasjobs_cd[3]),bins=metalbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel('metal', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/metal_zhao.png')
plt.close(fig9)

average0 = np.average(infocasjobs_eli[3])
ESM0 = np.std(infocasjobs_eli[3])/sqrt(len(infocasjobs_eli[3]))
ksval0a=ks_2samp(infocasjobs_eli[3],infocasjobs_eli[3])[1]
ksval0c=ks_2samp(infocasjobs_eli[3],infocasjobs_cd[3])[1]
std0 = np.std(infocasjobs_eli[3])

average02 = np.average(infocasjobs_cd[3])
ESM02 = np.std(infocasjobs_cd[3])/sqrt(len(infocasjobs_cd[3]))
ksval02a=ks_2samp(infocasjobs_cd[3],infocasjobs_eli[3])[1]
ksval02c=ks_2samp(infocasjobs_cd[3],infocasjobs_cd[3])[1]
std02 = np.std(infocasjobs_cd[3])

print 'metal', average02, ESM02,std02, ksval02a,ksval02c
print 'metal', average0, ESM0,std0,ksval0a,ksval0c


#MASSA ESTELAR

fig9=plt.figure()
plt.hist(infocasjobs_eli[1],bins=starmassbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(infocasjobs_cd[1],bins=starmassbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend(loc='upper left')
plt.xlabel(r'$\log{M*/M_\odot}$', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/starmass_zhao.png')
plt.close(fig9)

average0 = np.average(infocasjobs_eli[1])
ESM0 = np.std(infocasjobs_eli[1])/sqrt(len(infocasjobs_eli[1]))
ksval0a=ks_2samp(infocasjobs_eli[1],infocasjobs_eli[1])[1]
ksval0c=ks_2samp(infocasjobs_eli[1],infocasjobs_cd[1])[1]
std0 = np.std(infocasjobs_eli[1])

average02 = np.average(infocasjobs_cd[1])
ESM02 = np.std(infocasjobs_cd[1])/sqrt(len(infocasjobs_cd[1]))
ksval02a=ks_2samp(infocasjobs_cd[1],infocasjobs_eli[1])[1]
ksval02c=ks_2samp(infocasjobs_cd[1],infocasjobs_cd[1])[1]
std02 = np.std(infocasjobs_cd[1])

print 'starmass', average02, ESM02,std02, ksval02a,ksval02c
print 'starmass', average0, ESM0,std0,ksval0a,ksval0c

'''
#WHA

fig7=plt.figure()
plt.hist(whares,bins=whabins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(whaassim,bins=whabins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(whaenv,bins=whabins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel('[$H\\alpha]$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig7.savefig('zhao/hist_graf_vmine/wha_zhao.png')
plt.close(fig7)

average0 = np.average(whares)
ESM0 = np.std(whares)/sqrt(len(whares))
ksval0a=ks_2samp(whares,whares)[1]
ksval0b=ks_2samp(whares,whaassim)[1]
ksval0c=ks_2samp(whares,whaenv)[1]
std0 = np.std(whares)

average01 = np.average(whaassim)
ESM01 = np.std(whaassim)/sqrt(len(whaassim))
ksval01a=ks_2samp(whaassim,whares)[1]
ksval01b=ks_2samp(whaassim,whaassim)[1]
ksval01c=ks_2samp(whaassim,whaenv)[1]
std01 = np.std(whaassim)

average02 = np.average(whaenv)
ESM02 = np.std(whaenv)/sqrt(len(whaenv))
ksval02a=ks_2samp(whaenv,whares)[1]
ksval02b=ks_2samp(whaenv,whaassim)[1]
ksval02c=ks_2samp(whaenv,whaenv)[1]
std02 = np.std(whaenv)

print '$[H_alpha]$', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print '$[H_alpha]$', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print '$[H_alpha]$', average0, ESM0,std0,ksval0a,ksval0b,ksval0c

#indice de sersic

fig8=plt.figure()
plt.hist(nres,bins=nbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(nassim,bins=nbins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(nenv,bins=nbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$n$', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig8.savefig('zhao/hist_graf_vmine/n_zhao.png')
plt.close(fig8)

average0 = np.average(nres)
ESM0 = np.std(nres)/sqrt(len(nres))
ksval0a=ks_2samp(nres,nres)[1]
ksval0b=ks_2samp(nres,nassim)[1]
ksval0c=ks_2samp(nres,nenv)[1]
std0 = np.std(nres)

average01 = np.average(nassim)
ESM01 = np.std(nassim)/sqrt(len(nassim))
ksval01a=ks_2samp(nassim,nres)[1]
ksval01b=ks_2samp(nassim,nassim)[1]
ksval01c=ks_2samp(nassim,nenv)[1]
std01 = np.std(nassim)

average02 = np.average(nenv)
ESM02 = np.std(nenv)/sqrt(len(nenv))
ksval02a=ks_2samp(nenv,nres)[1]
ksval02b=ks_2samp(nenv,nassim)[1]
ksval02c=ks_2samp(nenv,nenv)[1]
std02 = np.std(nenv)

print 'n', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print 'n', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print 'n', average0, ESM0,std0,ksval0a,ksval0b,ksval0c
'''

