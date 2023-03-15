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

infoall=[[] for i in range(3)] #cluster,logrff,eta,bt,chiratio
infail=[[] for i in range(5)]
infoexp=[[] for i in range(5)]

############################################
with open('pargal_L07_compact_casjobs.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_casjobs.dat','r')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls2=inp2.readline()
	ll2=ls2.split()
	ls3=inp3.readline()
	ll3=ls3.split()
	if [int(ll3[2]),int(ll3[21])] == [0,0]:
		infoall[0].append(ll1[0])
		infoall[1].append(np.log10(float(ll1[7])))
		infoall[2].append((float(ll1[7]) - float(ll1[9])))
		if ll2[1] == 'fail':
			infail[0].append(ll1[0])
			infail[1].append(np.log10(float(ll1[7])))
			infail[2].append((float(ll1[7]) - float(ll1[9])))
			infail[3].append(0)
			infail[4].append(0)
		else:
			infoexp[0].append(ll1[0])
			infoexp[1].append(np.log10(float(ll1[7])))
			infoexp[2].append((float(ll1[7]) - float(ll1[9])))
			infoexp[3].append(10**(-0.4*float(ll2[3]))/(10**(-0.4*float(ll2[3]))+10**(-0.4*float(ll2[8]))))
			infoexp[4].append(float(ll1[1])/float(ll2[1]))


		
########################################################
# RFXETA: COM LINHAS E SEM LINHAS

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.scatter(infoexp[1],infoexp[2],s=20,c='white',edgecolors='black')
plt.scatter(infail[1],infail[2],s=20,c='white',edgecolors='black')
plt.plot([np.log10(0.0198),np.log10(0.0198)],[-0.01,0.1],c='black',linestyle='--')
plt.legend()
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/rffxeta_curvlines.png')
plt.close(fig1)

fig2=plt.figure()
plt.scatter(infoexp[1],infoexp[2],s=20,c='white',edgecolors='black')
plt.scatter(infail[1],infail[2],s=20,c='white',edgecolors='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/rffxeta_zhao_nolines.png')
plt.close(fig2)
###########################

#CHI2_S/CHI2_E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
plt.scatter(infoexp[1],infoexp[2],c=infoexp[4],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.06,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(infail[1],infail[2],c='white',s=20,edgecolors='black')
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()

cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chixchi.png')
plt.close(fig1)

#RATIO B/T

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
plt.scatter(infoall[1],infoall[2],c=infoall[3],s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.scatter(infail[1],infail[2],c='white',s=20,edgecolors='black')
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/bt.png')
plt.close(fig1)


'''
fig1=plt.figure()
fraclist=[0.5]
plt.scatter(rffall_chi,etaall_chi,c=ratiochi2_obs,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0229,0.3,1000)),np.linspace(0.0229,0.3,1000)-0.5*np.linspace(0.0229,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_2lines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffall_chi,etaall_chi,c=ratiochi2_obs,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_nolines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffall_chi,etaall_chi,c=ratiochi2_obs,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_onelines.png')
plt.close(fig1)

##########################################

fig1=plt.figure()
fraclist=[0.5]
plt.scatter(rffall_chi,etaall_chi,c=bt_obs,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0229,0.3,1000)),np.linspace(0.0229,0.3,1000)-0.5*np.linspace(0.0229,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_2lines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffall_chi,etaall_chi,c=bt_obs,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_nolines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffall_chi,etaall_chi,c=bt_obs,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_onelines.png')
plt.close(fig1)
############################################

# B/T SIMULADO

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
plt.scatter(rffsim,etasim,c=bt_sim,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_sim_curvlines.png')
plt.close(fig1)

fig1=plt.figure()
fraclist=[0.5]
plt.scatter(rffsim,etasim,c=bt_sim,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0045,0.3,1000)),np.linspace(0.0045,0.3,1000)-0.5*np.linspace(0.0045,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0045),np.log10(0.0045)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_sim_2lines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffsim,etasim,c=bt_sim,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_sim_nolines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffsim,etasim,c=bt_sim,s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
plt.plot([np.log10(0.0045),np.log10(0.0045)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_bt_sim_onelines.png')
plt.close(fig1)

###########################################

#CHI2_S/CHI2_E SIMULADO		

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
plt.scatter(rffsim_chi,etasim_chi,c=ratiochi2_sim,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_sim_curvlines.png')
plt.close(fig1)

fig1=plt.figure()
fraclist=[0.5]
plt.scatter(rffsim_chi,etasim_chi,c=ratiochi2_sim,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0045,0.3,1000)),np.linspace(0.0045,0.3,1000)-0.5*np.linspace(0.0045,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0045),np.log10(0.0045)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_sim_2lines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffsim_chi,etasim_chi,c=ratiochi2_sim,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_sim_nolines.png')
plt.close(fig1)

fig1=plt.figure()
plt.scatter(rffsim_chi,etasim_chi,c=ratiochi2_sim,s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.1,edgecolors='black')
plt.plot([np.log10(0.0045),np.log10(0.0045)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
cbar=plt.colorbar()
cbar.set_label(r'$\chi^2_S/\chi^2_D$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_chiratio_sim_onelines.png')
plt.close(fig1)


##########################################
#GRAFICOS MORFOLOGICOS 

fig0=plt.figure()
fraclist=[0.5]
plt.scatter(rffeli,etaeli,c='green',edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0229,0.3,1000)),np.linspace(0.0229,0.3,1000)-0.5*np.linspace(0.0229,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$ E')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_eli_2lines.png')
plt.close(fig0)

fig0=plt.figure()
fraclist=[0.5]
plt.scatter(rffecd,etaecd,c='green',edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0229,0.3,1000)),np.linspace(0.0229,0.3,1000)-0.5*np.linspace(0.0229,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$ E/cD')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_ecd_2lines.png')
plt.close(fig0)

fig0=plt.figure()
fraclist=[0.5]
plt.scatter(rffcd,etacd,c='green',edgecolors='black')
for item in fraclist:
	plt.plot(np.log10(np.linspace(0.0229,0.3,1000)),np.linspace(0.0229,0.3,1000)-0.5*np.linspace(0.0229,0.3,1000),label=str(0.5),c='black')
plt.plot([np.log10(0.0229),np.log10(0.0229)],[-0.01,0.1],c='black',linestyle='--')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$ cD')
plt.ylabel(r'$\eta$')
plt.savefig('zhao/hist_graf_vmine/rffxeta_zhao_cd_2lines.png')
plt.close(fig0)
#########################################
#HISTOGRAMAS
magbins=np.arange(min(magabs),-19,0.3)
nbins=np.arange(min(n),max(n),1.4)
rebins=np.arange(min(refparsecs),3.5,0.3)
axbins=np.arange(min(axis),max(axis),0.05)
cbins=np.arange(1.8,4,0.15)
whabins=np.arange(min(wha),5.,0.2)
agebins=np.arange(min(age),max(age),1)
metalbins=np.arange(min(metal),-1.8,0.05)
starmassbins=np.arange(10.,max(starmass),0.1)
#ksval1=ks_2samp(,)[1]
#xbins=np.arange()

#MAGNITUDE ABSOLUTA

fig3=plt.figure()
plt.hist(magabsres,bins=magbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(magabsassim,bins=magbins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(magabsenv,bins=magbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$Mag_r$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig3.savefig('zhao/hist_graf_vmine/magabs_zhao.png')
plt.close(fig3)

average0 = np.average(magabsres)
ESM0 = np.std(magabsres)/sqrt(len(magabsres))
ksval0a=ks_2samp(magabsres,magabsres)[1]
ksval0b=ks_2samp(magabsres,magabsassim)[1]
ksval0c=ks_2samp(magabsres,magabsenv)[1]
std0 = np.std(magabsres)

average01 = np.average(magabsassim)
ESM01 = np.std(magabsassim)/sqrt(len(magabsassim))
ksval01a=ks_2samp(magabsassim,magabsres)[1]
ksval01b=ks_2samp(magabsassim,magabsassim)[1]
ksval01c=ks_2samp(magabsassim,magabsenv)[1]
std01 = np.std(magabsassim)

average02 = np.average(magabsenv)
ESM02 = np.std(magabsenv)/sqrt(len(magabsenv))
ksval02a=ks_2samp(magabsenv,magabsres)[1]
ksval02b=ks_2samp(magabsenv,magabsassim)[1]
ksval02c=ks_2samp(magabsenv,magabsenv)[1]
std02 = np.std(magabsenv)

print '                     MEDIA   ESM STD  KS KS KS (ORDEM DO TCC)'
print '$Mag_r$', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print '$Mag_r$', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print '$Mag_r$', average0, ESM0,std0,ksval0a,ksval0b,ksval0c

#RAIO EFETIVO PARSECS

fig4=plt.figure()
plt.hist(refparsecsres,bins=rebins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(refparsecsassim,bins=rebins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(refparsecsenv,bins=rebins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$\log R_e$ (pc)', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig4.savefig('zhao/hist_graf_vmine/refparsecs_zhao.png')
plt.close(fig4)

average0 = np.average(refparsecsres)
ESM0 = np.std(refparsecsres)/sqrt(len(refparsecsres))
ksval0a=ks_2samp(refparsecsres,refparsecsres)[1]
ksval0b=ks_2samp(refparsecsres,refparsecsassim)[1]
ksval0c=ks_2samp(refparsecsres,refparsecsenv)[1]
std0 = np.std(refparsecsres)

average01 = np.average(refparsecsassim)
ESM01 = np.std(refparsecsassim)/sqrt(len(refparsecsassim))
ksval01a=ks_2samp(refparsecsassim,refparsecsres)[1]
ksval01b=ks_2samp(refparsecsassim,refparsecsassim)[1]
ksval01c=ks_2samp(refparsecsassim,refparsecsenv)[1]
std01 = np.std(refparsecsassim)

average02 = np.average(refparsecsenv)
ESM02 = np.std(refparsecsenv)/sqrt(len(refparsecsenv))
ksval02a=ks_2samp(refparsecsenv,refparsecsres)[1]
ksval02b=ks_2samp(refparsecsenv,refparsecsassim)[1]
ksval02c=ks_2samp(refparsecsenv,refparsecsenv)[1]
std02 = np.std(refparsecsenv)


print '$\log{R_e}$', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print '$\log{R_e}$', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print '$\log{R_e}$', average0, ESM0,std0,ksval0a,ksval0b,ksval0c

#AXIS

fig5=plt.figure()
plt.hist(axisres,bins=axbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(axisassim,bins=axbins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(axisenv,bins=axbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend(loc='upper left')
plt.xlabel(r'$b/a$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig5.savefig('zhao/hist_graf_vmine/axis_zhao.png')
plt.close(fig5)

average0 = np.average(axisres)
ESM0 = np.std(axisres)/sqrt(len(axisres))
ksval0a=ks_2samp(axisres,axisres)[1]
ksval0b=ks_2samp(axisres,axisassim)[1]
ksval0c=ks_2samp(axisres,axisenv)[1]
std0 = np.std(axisres)

average01 = np.average(axisassim)
ESM01 = np.std(axisassim)/sqrt(len(axisassim))
ksval01a=ks_2samp(axisassim,axisres)[1]
ksval01b=ks_2samp(axisassim,axisassim)[1]
ksval01c=ks_2samp(axisassim,axisenv)[1]
std01 = np.std(axisassim)

average02 = np.average(axisenv)
ESM02 = np.std(axisenv)/sqrt(len(axisenv))
ksval02a=ks_2samp(axisenv,axisres)[1]
ksval02b=ks_2samp(axisenv,axisassim)[1]
ksval02c=ks_2samp(axisenv,axisenv)[1]
std02 = np.std(axisenv)

print '$b/a$', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print '$b/a$', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print '$b/a$', average0, ESM0,std0,ksval0a,ksval0b,ksval0c


#CONCENTRATION

fig6=plt.figure()
plt.hist(concres,bins=cbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(concassim,bins=cbins,edgecolor='red',histtype='step',density=True,label='Assimetricas')
plt.hist(concenv,bins=cbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$C$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig6.savefig('zhao/hist_graf_vmine/conc_zhao.png')
plt.close(fig6)

average0 = np.average(concres)
ESM0 = np.std(concres)/sqrt(len(concres))
ksval0a=ks_2samp(concres,concres)[1]
ksval0b=ks_2samp(concres,concassim)[1]
ksval0c=ks_2samp(concres,concenv)[1]
std0 = np.std(concres)

average01 = np.average(concassim)
ESM01 = np.std(concassim)/sqrt(len(concassim))
ksval01a=ks_2samp(concassim,concres)[1]
ksval01b=ks_2samp(concassim,concassim)[1]
ksval01c=ks_2samp(concassim,concenv)[1]
std01 = np.std(concassim)

average02 = np.average(concenv)
ESM02 = np.std(concenv)/sqrt(len(concenv))
ksval02a=ks_2samp(concenv,concres)[1]
ksval02b=ks_2samp(concenv,concassim)[1]
ksval02c=ks_2samp(concenv,concenv)[1]
std02 = np.std(concenv)

print '$C$', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print '$C$', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print '$C$', average0, ESM0,std0,ksval0a,ksval0b,ksval0c


#	WHA

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

#AGE

fig9=plt.figure()
plt.hist(ageres,bins=agebins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(ageassim,bins=agebins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(ageenv,bins=agebins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$ \tau$ (Gyr)', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('zhao/hist_graf_vmine/age_zhao.png')
plt.close(fig9)

average0 = np.average(ageres)
ESM0 = np.std(ageres)/sqrt(len(ageres))
ksval0a=ks_2samp(ageres,ageres)[1]
ksval0b=ks_2samp(ageres,ageassim)[1]
ksval0c=ks_2samp(ageres,ageenv)[1]
std0 = np.std(ageres)

average01 = np.average(ageassim)
ESM01 = np.std(ageassim)/sqrt(len(ageassim))
ksval01a=ks_2samp(ageassim,ageres)[1]
ksval01b=ks_2samp(ageassim,ageassim)[1]
ksval01c=ks_2samp(ageassim,ageenv)[1]
std01 = np.std(ageassim)

average02 = np.average(ageenv)
ESM02 = np.std(ageenv)/sqrt(len(ageenv))
ksval02a=ks_2samp(ageenv,ageres)[1]
ksval02b=ks_2samp(ageenv,ageassim)[1]
ksval02c=ks_2samp(ageenv,ageenv)[1]
std02 = np.std(ageenv)

print 'age', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print 'age', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print 'age', average0, ESM0,std0,ksval0a,ksval0b,ksval0c

#METALICIDADE

fig9=plt.figure()
plt.hist(metalres,bins=metalbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(metalassim,bins=metalbins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(metalenv,bins=metalbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel('metal', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('zhao/hist_graf_vmine/metal_zhao.png')
plt.close(fig9)

average0 = np.average(metalres)
ESM0 = np.std(metalres)/sqrt(len(metalres))
ksval0a=ks_2samp(metalres,metalres)[1]
ksval0b=ks_2samp(metalres,metalassim)[1]
ksval0c=ks_2samp(metalres,metalenv)[1]
std0 = np.std(metalres)

average01 = np.average(metalassim)
ESM01 = np.std(metalassim)/sqrt(len(metalassim))
ksval01a=ks_2samp(metalassim,metalres)[1]
ksval01b=ks_2samp(metalassim,metalassim)[1]
ksval01c=ks_2samp(metalassim,metalenv)[1]
std01 = np.std(metalassim)

average02 = np.average(metalenv)
ESM02 = np.std(metalenv)/sqrt(len(metalenv))
ksval02a=ks_2samp(metalenv,metalres)[1]
ksval02b=ks_2samp(metalenv,metalassim)[1]
ksval02c=ks_2samp(metalenv,metalenv)[1]
std02 = np.std(metalenv)

print 'metal', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print 'metal', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print 'metal', average0, ESM0,std0,ksval0a,ksval0b,ksval0c


#MASSA ESTELAR

fig9=plt.figure()
plt.hist(starmassres,bins=starmassbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(starmassassim,bins=starmassbins,edgecolor='red',histtype='step',density=True,label='Assimetrica')
plt.hist(starmassenv,bins=starmassbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend(loc='upper left')
plt.xlabel(r'$\log{M*/M_\odot}$', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('zhao/hist_graf_vmine/starmass_zhao.png')
plt.close(fig9)

average0 = np.average(starmassres)
ESM0 = np.std(starmassres)/sqrt(len(starmassres))
ksval0a=ks_2samp(starmassres,starmassres)[1]
ksval0b=ks_2samp(starmassres,starmassassim)[1]
ksval0c=ks_2samp(starmassres,starmassenv)[1]
std0 = np.std(starmassres)

average01 = np.average(starmassassim)
ESM01 = np.std(starmassassim)/sqrt(len(starmassassim))
ksval01a=ks_2samp(starmassassim,starmassres)[1]
ksval01b=ks_2samp(starmassassim,starmassassim)[1]
ksval01c=ks_2samp(starmassassim,starmassenv)[1]
std01 = np.std(starmassassim)

average02 = np.average(starmassenv)
ESM02 = np.std(starmassenv)/sqrt(len(starmassenv))
ksval02a=ks_2samp(starmassenv,starmassres)[1]
ksval02b=ks_2samp(starmassenv,starmassassim)[1]
ksval02c=ks_2samp(starmassenv,starmassenv)[1]
std02 = np.std(starmassenv)

print 'starmass', average02, ESM02,std02, ksval02a,ksval02b,ksval02c
print 'starmass', average01, ESM01,std01,ksval01a,ksval01b,ksval01c
print 'starmass', average0, ESM0,std0,ksval0a,ksval0b,ksval0c
'''
