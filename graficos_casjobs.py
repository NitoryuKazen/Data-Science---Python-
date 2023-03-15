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
'''
HEADER PARGAL:

0 - cluster
##################
#SERSIC 
1 - chi2
2 - re
3 - mag
4 - n
5 - axis
6 - pa
#################
#TESTES E QUANTIFICACAO DOS RESIDUOS
7 - float(rff) RFF
8 - float(A0) ASSIMETRIA DE ABRAHAM
9 - float(A1) ASSIMETRIA MODIFICADA DE ABRAHAM
10 - delta
11 - gama
12 - flagdelta
13 - flagmask
#################
#SERSIC + EXPONENCIAL
14 - chi2_exp
15 - re_exp
16 - mag_exp
17 - n_exp
18 - axis_exp
19 - pa_exp
20 - rs_exp
21 - magd_exp
22 - axisd_exp
23 - pad_exp
##################
#SERSIC + SERSIC
24 - chi2_ss
25 - re_ss
26 - mag_ss
27 - n_ss
28 - axis_ss
29 - pa_ss
30 - rd_ss
31 - magd_ss
32 - nd_ss
33 - axisd_ss
34 - pad_ss
##################
#CASJOBS
35 - cModelAbsMag_r
36 - logMass
37 - age
38 - metallicity
39 - petroR90_r
40 - petroR50_r
'''
###############################################
rff,eta,chi2_sse,chi2_sss,chi2_ssse,bt_se,bt_ss,axis_ssse,magb_ssse,magd_ssse,magabs,mass,age,metal,conc=[[] for i in range(15)]
info_eli=[[] for i in range(15)]
info_cd=[[] for i in range(15)]
info_ass=[[] for i in range(15)]
############################################
with open('pargal_L07_compact_casjobs_v2.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_casjobs_v2.dat','r')

for ik in range(0,1):
	ls1=inp1.readline()
	ll1=ls1.split()
	if [int(float(ll1[12])),int(float(ll1[13]))] == [0,0]:
		#for i in range(15):
		rff.append(float(ll1[24]))
		print rff,eta,chi2_sse,chi2_sss,chi2_ssse,bt_se,bt_ss,axis_ssse,magb_ssse,magd_ssse,magabs,mass,age,metal,conc
		'''
			info_all[1].append(float(ll1[25]))
			info_all[2].append(float(ll1[26]))
			info_all[3].append(float(ll1[27]))
			info_all[4].append(float(ll1[28])/float(ll1[29]))
		if (1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			infocasjobs_cd[0].append(float(ll1[24]))
			infocasjobs_cd[1].append(float(ll1[25]))
			infocasjobs_cd[2].append(float(ll1[26]))
			infocasjobs_cd[3].append(float(ll1[27]))
			infocasjobs_cd[4].append(float(ll1[28])/float(ll1[29]))
		elif (float(ll1[7]) - float(ll1[9]))/float(ll1[7]) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			infocasjobs_ass[0].append(float(ll1[24]))
			infocasjobs_ass[1].append(float(ll1[25]))
			infocasjobs_ass[2].append(float(ll1[26]))
			infocasjobs_ass[3].append(float(ll1[27]))
			infocasjobs_ass[4].append(float(ll1[28])/float(ll1[29]))
		else:
			infocasjobs_eli[0].append(float(ll1[24]))
			infocasjobs_eli[1].append(float(ll1[25]))
			infocasjobs_eli[2].append(float(ll1[26]))
			infocasjobs_eli[3].append(float(ll1[27]))
			infocasjobs_eli[4].append(float(ll1[28])/float(ll1[29]))
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
magbins=np.arange(min(infocasjobs[0]),-19,0.3)
cbins=np.arange(1.8,4,0.15)
agebins=np.arange(min(infocasjobs[2]),max(infocasjobs[2]),1)
metalbins=np.arange(min(infocasjobs[3]),max(infocasjobs[3]),0.05)
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
plt.hist(infocasjobs_eli[3],bins=metalbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(infocasjobs_cd[3],bins=metalbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
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

