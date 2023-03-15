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

#1 - chi2
#7 - float(rff) RFF
#9 - float(A1) ASSIMETRIA MODIFICADA DE ABRAHAM
#12 - flagdelta
#13 - flagmask
#14 - chi2_exp
#16 - mag_exp
#18 - axis_exp
#21 - magd_exp
#22 - axisd_exp
##################
#24 - chi2_ss
#26 - mag_ss
#28 - axis_ss
#31 - magd_ss
#33 - axisd_ss
#35 - cModelAbsMag_r
#36 - logMass
#37 - age
#38 - metallicity
#39 - petroR90_r
#40 - petroR50_r
###############################################
#0,1,2=all,cd,eli
infoall=rff,eta,chi2_s,chi2_se,chi2_ss,magb_se,magb_ss,magd_se,magd_ss,axisb_se,axisb_ss,axisd_se,axisd_ss,pab_se,pab_ss,pad_se,pad_ss,bt_se,bt_ss,magabs,mass,age,metal,conc=[[[] for i in range(3)] for i in range(24)]

inforate=chi2_sse,chi2_sss,chi2_ssse,axisb_ssse,axisd_ssse,magb_ssse,magd_ssse,pab_ssse,pad_ssse=[[[] for i in range(3)] for i in range(9)]
######################'######################
with open('pargal_L07_compact_casjobs_v2.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_casjobs_v2.dat','r')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	if [int(float(ll1[12])),int(float(ll1[13]))] == [0,0]:
		rff[0].append(np.log10(float(ll1[7])))
		eta[0].append(float(ll1[7]) - float(ll1[9]))
		chi2_s[0].append(float(ll1[1]))

		if float(ll1[14]) != 0.0:
			chi2_se[0].append(float(ll1[14]))
			magb_se[0].append(float(ll1[16]))
			magd_se[0].append(float(ll1[21]))
			axisb_se[0].append(float(ll1[18]))
			axisd_se[0].append(float(ll1[22]))
			pab_se[0].append(float(ll1[19]))
			pad_se[0].append(float(ll1[23]))
			bt_se[0].append(10**(-0.4*float(ll1[16]))/(10**(-0.4*float(ll1[16]))+10**(-0.4*float(ll1[21]))))
			#
			chi2_sse[0].append(float(ll1[1])/float(ll1[14]))
			chi2_ssse[0].append(float(ll1[24])/float(ll1[14]))
			axisb_ssse[0].append(float(ll1[28])/float(ll1[18]))
			axisd_ssse[0].append(float(ll1[33])/float(ll1[22]))
			magb_ssse[0].append(float(ll1[26])/float(ll1[16]))
			magd_ssse[0].append(float(ll1[31])/float(ll1[21]))
			pab_ssse[0].append(float(ll1[29])/float(ll1[19]))
			pad_ssse[0].append(float(ll1[34])/float(ll1[23]))
		elif float(ll1[14]) == 0.0:
			chi2_se[0].append(0.0)
			magb_se[0].append(0.0)
			magd_se[0].append(0.0)
			axisb_se[0].append(0.0)
			axisd_se[0].append(0.0)
			pab_se[0].append(0.0)
			pad_se[0].append(0.0)
			bt_se[0].append(0.0)
			#
			chi2_sse[0].append(0.0)
			chi2_ssse[0].append(0.0)
			axisb_ssse[0].append(0.0)
			axisd_ssse[0].append(0.0)
			magb_ssse[0].append(0.0)
			magd_ssse[0].append(0.0)
			pab_ssse[0].append(0.0)
			pad_ssse[0].append(0.0)

		if float(ll1[24]) != 0.0:
			chi2_ss[0].append(float(ll1[24]))
			magb_ss[0].append(float(ll1[26]))
			magd_ss[0].append(float(ll1[31]))
			axisb_ss[0].append(float(ll1[28]))
			axisd_ss[0].append(float(ll1[33]))
			pab_ss[0].append(float(ll1[29]))
			pad_ss[0].append(float(ll1[34]))
			bt_ss[0].append(10**(-0.4*float(ll1[26]))/(10**(-0.4*float(ll1[26]))+10**(-0.4*float(ll1[31]))))
			#
			chi2_sss[0].append(float(ll1[1])/float(ll1[24]))
		elif float(ll1[24]) == 0.0:
			chi2_ss[0].append(0.0)
			magb_ss[0].append(0.0)
			magd_ss[0].append(0.0)
			axisb_ss[0].append(0.0)
			axisd_ss[0].append(0.0)
			pab_ss[0].append(0.0)
			pad_ss[0].append(0.0)
			bt_ss[0].append(0.0)
			#
			chi2_sss[0].append(0.0)

		if float(ll1[35]) != 0.0:
			magabs[0].append(float(ll1[35]))
			mass[0].append(float(ll1[36]))
			age[0].append(float(ll1[37]))
			metal[0].append(float(ll1[38]))
			conc[0].append(float(ll1[39])/float(ll1[40]))
		elif float(ll1[35]) == 0.0:
			magabs[0].append(0.0)
			mass[0].append(0.0)
			age[0].append(0.0)
			metal[0].append(0.0)
			conc[0].append(0.0)
							
		if (1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			rff[1].append(np.log10(float(ll1[7])))
			eta[1].append(float(ll1[7]) - float(ll1[9]))
			chi2_s[1].append(float(ll1[1]))

			if float(ll1[14]) != 0.0:
				chi2_se[1].append(float(ll1[14]))
				magb_se[1].append(float(ll1[16]))
				magd_se[1].append(float(ll1[21]))
				axisb_se[1].append(float(ll1[18]))
				axisd_se[1].append(float(ll1[22]))
				pab_se[1].append(float(ll1[19]))
				pad_se[1].append(float(ll1[23]))
				bt_se[1].append(10**(-0.4*float(ll1[16]))/(10**(-0.4*float(ll1[16]))+10**(-0.4*float(ll1[21]))))
				#
				chi2_sse[1].append(float(ll1[1])/float(ll1[14]))
				chi2_ssse[1].append(float(ll1[24])/float(ll1[14]))
				axisb_ssse[1].append(float(ll1[28])/float(ll1[18]))
				axisd_ssse[1].append(float(ll1[33])/float(ll1[22]))
				magb_ssse[1].append(float(ll1[26])/float(ll1[16]))
				magd_ssse[1].append(float(ll1[31])/float(ll1[21]))
				pab_ssse[1].append(float(ll1[29])/float(ll1[19]))
				pad_ssse[1].append(float(ll1[34])/float(ll1[23]))
			elif float(ll1[14]) == 0.0:
				chi2_se[1].append(0.0)
				magb_se[1].append(0.0)
				magd_se[1].append(0.0)
				axisb_se[1].append(0.0)
				axisd_se[1].append(0.0)
				pab_se[1].append(0.0)
				pad_se[1].append(0.0)
				bt_se[1].append(0.0)
				#
				chi2_sse[1].append(0.0)
				chi2_ssse[1].append(0.0)
				axisb_ssse[1].append(0.0)
				axisd_ssse[1].append(0.0)
				magb_ssse[1].append(0.0)
				magd_ssse[1].append(0.0)
				pab_ssse[1].append(0.0)
				pad_ssse[1].append(0.0)

			if float(ll1[24]) != 0.0:
				chi2_ss[1].append(float(ll1[24]))
				magb_ss[1].append(float(ll1[26]))
				magd_ss[1].append(float(ll1[31]))
				axisb_ss[1].append(float(ll1[28]))
				axisd_ss[1].append(float(ll1[33]))
				pab_ss[1].append(float(ll1[29]))
				pad_ss[1].append(float(ll1[34]))
				bt_ss[1].append(10**(-0.4*float(ll1[26]))/(10**(-0.4*float(ll1[26]))+10**(-0.4*float(ll1[31]))))
				#
				chi2_sss[1].append(float(ll1[1])/float(ll1[24]))
			elif float(ll1[24]) == 0.0:
				chi2_ss[1].append(0.0)
				magb_ss[1].append(0.0)
				magd_ss[1].append(0.0)
				axisb_ss[1].append(0.0)
				axisd_ss[1].append(0.0)
				pab_ss[1].append(0.0)
				pad_ss[1].append(0.0)
				bt_ss[1].append(0.0)
				#
				chi2_sss[1].append(0.0)

			if float(ll1[35]) != 0.0:
				magabs[1].append(float(ll1[35]))
				mass[1].append(float(ll1[36]))
				age[1].append(float(ll1[37]))
				metal[1].append(float(ll1[38]))
				conc[1].append(float(ll1[39])/float(ll1[40]))
			elif float(ll1[35]) == 0.0:
				magabs[1].append(0.0)
				mass[1].append(0.0)
				age[1].append(0.0)
				metal[1].append(0.0)
				conc[1].append(0.0)

		elif (float(ll1[7]) - float(ll1[9]))/float(ll1[7]) < 0.5 and np.log10(float(ll1[7])) > np.log10(0.0198):
			pass
		else:
			rff[2].append(np.log10(float(ll1[7])))
			eta[2].append(float(ll1[7]) - float(ll1[9]))
			chi2_s[2].append(float(ll1[1]))

			if float(ll1[14]) != 0.0:
				chi2_se[2].append(float(ll1[14]))
				magb_se[2].append(float(ll1[16]))
				magd_se[2].append(float(ll1[21]))
				axisb_se[2].append(float(ll1[18]))
				axisd_se[2].append(float(ll1[22]))
				pab_se[2].append(float(ll1[19]))
				pad_se[2].append(float(ll1[23]))
				bt_se[2].append(10**(-0.4*float(ll1[16]))/(10**(-0.4*float(ll1[16]))+10**(-0.4*float(ll1[21]))))
				#
				chi2_sse[2].append(float(ll1[1])/float(ll1[14]))
				chi2_ssse[2].append(float(ll1[24])/float(ll1[14]))
				axisb_ssse[2].append(float(ll1[28])/float(ll1[18]))
				axisd_ssse[2].append(float(ll1[33])/float(ll1[22]))
				magb_ssse[2].append(float(ll1[26])/float(ll1[16]))
				magd_ssse[2].append(float(ll1[31])/float(ll1[21]))
				pab_ssse[2].append(float(ll1[29])/float(ll1[19]))
				pad_ssse[2].append(float(ll1[34])/float(ll1[23]))
			elif float(ll1[14]) == 0.0:
				chi2_se[2].append(0.0)
				magb_se[2].append(0.0)
				magd_se[2].append(0.0)
				axisb_se[2].append(0.0)
				axisd_se[2].append(0.0)
				pab_se[2].append(0.0)
				pad_se[2].append(0.0)
				bt_se[2].append(0.0)
				#
				chi2_sse[2].append(0.0)
				chi2_ssse[2].append(0.0)
				axisb_ssse[2].append(0.0)
				axisd_ssse[2].append(0.0)
				magb_ssse[2].append(0.0)
				magd_ssse[2].append(0.0)
				pab_ssse[2].append(0.0)
				pad_ssse[2].append(0.0)

			if float(ll1[24]) != 0.0:
				chi2_ss[2].append(float(ll1[24]))
				magb_ss[2].append(float(ll1[26]))
				magd_ss[2].append(float(ll1[31]))
				axisb_ss[2].append(float(ll1[28]))
				axisd_ss[2].append(float(ll1[33]))
				pab_ss[2].append(float(ll1[29]))
				pad_ss[2].append(float(ll1[34]))
				bt_ss[2].append(10**(-0.4*float(ll1[26]))/(10**(-0.4*float(ll1[26]))+10**(-0.4*float(ll1[31]))))
				#
				chi2_sss[2].append(float(ll1[1])/float(ll1[24]))
			elif float(ll1[24]) == 0.0:
				chi2_ss[2].append(0.0)
				magb_ss[2].append(0.0)
				magd_ss[2].append(0.0)
				axisb_ss[2].append(0.0)
				axisd_ss[2].append(0.0)
				pab_ss[2].append(0.0)
				pad_ss[2].append(0.0)
				bt_ss[2].append(0.0)
				#
				chi2_sss[2].append(0.0)

			if float(ll1[35]) != 0.0:
				magabs[2].append(float(ll1[35]))
				mass[2].append(float(ll1[36]))
				age[2].append(float(ll1[37]))
				metal[2].append(float(ll1[38]))
				conc[2].append(float(ll1[39])/float(ll1[40]))
			elif float(ll1[35]) == 0.0:
				magabs[2].append(0.0)
				mass[2].append(0.0)
				age[2].append(0.0)
				metal[2].append(0.0)
				conc[2].append(0.0)

###########################################################
#VALORES ESTATISTICOS

vec_med=[[] for i in range(24)]
ksval=[[] for i in range(24)]
vec_std=[[] for i in range(24)]
vec_ste=[[] for i in range(24)]
vec_type=['all','E','cD']
vec_header=['rff','eta','chi2_s','chi2_se','chi2_ss','magb_se','magb_ss','magd_se','magd_ss','axisb_se','axisb_ss','axisd_se','axisd_ss','pab_se','pab_ss','pad_se','pad_ss','bt_se','bt_ss','magabs','mass','age','metal','conc']

for i in range(24):
	ksval[i].append((str(vec_header[i]),ks_2samp(infoall[i][2],infoall[i][1])[1]))
	for j in range(3):
		vec_med[i].append((str(vec_header[i])+' '+str(vec_type[j]),np.average(infoall[i][j])))
		vec_std[i].append((str(vec_header[i])+' '+str(vec_type[j]),np.std(infoall[i][j])))
		vec_ste[i].append((str(vec_header[i])+' '+str(vec_type[j]),np.std(infoall[i][j])/sqrt(len(infoall[i][j]))))

print('TESTE KS cD X E \n')
for i in range(24):
	print(ksval[i])
print('----------------------------------------------------')
print('VALOR MEDIO (AVERAGE) \n')
for i in range(24):
	print(vec_med[i])
print('------------------------------------------------------')
print('DESVIO PADRAO \n')
for i in range(24):
	print(vec_std[i])
print('--------------------------------------------------------')
print('ERRO PADRAO DA MEDIA (STD/SQRT) \n')
for i in range(24):
	print(vec_ste[i])
########################################################
# RFXETA: COM LINHAS E SEM LINHAS

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[1.0,0.9,0.5,0.2,0.1,0.0]
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.scatter(rff[1],eta[1],s=20,c='blue',edgecolors='black')
plt.scatter(rff[0],eta[0],s=20,c='red',edgecolors='black')
plt.scatter(rff[2],eta[2],s=20,c='green',edgecolors='black')
plt.plot([np.log10(0.0198),np.log10(0.0198)],[-0.01,0.1],c='black',linestyle='--')
plt.legend()
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/rffxeta_curvlines_casjobs.png')
plt.close(fig1)
##################################################################
################################################################
##############################################################3
#PARAMETROS DO GALFIT E CASJOBS

#RATIO B/T
#SE

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(bt_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(bt_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$B/T (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/bt_se.png')
plt.close(fig1)

#SS

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(bt_ss[0])):
	if bt_ss[0][i] != 0.0:
		vechi2[2].append(bt_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.48,vmax=0.52,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$B/T (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/bt_ss.png')
plt.close(fig1)

###############################################################
#CHI2
#S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_s[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(chi2_s[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1.,vmax=1.1,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$\chi^2 (S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chi2_s.png')
plt.close(fig1)

#CHI2 
#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(chi2_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1.,vmax=1.1,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$\chi^2 (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chi2_se.png')
plt.close(fig1)

#CHI2
#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(chi2_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1.,vmax=1.1,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$\chi^2 (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chi2_ss.png')
plt.close(fig1)
##############################################################
#MAGNITUDE
#BOJO

#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magb_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(magb_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=15.,vmax=17.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$magb_se (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/magb_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magb_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(magb_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=15.,vmax=17.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$mag_b (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/magb_ss.png')
plt.close(fig1)

#ENVELOPE
#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magd_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(magd_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=15.,vmax=17.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$mag_d (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/magd_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magd_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(magd_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=15.,vmax=17.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$mag_d (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/magd_ss.png')
plt.close(fig1)
###############################################################
#RAZAO AXIAL
#BOJO

#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisb_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(axisb_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.5,vmax=1.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$AR_b (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axisb_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisb_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(axisb_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.5,vmax=1.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$axis_b (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axisb_ss.png')
plt.close(fig1)

#ENVELOPE
#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisd_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(axisd_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.5,vmax=1.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$AR_d (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axisd_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisd_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(axisd_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.5,vmax=1.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$AR_d (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axisd_ss.png')
plt.close(fig1)

################################################################
#ANGULO DE POSICAO
#BOJO
#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(pab_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(pab_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=-90,vmax=90,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$PA_b (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/pab_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(pab_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(pab_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=-90,vmax=90,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$PA_b (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/pab_ss.png')
plt.close(fig1)

#ENVELOPE
#S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(pad_se[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(pad_se[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=-90,vmax=90,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$PA_d (S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/pad_se.png')
plt.close(fig1)

#S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(pad_ss[0])):
	if bt_se[0][i] != 0.0:
		vechi2[2].append(pad_ss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=-90,vmax=90,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$PA_d (S+S)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/pad_ss.png')
plt.close(fig1)
###############################################################
###############################################################
###############################################################
#PARAMETROS DA RAZAO DE PARAMETROS DO GALFIT

#CHI2/CHI2
#S/S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_sse[0])):
	if chi2_sse[0][i] != 0.0:
		vechi2[2].append(chi2_sse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	elif chi2_sse[0][i] == 0.0:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.06,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar.set_label(r'$\chi^2_S/\chi^2_(S+E)$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chixchi_SSE.png')
plt.close(fig1)

#S/S+S

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_sss[0])):
	if chi2_sss[0][i] != 0.0:
		vechi2[2].append(chi2_sss[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])

plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.06,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar.set_label(r'$\chi^2_S/\chi^2_(S+S)$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chixchi_SSS.png')
plt.close(fig1)

#S+S/S+E

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(chi2_ssse[0])):
	if chi2_ssse[0][i] != 0.0:
		vechi2[2].append(chi2_ssse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])

plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=1,vmax=1.06,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
cbar.set_label(r'$\chi^2_(S+S)/\chi^2_(S+E)$', rotation=90)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/chixchi_SSSE.png')
plt.close(fig1)

########################################################################################
#RAZAO DA RAZAO AXIAL DO BOJO E DO DISCO (S+S/S+E)
#BOJO

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisb_ssse[0])):
	if axisb_ssse[0][i] != 0.0:
		vechi2[2].append(axisb_ssse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.75,vmax=1.15,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$AR_b(S+S)/AR_b(S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axis_bojo.png')
plt.close(fig1)

#ENVELOPE

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(axisd_ssse[0])):
	if axisd_ssse[0][i] != 0.0:
		vechi2[2].append(axisd_ssse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$AR_d(S+S)/AR_d(S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/axis_envelope.png')
plt.close(fig1)
###################################################################################
#RAZAO MAGNITUDE (S+S/S+E) DO BOJO E DISCO
#BOJO

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magb_ssse[0])):
	if magb_ssse[0][i] != 0.0:
		vechi2[2].append(magb_ssse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),vmin=0.8,vmax=1.,edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$mag_b(S+S)/mag_b(S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/mag_bojo.png')
plt.close(fig1)

#ENVELOPE

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
vechi2=[[],[],[],[],[]]
for i in range(len(magd_ssse[0])):
	if magd_ssse[0][i] != 0.0:
		vechi2[2].append(magd_ssse[0][i])
		vechi2[0].append(rff[0][i])
		vechi2[1].append(eta[0][i])
	else:
		vechi2[3].append(rff[0][i])
		vechi2[4].append(eta[0][i])
plt.scatter(vechi2[0],vechi2[1],c=vechi2[2],s=20,cmap=plt.cm.get_cmap('jet'),edgecolors='black')
cbar=plt.colorbar()
plt.scatter(vechi2[3],vechi2[4],c='white',s=20,edgecolors='black')
plt.plot(np.log10(x),x-0.5*x,label=str(0.5),color='black')
cbar.set_label(r'$mag_d(S+S)/mag_d(S+E)$', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.legend()
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig('graficos/mag_envelope.png')
plt.close(fig1)

#########################################
#HISTOGRAMAS
magbins=np.arange(min(magabs[0]),-19,0.3)
cbins=np.arange(1.8,4,0.15)
agebins=np.arange(min(age[0]),max(age[0]),1)
metalbins=np.arange(min(metal[0]),max(metal[0]),0.05)
starmassbins=np.arange(10.,max(mass[0]),0.1)

#MAGNITUDE ABSOLUTA

fig3=plt.figure()
plt.hist(magabs[2],bins=magbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(magabs[1],bins=magbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$Mag_r$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig3.savefig('graficos/magabs.png')
plt.close(fig3)

#CONCENTRATION

fig6=plt.figure()
plt.hist(np.log10(conc[2]),bins=cbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(np.log10(conc[1]),bins=cbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$log(C)$', fontsize=15)
plt.ylabel('Ocorrencias (norm)', fontsize=15)
fig6.savefig('graficos/conc.png')
plt.close(fig6)

#AGE

fig9=plt.figure()
plt.hist(age[2],bins=agebins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(age[1],bins=agebins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel(r'$ \tau$ (Gyr)', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/age.png')
plt.close(fig9)

#METALICIDADE

fig9=plt.figure()
plt.hist(metal[2],bins=metalbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(metal[1],bins=metalbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend()
plt.xlabel('Z', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/metal.png')
plt.close(fig9)

#MASSA ESTELAR

fig9=plt.figure()
plt.hist(mass[2],bins=starmassbins,edgecolor='green',histtype='step',density=True,label='Elipticas')
plt.hist(mass[1],bins=starmassbins,edgecolor='blue',histtype='step',density=True,label='Envelope')
plt.legend(loc='upper left')
plt.xlabel(r'$\log{M*/M_\odot}$', fontsize=15)
plt.ylabel('Ocorrencias (norm) ', fontsize=15)
fig9.savefig('graficos/starmass.png')
plt.close(fig9)
