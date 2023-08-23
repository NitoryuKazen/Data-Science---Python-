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
###############################################
'''
0 - geral 
1 - nossas cDs
2 - nossas E
3 - nossas asy
4 - zhao cDs
5 - zhao E
6 - zhao cD/E + E/cD
'''


#19,22,23,26,27,30,31,34,35,38,39,44,49,51,53,54,55
rff,eta=[[[] for i in range(7)] for i in range(2)]
################################################################################################
med_a3,med_a4,med_b3,med_b4,med_disk=stats_info=[[[] for i in range(7)] for i in range(5)]

zhao_cd_env,zhao_cd_e,zhao_e_env,zhao_e_e=zhao_info=[[[] for i in range(5)] for i in range(4)]
####################################################################################################
af1_logfix,af2_logfix,bf2_logfix,af1_fix,af2_fix,bf2_fix = gr_fix = [[[] for i in range(7)] for i in range(6)]

af1_logfree,af2_logfree,bf2_logfree,af1_free,af2_free,bf2_free = gr_free = [[[] for i in range(7)] for i in range(6)]
###################################################################################################
gr_fix_cd_env,gr_fix_cd_e,gr_fix_e_env,gr_fix_e_e=zhao_gr_fix=[[[] for i in range(6)] for i in range(4)]

gr_free_cd_env,gr_free_cd_e,gr_free_e_env,gr_free_e_e=zhao_gr_free=[[[] for i in range(6)] for i in range(4)]

cluster=[]
############################################
with open('graph_stats_rff.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('graph_stats_rff.dat','r')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	rff[0].append(np.log10(float(ll1[44])))
	eta[0].append(float(ll1[45]))
	med_a3[0].append(float(ll1[19]))
	med_a4[0].append(float(ll1[20]))
	med_b3[0].append(float(ll1[21]))
	med_b4[0].append(float(ll1[22]))
	med_disk[0].append(float(ll1[23]))
	af1_logfix[0].append(float(ll1[24]))
	af2_logfix[0].append(float(ll1[26]))
	bf2_logfix[0].append(float(ll1[27]))
	af1_fix[0].append(float(ll1[29]))
	af2_fix[0].append(float(ll1[31]))
	bf2_fix[0].append(float(ll1[32]))
	af1_logfree[0].append(float(ll1[34]))
	af2_logfree[0].append(float(ll1[36]))
	bf2_logfree[0].append(float(ll1[37]))
	af1_free[0].append(float(ll1[39]))
	af2_free[0].append(float(ll1[41]))
	bf2_free[0].append(float(ll1[42]))
	############################################################################
	##########################################################################

	if (1-float(ll1[45])/float(ll1[44])) < 0.5 and np.log10(float(ll1[44])) > np.log10(0.0198):
		rff[1].append(np.log10(float(ll1[44])))
		eta[1].append(float(ll1[45]))
		med_a3[1].append(float(ll1[19]))
		med_a4[1].append(float(ll1[20]))
		med_b3[1].append(float(ll1[21]))
		med_b4[1].append(float(ll1[23]))
		med_disk[1].append(float(ll1[23]))
		af1_logfix[1].append(float(ll1[24]))
		af2_logfix[1].append(float(ll1[26]))
		bf2_logfix[1].append(float(ll1[27]))
		af1_fix[1].append(float(ll1[29]))
		af2_fix[1].append(float(ll1[31]))
		bf2_fix[1].append(float(ll1[32]))
		af1_logfree[1].append(float(ll1[34]))
		af2_logfree[1].append(float(ll1[36]))
		bf2_logfree[1].append(float(ll1[37]))
		af1_free[1].append(float(ll1[39]))
		af2_free[1].append(float(ll1[41]))
		bf2_free[1].append(float(ll1[42]))
		print(ll1[0],'envelope')
		if ll1[46] == 'cD':
			zhao_cd_env[0].append(float(ll1[19]))
			zhao_cd_env[1].append(float(ll1[20]))
			zhao_cd_env[2].append(float(ll1[21]))
			zhao_cd_env[3].append(float(ll1[22]))
			zhao_cd_env[4].append(float(ll1[23]))
			gr_fix_cd_env[0].append(float(ll1[24]))
			gr_fix_cd_env[1].append(float(ll1[26]))
			gr_fix_cd_env[2].append(float(ll1[27]))
			gr_fix_cd_env[3].append(float(ll1[29]))
			gr_fix_cd_env[4].append(float(ll1[31]))
			gr_fix_cd_env[5].append(float(ll1[32]))
			gr_free_cd_env[0].append(float(ll1[34]))
			gr_free_cd_env[1].append(float(ll1[36]))
			gr_free_cd_env[2].append(float(ll1[37]))
			gr_free_cd_env[3].append(float(ll1[39]))
			gr_free_cd_env[4].append(float(ll1[41]))
			gr_free_cd_env[5].append(float(ll1[42]))

		elif ll1[46] == 'E':
			zhao_e_env[0].append(float(ll1[19]))
			zhao_e_env[1].append(float(ll1[20]))
			zhao_e_env[2].append(float(ll1[21]))
			zhao_e_env[3].append(float(ll1[22]))
			zhao_e_env[4].append(float(ll1[23]))
			gr_fix_e_env[0].append(float(ll1[24]))
			gr_fix_e_env[1].append(float(ll1[26]))
			gr_fix_e_env[2].append(float(ll1[27]))
			gr_fix_e_env[3].append(float(ll1[29]))
			gr_fix_e_env[4].append(float(ll1[31]))
			gr_fix_e_env[5].append(float(ll1[32]))
			gr_free_e_env[0].append(float(ll1[34]))
			gr_free_e_env[1].append(float(ll1[36]))
			gr_free_e_env[2].append(float(ll1[37]))
			gr_free_e_env[3].append(float(ll1[39]))
			gr_free_e_env[4].append(float(ll1[41]))
			gr_free_e_env[5].append(float(ll1[42]))

	elif (float(ll1[45]))/float(ll1[44]) < 0.5 and np.log10(float(ll1[44])) > np.log10(0.0198):
		rff[3].append(np.log10(float(ll1[44])))
		eta[3].append(float(ll1[45]))
		med_a3[3].append(float(ll1[19]))
		med_a4[3].append(float(ll1[20]))
		med_b3[3].append(float(ll1[21]))
		med_b4[3].append(float(ll1[22]))
		med_disk[3].append(float(ll1[23]))
		af1_logfix[3].append(float(ll1[24]))
		af2_logfix[3].append(float(ll1[26]))
		bf2_logfix[3].append(float(ll1[27]))
		af1_fix[3].append(float(ll1[29]))
		af2_fix[3].append(float(ll1[31]))
		bf2_fix[3].append(float(ll1[32]))
		af1_logfree[3].append(float(ll1[34]))
		af2_logfree[3].append(float(ll1[36]))
		bf2_logfree[3].append(float(ll1[37]))
		af1_free[3].append(float(ll1[39]))
		af2_free[3].append(float(ll1[41]))
		bf2_free[3].append(float(ll1[42]))
		print(ll1[0],'assimetrica')
	elif np.log10(float(ll1[44])) < np.log10(0.0198):
		rff[2].append(np.log10(float(ll1[44])))
		eta[2].append(float(ll1[45]))
		med_a3[2].append(float(ll1[19]))
		med_a4[2].append(float(ll1[20]))
		med_b3[2].append(float(ll1[21]))
		med_b4[2].append(float(ll1[22]))
		med_disk[2].append(float(ll1[23]))
		af1_logfix[2].append(float(ll1[24]))
		af2_logfix[2].append(float(ll1[26]))
		bf2_logfix[2].append(float(ll1[27]))
		af1_fix[2].append(float(ll1[29]))
		af2_fix[2].append(float(ll1[31]))
		bf2_fix[2].append(float(ll1[32]))
		af1_logfree[2].append(float(ll1[34]))
		af2_logfree[2].append(float(ll1[36]))
		bf2_logfree[2].append(float(ll1[37]))
		af1_free[2].append(float(ll1[39]))
		af2_free[2].append(float(ll1[41]))
		bf2_free[2].append(float(ll1[42]))
		print(ll1[0],'eliptica')
		if ll1[46] == 'cD':
			zhao_cd_e[0].append(float(ll1[19]))
			zhao_cd_e[1].append(float(ll1[20]))
			zhao_cd_e[2].append(float(ll1[21]))
			zhao_cd_e[3].append(float(ll1[22]))
			zhao_cd_e[4].append(float(ll1[23]))
			gr_fix_cd_e[0].append(float(ll1[24]))
			gr_fix_cd_e[1].append(float(ll1[26]))
			gr_fix_cd_e[2].append(float(ll1[27]))
			gr_fix_cd_e[3].append(float(ll1[29]))
			gr_fix_cd_e[4].append(float(ll1[31]))
			gr_fix_cd_e[5].append(float(ll1[32]))
			gr_free_cd_e[0].append(float(ll1[34]))
			gr_free_cd_e[1].append(float(ll1[36]))
			gr_free_cd_e[2].append(float(ll1[37]))
			gr_free_cd_e[3].append(float(ll1[39]))
			gr_free_cd_e[4].append(float(ll1[41]))
			gr_free_cd_e[5].append(float(ll1[42]))

		elif ll1[46] == 'E':
			zhao_e_e[0].append(float(ll1[19]))
			zhao_e_e[1].append(float(ll1[20]))
			zhao_e_e[2].append(float(ll1[21]))
			zhao_e_e[3].append(float(ll1[22]))
			zhao_e_e[4].append(float(ll1[23]))
			gr_fix_e_e[0].append(float(ll1[24]))
			gr_fix_e_e[1].append(float(ll1[26]))
			gr_fix_e_e[2].append(float(ll1[27]))
			gr_fix_e_e[3].append(float(ll1[29]))
			gr_fix_e_e[4].append(float(ll1[31]))
			gr_fix_e_e[5].append(float(ll1[32]))
			gr_free_e_e[0].append(float(ll1[34]))
			gr_free_e_e[1].append(float(ll1[36]))
			gr_free_e_e[2].append(float(ll1[37]))
			gr_free_e_e[3].append(float(ll1[39]))
			gr_free_e_e[4].append(float(ll1[41]))
			gr_free_e_e[5].append(float(ll1[42]))

	#################################################################################
	###########################################################################
	
	if ll1[46] == 'cD':
		rff[4].append(np.log10(float(ll1[44])))
		eta[4].append(float(ll1[45]))
		med_a3[4].append(float(ll1[19]))
		med_a4[4].append(float(ll1[20]))
		med_b3[4].append(float(ll1[21]))
		med_b4[4].append(float(ll1[22]))
		med_disk[4].append(float(ll1[23]))
		af1_logfix[4].append(float(ll1[24]))
		af2_logfix[4].append(float(ll1[26]))
		bf2_logfix[4].append(float(ll1[27]))
		af1_fix[4].append(float(ll1[29]))
		af2_fix[4].append(float(ll1[31]))
		bf2_fix[4].append(float(ll1[32]))
		af1_logfree[4].append(float(ll1[34]))
		af2_logfree[4].append(float(ll1[36]))
		bf2_logfree[4].append(float(ll1[37]))
		af1_free[4].append(float(ll1[39]))
		af2_free[4].append(float(ll1[41]))
		bf2_free[4].append(float(ll1[42]))

	if ll1[46] == 'E':
		rff[5].append(np.log10(float(ll1[44])))
		eta[5].append(float(ll1[45]))
		med_a3[5].append(float(ll1[19]))
		med_a4[5].append(float(ll1[20]))
		med_b3[5].append(float(ll1[21]))
		med_b4[5].append(float(ll1[22]))
		med_disk[5].append(float(ll1[23]))
		af1_logfix[5].append(float(ll1[24]))
		af2_logfix[5].append(float(ll1[26]))
		bf2_logfix[5].append(float(ll1[27]))
		af1_fix[5].append(float(ll1[29]))
		af2_fix[5].append(float(ll1[31]))
		bf2_fix[5].append(float(ll1[32]))
		af1_logfree[5].append(float(ll1[34]))
		af2_logfree[5].append(float(ll1[36]))
		bf2_logfree[5].append(float(ll1[37]))
		af1_free[5].append(float(ll1[39]))
		af2_free[5].append(float(ll1[41]))
		bf2_free[5].append(float(ll1[42]))
	if ll1[46] == 'cD/E' or ll1[46] == 'E/cD':
		rff[6].append(np.log10(float(ll1[44])))
		eta[6].append(float(ll1[45]))
		med_a3[6].append(float(ll1[19]))
		med_a4[6].append(float(ll1[20]))
		med_b3[6].append(float(ll1[21]))
		med_b4[6].append(float(ll1[22]))
		med_disk[6].append(float(ll1[23]))
		af1_logfix[6].append(float(ll1[24]))
		af2_logfix[6].append(float(ll1[26]))
		bf2_logfix[6].append(float(ll1[27]))
		af1_fix[6].append(float(ll1[29]))
		af2_fix[6].append(float(ll1[31]))
		bf2_fix[6].append(float(ll1[32]))
		af1_logfree[6].append(float(ll1[34]))
		af2_logfree[6].append(float(ll1[36]))
		bf2_logfree[6].append(float(ll1[37]))
		af1_free[6].append(float(ll1[39]))
		af2_free[6].append(float(ll1[41]))
		bf2_free[6].append(float(ll1[42]))
#########################################
#VALORES ESTATISTICOS E CLEANER E MULTIPLICADOR
#0,1,2,3,4,5,6
'''
###################################################
medias_fix=[[] for i in range(len(gr_fix))]
medias_free=[[] for i in range(len(gr_free))]

for vec in gr_fix:
	for item in vec:
		medias_fix[gr_fix.index(vec)].append(np.average(item))
for vec in gr_free:
	for item in vec:
		medias_free[gr_free.index(vec)].append(np.average(item))
###############################################
medias_zhao_fix=[[] for i in range(len(zhao_gr_fix))]
medias_zhao_free=[[] for i in range(len(zhao_gr_free))]

for vec in zhao_gr_free:
	for item in vec:
		medias_zhao_free[zhao_gr_free.index(vec)].append(np.average(item))

#################################################
medias=[[] for i in range(len(stats_info))]

for vec in stats_info:
	for item in vec:
		medias[stats_info.index(vec)].append(np.average(item))

##############################################################################
medias_zhao=[[] for i in range(len(zhao_info))]

for vec in zhao_info:
	for item in vec:
		medias_zhao[zhao_info.index(vec)].append(np.average(item))

##########################################
#HISTOGRAMAS

#0 - geral 
#1 - nossas cDs
#2 - nossas E
#3 - nossas asy
#4 - zhao cDs
#5 - zhao E
#6 - zhao cD/E + E/cD
#------------------------------------------------#
#ORDEM
	#CD+E
	#CD_ZHAO+CD
	#CD_ZHAO+E
	#CD_ZHAO+E_ZHAO	

###########################################################################################################
#HISTOGRAMAS DE FOURIER

double_tuple=[(2,1,'green','blue',r'$E$',r'$cD$','_e_cd'),(4,1,'red','blue',r'$Zhao_{cD}$',r'$cD$','_zhao_cd_cd'),(4,2,'red','green',r'$Zhao_{cD}$',r'$E$','_zhao_cd_e'), (4,5,'red','purple',r'$Zhao_{cD}$',r'$Zhao_{E}$','_zhao_cd_zhao_e')]

name_tuple=[(med_a3,0,r'$a_3$',r'$_{med}$',medias,'med_a3'),(med_a4,1,r'$a_4$',r'$_{med}$',medias,'med_a4'),(med_b3,2,r'$b_3$',r'$_{med}$',medias,'med_b3'),(med_b4,3,r'$b_4$',r'$_{med}$',medias,'med_b4'),(med_disk,4,r'$(a4/SMA)$',r'$_{med}$',medias,'med_disk')]

fix_tuple=[(af1_logfix,0,r'$\alpha_lf$',medias_fix,'af1_logfix'),(af2_logfix,1,r'$a_lf$',medias_fix,'af2_logfix'),(bf2_logfix,2,r'$b_lf$',medias_fix,'bf2_logfix'),(af1_fix,3,r'$\alpha_f$',medias_fix,'af1_fix'),(af2_fix,4,r'$a_f$',medias_fix,'af1_fix'),(bf2_fix,5,r'$b_f$',medias_fix,'bf2_fix')]

free_tuple=[(af1_logfree,0,r'$\alpha_ll$',medias_free,'af1_logfree'),(af2_logfree,1,r'$a_ll$',medias_free,'af2_logfree'),(bf2_logfree,2,r'$b_ll$',medias_free,'bf2_logfree'),(af1_free,3,r'$\alpha_l$',medias_free,'af1_free'),(af2_free,4,r'$a_l$',medias_free,'af1_free'),(bf2_free,5,r'$b_l$',medias_free,'bf2_free')]

for fnome,i,sname,sub,medvec,strname in name_tuple:
	freq=np.abs(min(fnome[0]))+np.abs(max(fnome[0]))
	bins=np.arange(min(fnome[0]),max(fnome[0]),freq/20)

	for idx,idy,cx,cy,tx,ty,idsave in double_tuple:
		fig=plt.figure(tight_layout=True)
		plt.suptitle(sname+' '+tx+'-'+ty,fontsize=15)
		x1,x2,x3 = plt.hist(fnome[idx],bins=bins,edgecolor=cx,histtype='step',density=True,label=tx)
		y1,y2,y3 = plt.hist(fnome[idy],bins=bins,edgecolor=cy,histtype='step',density=True,label=ty)
		lmax=0
		if x1.max() >= y1.max():
			lmax=x1.max()
		else:
			lmax=y1.max()
		plt.plot([medvec[i][idx],medvec[i][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+sub+' '+format(medvec[i][idx],'.2E'))
		plt.plot([medvec[i][idy],medvec[i][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+sub+' '+format(medvec[i][idy],'.2E')) 
		plt.plot([],[],color=cy,linewidth=1,ls='--',label='KS'+ty+tx+' '+format(ks_2samp(fnome[idx],fnome[idy])[1],'.3E')) 
		plt.legend()
		plt.xlabel(sname,fontsize=10)
		plt.ylabel('Ocorrências (norm)', fontsize=10)
		fig.savefig('graficos_astro/'+strname+idsave+'.png')
		plt.show()
		plt.close(fig)

for fnome,i,sname,medvec,strname in fix_tuple:
	freq=np.abs(min(fnome[0]))+np.abs(max(fnome[0]))
	bins=np.arange(min(fnome[0]),max(fnome[0]),freq/20)

	for idx,idy,cx,cy,tx,ty,idsave in double_tuple:
		fig=plt.figure(tight_layout=True)
		plt.suptitle(sname+' '+tx+'-'+ty,fontsize=15)
		x1,x2,x3 = plt.hist(fnome[idx],bins=bins,edgecolor=cx,histtype='step',density=True,label=tx)
		y1,y2,y3 = plt.hist(fnome[idy],bins=bins,edgecolor=cy,histtype='step',density=True,label=ty)
		lmax=0
		if x1.max() >= y1.max():
			lmax=x1.max()
		else:
			lmax=y1.max()
		plt.plot([medvec[i][idx],medvec[i][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+' '+format(medvec[i][idx],'.2E'))
		plt.plot([medvec[i][idy],medvec[i][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+' '+format(medvec[i][idy],'.2E')) 
		plt.plot([],[],color=cy,linewidth=1,ls='--',label='KS'+ty+tx+' '+format(ks_2samp(fnome[idx],fnome[idy])[1],'.3E')) 
		plt.legend()
		plt.xlabel(sname,fontsize=10)
		plt.ylabel('Ocorrências (norm)', fontsize=10)
		fig.savefig('graficos_astro/'+strname+idsave+'.png')
		plt.show()
		plt.close(fig)

for fnome,i,sname,medvec,strname in free_tuple:
	freq=np.abs(min(fnome[0]))+np.abs(max(fnome[0]))
	bins=np.arange(min(fnome[0]),max(fnome[0]),freq/20)

	for idx,idy,cx,cy,tx,ty,idsave in double_tuple:
		fig=plt.figure(tight_layout=True)
		plt.suptitle(sname+' '+tx+'-'+ty,fontsize=15)
		x1,x2,x3 = plt.hist(fnome[idx],bins=bins,edgecolor=cx,histtype='step',density=True,label=tx)
		y1,y2,y3 = plt.hist(fnome[idy],bins=bins,edgecolor=cy,histtype='step',density=True,label=ty)
		lmax=0
		if x1.max() >= y1.max():
			lmax=x1.max()
		else:
			lmax=y1.max()
		plt.plot([medvec[i][idx],medvec[i][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+' '+format(medvec[i][idx],'.2E'))
		plt.plot([medvec[i][idy],medvec[i][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+' '+format(medvec[i][idy],'.2E')) 
		plt.plot([],[],color=cy,linewidth=1,ls='--',label='KS'+ty+tx+' '+format(ks_2samp(fnome[idx],fnome[idy])[1],'.3E')) 
		plt.legend()
		plt.xlabel(sname,fontsize=10)
		plt.ylabel('Ocorrências (norm)', fontsize=10)
		fig.savefig('graficos_astro/'+strname+idsave+'.png')
		plt.show()
		plt.close(fig)

#################################################################################################################
#HISTOGRAMAS DOS SLOPES
###################################################################################################
'''
'''
nosso_tuple=[(med_a3,medias,0,r'$a_3$ ',r'$_{med}$','med_a3'),(med_a4,medias,1,r'$a_4$',r'$_{med}$','med_a4'),(med_b3,medias,2,r'$b_3$',r'$_{med}$','med_b3'),(med_b4,medias,3,r'$b_4$',r'$_{med}$','med_b4'),(med_disk,medias,4,r'$(a4/SMA)$',r'$_{med}$','med_disk')]#,(slope_fix,medias_slope,0,r'$100\alpha_{fix}$',' '),(slope_free,medias_slope,1,r'$100\alpha_{free}$',' ')]

zhao_tuple=[(zhao_cd_env,r'$ZAC_{env} cD$',0,medias_zhao,'black',1,'blue',r'$cD$','zhao_cd_env'),(zhao_cd_e,r'$ZAC_{e} cD$',1,medias_zhao,'black',2,'green',r'$E$','zhao_cd_e'),(zhao_e_env,r'$ZAC_{env} E$',2,medias_zhao,'gray',1,'blue',r'$cD$','zhao_e_env'),(zhao_e_e,r'$ZAC_{e} E$',3,medias_zhao,'black',2,'green',r'$E$','zhao_e_e')]#,zhao_cd_e,zhao_e_env,zhao_e_e]

zhao_gr_fix_tuple=[(gr_fix_cd_env,r'$Zhao cD$',0,medias_zhao_slopes,'black',1,'blue',r'$cD$','zhao_gr_cd_env'),(zhao_gr_cd_e,r'$ZAC_{e} cD$',1,medias_zhao_slopes,'black',2,'green',r'$E$','zhao_gr_cd_e'),(zhao_gr_e_env,r'$ZAC_{env} E$',2,medias_zhao_slopes,'gray',1,'blue',r'$cD$','zhao_gr_e_env'),(zhao_gr_e_e,r'$ZAC_{e} E$',3,medias_zhao_slopes,'black',2,'green',r'$E$','zhao_gr_e_e')]

zhao_gr_tuple=[(zhao_gr_cd_env,r'$ZAC_{env} cD$',0,medias_zhao_slopes,'black',1,'blue',r'$cD$','zhao_gr_cd_env'),(zhao_gr_cd_e,r'$ZAC_{e} cD$',1,medias_zhao_slopes,'black',2,'green',r'$E$','zhao_gr_cd_e'),(zhao_gr_e_env,r'$ZAC_{env} E$',2,medias_zhao_slopes,'gray',1,'blue',r'$cD$','zhao_gr_e_env'),(zhao_gr_e_e,r'$ZAC_{e} E$',3,medias_zhao_slopes,'black',2,'green',r'$E$','zhao_gr_e_e')]

fix_tuple=[(af1_logfix,0,r'$\alpha_lf$',medias_zhao_fix,'af1_logfix'),(af2_logfix,1,r'$a_lf$',medias_zhao_fix,'af2_logfix'),(bf2_logfix,2,r'$b_lf$',medias_zhao_fix,'bf2_logfix'),(af1_fix,3,r'$\alpha_f$',medias_zhao_fix,'af1_fix'),(af2_fix,4,r'$a_f$',medias_zhao_fix,'af1_fix'),(bf2_fix,5,r'$b_f$',medias_zhao_fix,'bf2_fix')]

free_tuple=[(af1_logfree,0,r'$\alpha_ll$',medias_zhao_free,'af1_logfree'),(af2_logfree,1,r'$a_ll$',medias_zhao_free,'af2_logfree'),(bf2_logfree,2,r'$b_ll$',medias_zhao_free,'bf2_logfree'),(af1_free,3,r'$\alpha_l$',medias_zhao_free,'af1_free'),(af2_free,4,r'$a_l$',medias_zhao_free,'af1_free'),(bf2_free,5,r'$b_l$',medias_zhao_free,'bf2_free')]


#nosso_gr_tuple=[(slope_fix,medias_slopes,0,r'$\alpha_{fix}$',' ','slope_fix'),(slope_free,medias_slopes,1,r'$\alpha_{free}$',' ','slope_free')]

#gr_fix_cd_env,gr_fix_cd_e,gr_fix_e_env,gr_fix_e_e=zhao_gr_fix=[[[] for i in range(6)] for i in range(4)]

#gr_free_cd_env,gr_free_cd_e,gr_free_e_env,gr_free_e_e=zhao_gr_free=[[[] for i in range(6)] for i in range(4)]

for zname,szname,k,medzhao,cz,idx,cx,tx,ztrname in zhao_tuple:
	for z in range(len(zname)):
		fnome,medvec,j,sname,sub,strname = nosso_tuple[z]
		
		freq=np.abs(min(fnome[0]))+np.abs(max(fnome[0]))
		bins=np.arange(min(fnome[0]),max(fnome[0]),freq/15)

		fig=plt.figure(tight_layout=True)
		plt.suptitle(sname+' '+szname,fontsize=15)
		x1,x2,x3 = plt.hist(fnome[idx],bins=bins,edgecolor=cx,histtype='step',density=True,label=tx)
		y1,y2,y3 = plt.hist(zname[z],bins=bins,edgecolor=cz,histtype='step',density=True,label=szname)
		lmax=0
		if x1.max() >= y1.max():
			lmax=x1.max()
		else:
			lmax=y1.max()
		plt.plot([medvec[j][idx],medvec[j][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+' '+sub+' '+format(medvec[j][idx],'.2E'))
		plt.plot([medzhao[k][z],medzhao[k][z]],[0,lmax],color=cz,linewidth=1,ls='--',label=szname+' '+format(medzhao[k][z],'.2E')) 
		plt.plot([],[],color=cy,linewidth=1,ls='--',label='KS'+ty+szname+' '+format(ks_2samp(fnome[idx],zname[z])[1],'.2E')) 
		plt.legend()
		plt.xlabel(sname, fontsize=10)
		plt.ylabel('Ocorrências (norm)', fontsize=10)
		#plt.show()
		fig.savefig('graficos_astro/'+str(strname)+str(ztrname)+'.png')
		plt.close(fig)




for zname,szname,k,medzhao,cz,idx,cx,tx,ztrname in zhao_gr_tuple:
	for q in range(len(zhao_gr_tuple)):
		for z in range(len(zname)):
			fnome,medvec,j,sname,sub,strname = nosso_gr_tuple[z]
			
			freq=np.abs(min(fnome[0]))+np.abs(max(fnome[0]))
			bins=np.arange(min(fnome[0]),max(fnome[0]),freq/15)

			fig=plt.figure(tight_layout=True)
			plt.suptitle(sname+' '+szname,fontsize=15)
			x1,x2,x3 = plt.hist(fnome[idx],bins=bins,edgecolor=cx,histtype='step',density=True,label=tx)
			y1,y2,y3 = plt.hist(zname[z],bins=bins,edgecolor=cz,histtype='step',density=True,label=szname)
			lmax=0
			if x1.max() >= y1.max():
				lmax=x1.max()
			else:
				lmax=y1.max()
			plt.plot([medvec[j][idx],medvec[j][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+' '+sub+' '+format(medvec[j][idx],'.2E'))
			plt.plot([medzhao[q][z],medzhao[q][z]],[0,lmax],color=cz,linewidth=1,ls='--',label=szname+' '+format(medzhao[q][z],'.2E')) 
			plt.plot([],[],color=cy,linewidth=1,ls='--',label='KS'+ty+szname+' '+format(ks_2samp(fnome[idx],zname[z])[1],'.2E'))

			plt.legend()
			plt.xlabel(sname, fontsize=10)
			plt.ylabel('Ocorrências (norm)', fontsize=10)
			#plt.show()
			fig.savefig('graficos_astro/'+str(strname)+str(ztrname)+'.png')
			plt.close(fig)
			'''

#COEFICIENTE A3
'''
a3_freq=np.abs(min(med_a3[0]))+np.abs(max(med_a3[0]))
a3_bins=np.arange(min(med_a3[0]),max(med_a3[0]),a3_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_a3=plt.figure(tight_layout=True)
	plt.suptitle(r' $100a_3$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(med_a3[idx],bins=a3_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(med_a3[idy],bins=a3_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100a_3$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	plt.show()
#	fig_a3.savefig('graficos_astro/100a3_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_a3)
########################################################################################################
#COEFICIENTE A4

a4_freq=np.abs(min(med_a4[0]))+np.abs(max(med_a4[0]))

a4_bins=np.arange(min(med_a4[0]),max(med_a4[0]),a4_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_a4=plt.figure(tight_layout=True)
	plt.suptitle(r' $100a_4$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(med_a4[idx],bins=a4_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(med_a4[idy],bins=a4_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100a_4$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_a4.savefig('graficos_astro/100a4_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_a4)

#COEFICIENTE B3
b3_freq=np.abs(min(med_b3[0]))+np.abs(max(med_b3[0]))

b3_bins=np.arange(min(med_b3[0]),max(med_b3[0]),b3_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_b3=plt.figure(tight_layout=True)
	plt.suptitle(r' $100b_3$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(med_b3[idx],bins=b3_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(med_b3[idy],bins=b3_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100b_3$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_b3.savefig('graficos_astro/100b3_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_b3)

#COEFICIENTE B4
b4_freq=np.abs(min(med_b4[0]))+np.abs(max(med_b4[0]))

b4_bins=np.arange(min(med_b4[0]),max(med_b4[0]),b4_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_b4=plt.figure(tight_layout=True)
	plt.suptitle(r' $100b_4$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(med_b4[idx],bins=b4_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(med_b4[idy],bins=b4_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100b_4$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_b4.savefig('graficos_astro/100b4_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_b4)

#DISKNESS

disk_freq=np.abs(min(med_disk[0]))+np.abs(max(med_disk[0]))

disk_bins=np.arange(min(med_disk[0]),max(med_disk[0]),disk_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_disk=plt.figure(tight_layout=True)
	plt.suptitle(r' $100(a4/SMA)$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(med_disk[idx],bins=disk_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(med_disk[idy],bins=disk_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100(a4/SMA)$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_disk.savefig('graficos_astro/100disk_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_disk)

#SLOPE FREE
slope_free_freq=np.abs(min(slope_free[0]))+np.abs(max(slope_free[0]))

slope_free_bins=np.arange(min(slope_free[0]),max(slope_free[0]),slope_free_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_slope_free=plt.figure(tight_layout=True)
	plt.suptitle(r' $100\alpha_{free}$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(slope_free[idx],bins=slope_free_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(slope_free[idy],bins=slope_free_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100\alpha_{free}$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_slope_free.savefig('graficos_astro/100slope_free_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_slope_free)

#SLOPE FIX
slope_fix_freq=np.abs(min(slope_fix[0]))+np.abs(max(slope_fix[0]))

slope_fix_bins=np.arange(min(slope_fix[0]),max(slope_fix[0]),slope_fix_freq/15)

for idx,idy,cx,cy,tx,ty in double_tuple:
	fig_slope_fix=plt.figure(tight_layout=True)
	plt.suptitle(r' $100\alpha_{fix}$ '+tx+'-'+ty,fontsize=15)
	x1,x2,x3 = plt.hist(slope_fix[idx],bins=slope_fix_bins,edgecolor=cx,histtype='step',density=True,label=tx)
	y1,y2,y3 = plt.hist(slope_fix[idy],bins=slope_fix_bins,edgecolor=cy,histtype='step',density=True,label=ty)
	lmax=0
	if x1.max() >= y1.max():
		lmax=x1.max()
	else:
		lmax=y1.max()
	plt.plot([medias[0][idx],medias[0][idx]],[0,lmax],color=cx,linewidth=1,ls='--',label=tx+r'$_{med}$')
	plt.plot([medias[0][idy],medias[0][idy]],[0,lmax],color=cy,linewidth=1,ls='--',label=ty+r'$_{med}$') 
	plt.legend()
	plt.xlabel(r'$100\alpha_{fix}$', fontsize=10)
	plt.ylabel('Ocorrências (norm)', fontsize=10)
	fig_slope_fix.savefig('graficos_astro/100slope_fix_'+tx[1:-1]+'_'+ty[1:-1]+'.png')
	plt.close(fig_slope_fix)
	'''
################################################################################################
# RFXETA: COM LINHAS E SEM LINHAS
'''
fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[0.5]
color_idx = np.linspace(1, 0, len(fraclist))
for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))
plt.scatter(rff[0],eta[0],s=20,c=med_disk[0],cmap=plt.cm.get_cmap('jet'),vmin=-0.03,vmax=0.03,edgecolors='black')
plt.plot([np.log10(0.0198),np.log10(0.0198)],[-0.01,0.1],c='black',linestyle='--')
plt.legend()
cbar=plt.colorbar()
cbar.set_label('slope free', rotation=90)
plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
#plt.savefig('graficos_astro/rffxeta_slope_free.png')
plt.show()
plt.close(fig1)

fig1=plt.figure()
x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
fraclist=[0.5]
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
plt.savefig('graficos_astro/rffxeta.png')
plt.show()
plt.close(fig1)
'''

