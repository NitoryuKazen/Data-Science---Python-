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

'''

10 = assimetria abraham = 9
12 = rff = 7

'''

etaenv0=[]
rffenv0=[]

etaenv1=[]
rffenv1=[]

etaenv2=[]
rffenv2=[]

etaenv3=[]
rffenv3=[]

etaenv4=[]
rffenv4=[]

etaenv5=[]
rffenv5=[]

etaassim=[]
rffassim=[]

etares=[]
rffres=[]

clenv0=[]
clenv1=[]
clenv2=[]
clenv3=[]
clenv4=[]
clenv5=[]

classim=[]

clres=[]

eta=[]
rff=[]
cl=[]
t=[]
tt=[]

eta_cut=[[] for _ in xrange(9)]
rffcut=[]
#1247 	 1176.183000 	 25.026500 	 13.486400 	 3.975700 	 0.908700 	 -46.826200 	 nan 	 0.165555 	 nan
#retirado via nan value do rff
#inp2=open('pargal_type_split_clean.dat','w')
inp3=open('pargal_cut_split_clean.dat','w')

with open('data_indiv_clean.dat','r') as inpd:
	ninpd=len(inpd.readlines())
inpd=open('data_indiv_clean.dat','r')

with open('pargal_L07_clean.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_clean.dat','r')
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls2=inpd.readline()
	ll2=ls2.split()
	if ll1[0] != '1247' and ll2[0] != '1247':
		eta.append((float(ll1[7]) - float(ll1[9])))
		rff.append(np.log10(float(ll1[7])))
		rffcut.append(np.log10(float(ll1[7])))
		cluster=ll1[0]
		cl.append(ll1[0])
	####################
rffcut.sort()
for a in range(10):

	print rffcut[70+a*50]

	if (1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.5 and np.log10(float(ll1[7])) > -1.3:
		etaenv5.append((float(ll1[7]) - float(ll1[9])))
		rffenv5.append(np.log10(float(ll1[7])))
		clenv5.append(ll1[0])
		#call('mkdir /home/andre/Projetos/L07/alto_rff/envelope/'+cluster+'',shell=True)
		#call('cp /home/andre/Projetos/L07/'+cluster+'/ajust-bcg-r.fits /home/andre/Projetos/L07/alto_rff/envelope/'+cluster+'/',shell=True)
		#call('cp /home/andre/Projetos/L07/'+cluster+'/bcg_r_mask* /home/andre/Projetos/L07/alto_rff/envelope/'+cluster+'/',shell=True)		
#		inp2.write('%s \t envelope_5 \t %s \n'%(ll1[0],ll2[7]))
	'''
	elif 0.3<(1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.4 and np.log10(float(ll1[7])) > -1.3:
		etaenv4.append((float(ll1[7]) - float(ll1[9])))
		rffenv4.append(np.log10(float(ll1[7])))
		clenv4.append(ll1[0])
#		inp2.write('%s \t envelope_4 \t %s \n'%(ll1[0],ll2[7]))
		
	elif 0.2<(1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.3 and np.log10(float(ll1[7])) > -1.3:
		etaenv3.append((float(ll1[7]) - float(ll1[9])))
		rffenv3.append(np.log10(float(ll1[7])))
		clenv3.append(ll1[0])
#		inp2.write('%s \t envelope_3 \t %s \n'%(ll1[0],ll2[7]))

	elif 0.1<(1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.2 and np.log10(float(ll1[7])) > -1.3:
		etaenv2.append((float(ll1[7]) - float(ll1[9])))
		rffenv2.append(np.log10(float(ll1[7])))
		clenv2.append(ll1[0])
#		inp2.write('%s \t envelope_2 \t %s \n'%(ll1[0],ll2[7]))
		
	elif 0.0<(1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.1 and np.log10(float(ll1[7])) > -1.3:
		etaenv1.append((float(ll1[7]) - float(ll1[9])))
		rffenv1.append(np.log10(float(ll1[7])))
		clenv1.append(ll1[0])
#		inp2.write('%s \t envelope_1 \t %s \n'%(ll1[0],ll2[7]))

	elif (1-(float(ll1[7]) - float(ll1[9]))/float(ll1[7])) < 0.0 and np.log10(float(ll1[7])) > -1.3:
		etaenv0.append((float(ll1[7]) - float(ll1[9])))
		rffenv0.append(np.log10(float(ll1[7])))
		clenv0.append(ll1[0])
#		inp2.write('%s \t envelope_0 \t %s \n'%(ll1[0],ll2[7]))
	'''	
	elif (float(ll1[7]) - float(ll1[9]))/float(ll1[7]) < 0.5 and np.log10(float(ll1[7])) > -1.3:
		etaassim.append((float(ll1[7]) - float(ll1[9])))
		rffassim.append(np.log10(float(ll1[7])))
		classim.append(ll1[0])
		#call('mkdir /home/andre/Projetos/L07/alto_rff/assimetrica/'+cluster+'',shell=True)
		#call('cp /home/andre/Projetos/L07/'+cluster+'/ajust-bcg-r.fits /home/andre/Projetos/L07/alto_rff/assimetrica/'+cluster+'/',shell=True)
		#call('cp /home/andre/Projetos/L07/'+cluster+'/bcg_r_mask* /home/andre/Projetos/L07/alto_rff/assimetrica/'+cluster+'/',shell=True)		

#		inp2.write('%s \t assimetrica \t %s \n'%(ll1[0],ll2[7]))
	else:
		etares.append(float(ll1[7]) - float(ll1[9]))
		rffres.append(np.log10(float(ll1[7])))
		clres.append(ll1[0])
#		inp2.write('%s \t eliptica \t %s \n'%(ll1[0],ll2[7]))
	
	##################
#inp2.close()

cuts=[]

rffcut.sort()
print len(rff)
#for i in range(len(rffcut)):
for j in range(len(rff)):
	if rff[j] <= rffcut[70]:
		inp3.write('%s \t cut_0 \n'%(cl[j]))
	elif rffcut[70] < rff[j] <= rffcut[120]:
		inp3.write('%s \t cut_1 \n'%(cl[j]))
	elif rffcut[120] < rff[j] <= rffcut[170]:
		inp3.write('%s \t cut_2 \n'%(cl[j]))
	elif rffcut[170] < rff[j] <= rffcut[220]:
		inp3.write('%s \t cut_3 \n'%(cl[j]))
	elif rffcut[220] < rff[j] <= rffcut[270]:
		inp3.write('%s \t cut_4 \n'%(cl[j]))
	elif rffcut[270] < rff[j] <= rffcut[320]:
		inp3.write('%s \t cut_5 \n'%(cl[j]))
	elif rffcut[320] < rff[j] <= rffcut[370]:
		inp3.write('%s \t cut_6 \n'%(cl[j]))
	elif rffcut[370] < rff[j] <= rffcut[420]:
		inp3.write('%s \t cut_7 \n'%(cl[j]))
	elif rffcut[420] < rff[j] <= rffcut[470]:
		inp3.write('%s \t cut_8 \n'%(cl[j]))
	elif rffcut[470] < rff[j] <= rffcut[520]:
		inp3.write('%s \t cut_9 \n'%(cl[j]))
	else:
		print cl[j],rff[j]
'''
fig0=plt.figure()

plt.scatter(rff,eta,s=20,c='green',edgecolors='black')
plt.scatter(rffenv5,etaenv5,s=20,c='blue',edgecolors='black')
plt.scatter(rffenv4,etaenv4,s=20,c='blue',edgecolors='black')
plt.scatter(rffenv3,etaenv3,s=20,c='blue',edgecolors='black')
plt.scatter(rffenv2,etaenv2,s=20,c='blue',edgecolors='black')
plt.scatter(rffenv1,etaenv1,s=20,c='blue',edgecolors='black')
plt.scatter(rffenv0,etaenv0,s=20,c='blue',edgecolors='black')

plt.scatter(rffassim,etaassim,s=20,c='red',edgecolors='black')
x=np.linspace(0.001,0.3,1000)
fraclist=[0.5,0.4,0.3,0.2,0.1,0.0]
color_idx = np.linspace(1, 0, len(fraclist))

for item in fraclist:
	plt.plot(np.log10(x),x-item*x,label=str(item),color=plt.cm.plasma(color_idx[fraclist.index(item)]))

#plt.legend()

for a in range(9):
	plt.plot([rffcut[71+a*50],rffcut[71+a*50]],[-0.01,0.1],c='black',linestyle='--')

plt.xlim([-2.5,-0.7])
plt.ylim([-0.01,0.1])
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.show()
plt.savefig('rffxeta_zhao_2lines_clean.png')
plt.close(fig0)
'''
