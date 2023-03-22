from math import sin,cos,tan,pi,floor,log10,sqrt,atan,degrees
import numpy as np
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
import sys

eta=[]
rff=[]

delta=[]
gama=[]
cluster=[]

with open('pre_parser_info_clean_r4.dat','r') as inp3:
	ninp3=len(inp3.readlines())
inp3=open('pre_parser_info_clean_r4.dat','r')

with open('pargal_L07_16.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_16.dat','r')
flags=[[] for i in range(0,ninp1)]
flag1=[]
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls3=inp3.readline()
	ll3=ls3.split()
	eta.append((float(ll1[7]) - float(ll1[9])))
	rff.append(np.log10(float(ll1[7])))
	cluster.append(ll1[0])
	delta.append(float(ll1[10]))
	gama.append((float(ll1[11])))	
	flag1.append(ll3[2])
#	flags[ik].append(ll3[3])
#	flags[ik].append(ll3[4])
#	flags[ik].append(ll3[5])
#	flags[ik].append(ll3[6])
	####################

clt=[]
for i in range(len(cluster)):
	if delta[i] > 0.2 and flag1[i] == '1':
		clt.append(cluster[i])
		print cluster[i],rff[i],np.power(10,rff[i]),delta[i],gama[i],flag1[i]
		
print len(clt)

'''
bins=np.arange(0.0,0.2,0.02)
bins1=np.arange(0.0,0.9,0.02)

fig0=plt.figure()
plt.hist(gama,bins=bins,histtype='bar',facecolor='blue',edgecolor='black')
plt.xlabel('gama')
plt.savefig('gama_hist_v4.png')
#plt.show()
plt.close(fig0)

fig0=plt.figure()
plt.scatter(delta,rff,edgecolor='black')
#plt.xlim([-2.5,-0.7])
#plt.ylim([-0.01,0.1])
plt.xlabel('delta')
plt.ylabel('log rff')
plt.savefig('deltaxrff.png')
#plt.show()
plt.close(fig0)


fig0=plt.figure()
plt.hist(delta,bins=bins1,histtype='bar',facecolor='blue',edgecolor='black')
#plt.xlim([-2.5,-0.7])
#plt.ylim([-0.01,0.1])
plt.xlabel('delta')
#plt.ylabel(r'$\eta$')
plt.savefig('delta_hist_v4.png')
#plt.show()
plt.close(fig0)
'''
