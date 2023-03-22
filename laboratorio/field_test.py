from math import sin,cos,tan,pi,floor,log10,sqrt,atan2
import numpy as np
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits


ra_l07=[]
obj=[]
data_indiv=open('data_indiv_clean.dat','r')
for bcg in data_indiv.readlines():
	ra_l07.append(float(bcg.split()[5]))
	obj.append(bcg.split()[0])
	#print bcg.split()[5]

print ra_l07
ra=[]
magabs=[]
logmass=[]
age=[]
metal=[]
r90=[]
r50=[]
out_morph=open('L07_casjobs_morf.dat','r')
for item in out_morph.readlines():
	if item.split(',')[0] == 'name':
		pass
	else:
		ra.append(float(item.split(',')[1]))
		magabs.append(float(item.split(',')[4]))
		logmass.append(float(item.split(',')[5]))
		age.append(float(item.split(',')[6]))
		metal.append(float(item.split(',')[7]))
		r90.append(float(item.split(',')[8]))
		r50.append(float(item.split(',')[9]))
print ra

temp=[]
pargal=open('pargal_L07_compact.dat','r')
for tt in pargal.readlines():
	temp.append(tt[:-2])
ii=[]
jj=[]
pargal_casjobs=open('pargal_L07_compact_casjobs.dat','w')
for i in range(len(ra_l07)):
	for j in range(len(ra)):
		#print ra_l07[i],ra[j]
		if abs(ra_l07[i] - ra[j]) < 0.001:
			print ra_l07[i],ra[j]
			#ii.append(i)
			pargal_casjobs.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \n'%(temp[i],magabs[j],logmass[j],age[j],metal[j],r90[j],r50[j]))
			break
		#else:
			#print i
		#	pargal_casjobs.write('%s \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \n'%(temp[i]))
		#	pass
		#break	
#			ii.append(i)
#			jj.append(j)
































