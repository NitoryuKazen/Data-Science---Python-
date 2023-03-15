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
data_indiv=open('data_indiv_ecd.dat','r')
for bcg in data_indiv.readlines():
	ra_l07.append(float(bcg.split()[5]))
	obj.append(bcg.split()[0])
	#print bcg.split()[5]

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

temp=[]
pargal=open('pargal_L07_compact_v2.dat','r')
for tt in pargal.readlines():
	temp.append(tt[:-2])

ii=[]
jj=[]
ra_temp=np.asarray(ra)
pargal_casjobs=open('pargal_L07_compact_casjobs_v2.dat','w')
for i in range(len(ra_l07)):
	if min(np.abs(ra_temp - ra_l07[i])) < 0.001:
		galaxia = np.abs(ra_temp - ra_l07[i]).argmin()
		pargal_casjobs.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \n'%(temp[i],magabs[np.abs(ra_temp - ra_l07[i]).argmin()],logmass[np.abs(ra_temp - ra_l07[i]).argmin()],age[np.abs(ra_temp - ra_l07[i]).argmin()],metal[np.abs(ra_temp - ra_l07[i]).argmin()],r90[np.abs(ra_temp - ra_l07[i]).argmin()],r50[np.abs(ra_temp - ra_l07[i]).argmin()]))

	else:
		pargal_casjobs.write('%s \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \n'%(temp[i]))

		
'''
print jj, ii, len(jj) 

aa=[]

for i in range(len(obj)):
	if obj[i] not in jj:
		aa.append(obj[i])

print aa, len(aa)

for i in range(len(obj)):
	if obj[i] in jj:
		print obj[i],ii[i],i
		pargal_casjobs.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \n'%(temp[i],magabs[ii[i]],logmass[ii[i]],age[ii[i]],metal[ii[i]],r90[ii[i]],r50[ii[i]]))
	elif obj[i] in aa:
#			print i
		pargal_casjobs.write('%s \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \n'%(temp[i]))

#			ii.append(i)
#			jj.append(j)
'''
