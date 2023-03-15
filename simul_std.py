from math import sin,cos,tan,pi,floor,log10,sqrt,atan2
import numpy as np
import pyfits
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
import datetime 
from scipy.optimize import curve_fit
cl=[]
distall=[]
valall=[]
ok=[]
inp0=open('fail_simul.dat','a')
with open('data_indiv_clean.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_clean.dat','r')
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	cluster=ll1[0]	
	print ll1[0]
	if os.path.isfile('/mnt/Dados1/andre/WHL/'+cluster+'/ajust-bcg-r-exp.fits') == False or os.path.isfile('/mnt/Dados1/andre/WHL/'+cluster+'/bcg-simul-exp.fits'):
		pass
	else:
#
		mask = pyfits.getdata(cluster+'/bcg_r_mask.fits')
		mask2= pyfits.getdata(cluster+'/bcg_r_mask_b.fits')
#
		data = pyfits.getdata(cluster+'/ajust-bcg-r.fits',1)
		data1 = pyfits.getdata(cluster+'/ajust-bcg-r.fits',2)	
		header = pyfits.getheader(cluster+'/ajust-bcg-r.fits',2)
		header1=pyfits.getheader(cluster+'/ajust-bcg-r.fits',1)
#
		data_exp = pyfits.getdata(cluster+'/ajust-bcg-r-exp.fits',1)
		data1_exp = pyfits.getdata(cluster+'/ajust-bcg-r-exp.fits',2)	
		header_exp = pyfits.getheader(cluster+'/ajust-bcg-r-exp.fits',2)
		header1_exp=pyfits.getheader(cluster+'/ajust-bcg-r-exp.fits',1)
#
		data0=copy.deepcopy(data1)
#	
		data0_exp=copy.deepcopy(data1_exp)

#/mnt/Dados1/andre/L07/sersic_unid/high/
####
		print 'ok'
#		call('mkdir /mnt/Dados1/andre/L07/model_zhao_n/'+cluster+' ',shell=True)
		vec=[]		
		for i in range(0,data.shape[0]):
			for o in range(0,data.shape[1]):
				if mask[i,o] == 0 and mask2[i,o] == 0:
					vec.append(data[i,o])
		rows=data.shape[0]
		cols=data.shape[1]

		print cols,rows, len(vec)

####

		if len(vec) == 0 or len(vec) > 500000 or cols > len(vec):
			inp0.write('%s\n'%(ll1[0]))
		else:		
			x=np.random.rand(rows, len(vec)-1).argpartition(cols,axis=1)[:,:cols]+1

			print x.shape
			for i in range(0,data.shape[0]):
				for j in range(0,data.shape[1]):
					data0[i,j]+=vec[x[i,j]]
					data0_exp[i,j]+=vec[x[i,j]]
			print 'aqui'
		
			pyfits.writeto('/mnt/Dados1/andre/WHL/'+ll1[0]+'/bcg-simul-n.fits',data0,header=header1,clobber=True)
			pyfits.writeto('/mnt/Dados1/andre/WHL/'+ll1[0]+'/bcg-simul-exp.fits',data0_exp,header=header1_exp,clobber=True)


inp0.close()
