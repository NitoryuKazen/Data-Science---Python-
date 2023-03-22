from math import sin,cos,tan,pi,floor,log10,sqrt
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
###########################################

def lnt(cluster,X,Y):

#PROGRAMA PARTE 1 

	par = open('pargal_L07_exp.dat','a')
	info_err=open('aux_pargal_L07_exp.dat','a')
	
	os.chdir(cluster+'/')
########
	call('./galfit feedme_exp.r',shell=True)
	if os.path.isfile('ajust-bcg-exp.fits'):
		ajust1 = pyfits.getdata('ajust-bcg-exp.fits',1)
		header = pyfits.getheader('ajust-bcg-exp.fits',2)
		ajust3 = pyfits.getdata('ajust-bcg-exp.fits',3)
		mask = pyfits.getdata('bcg_r_mask.fits')
		mask_b = pyfits.getdata('bcg_r_mask_b.fits')
	##############################################################
		re = float(header['1_RE'].split()[0].replace('*',''))
		mag = float(header['1_MAG'].split()[0].replace('*',''))
		n = float(header['1_N'].split()[0].replace('*',''))
		axis = float(header['1_AR'].split()[0].replace('*',''))
		pa = float(header['1_PA'].split()[0].replace('*',''))
		chi2 = float(header['CHI2NU'])
		xc=float(header['1_XC'].split()[0].replace('*',''))
		yc=float(header['1_YC'].split()[0].replace('*',''))

		rs = float(header['2_RS'].split()[0].replace('*',''))
		magd = float(header['2_MAG'].split()[0].replace('*',''))
		axisd = float(header['2_AR'].split()[0].replace('*',''))
		pad = float(header['2_PA'].split()[0].replace('*',''))

	##################################################################3
		re_err = float(header['1_RE'].split()[2].replace('*',''))
		mag_err = float(header['1_MAG'].split()[2].replace('*',''))
		n_err = float(header['1_N'].split()[2].replace('*',''))
		axis_err = float(header['1_AR'].split()[2].replace('*',''))
		pa_err = float(header['1_PA'].split()[2].replace('*',''))

		rs_err = float(header['2_RS'].split()[2].replace('*',''))
		magd_err = float(header['2_MAG'].split()[2].replace('*',''))
		axisd_err = float(header['2_AR'].split()[2].replace('*',''))
		pad_err = float(header['2_PA'].split()[2].replace('*',''))

	#######################################################################################
		#CALCULO DO DELTA
		dist_c=[]
		cord=[[],[]]
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				if mask_b[i,j] == 1:
					dist_c.append(float(np.sqrt((i-Y)**2+(j-X)**2)))
					cord[0].append(j)
					cord[1].append(i)
		
		badxc=np.average(cord[0])	
		badyc=np.average(cord[1])
		
		goodcenter_dist=float(np.sqrt((badyc-Y)**2+(badxc-X)**2))

		badcenter_dist=[]
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				if mask_b[i,j] == 1:
					badcenter_dist.append(float(np.sqrt((i-badyc)**2+(j-badxc)**2)))
		dist_bad=np.average(badcenter_dist)
		
		delta=float(goodcenter_dist/dist_bad)
	###############################################################################
		p=0.
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				if mask_b[i,j] == 1:
					p+=1.
		d=0.
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				if mask[i,j] == 1 and mask_b[i,j] == 1:
					d+=1.
					
		if p == 0.:
			gama=1.
		else:
			gama=float(d/p)
	###############################################################################

		#CALCULO DO RFF
		xs3=[]
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				if mask_b[i,j]==0 and mask[i,j]==0:
					xs3.append(ajust1[i,j])
		sbk=np.std(xs3)
		xy=0
		xn=0
		nn2=0
		for i in range(0,ajust1.shape[0]): 
			for j in range(0,ajust1.shape[1]): 
				nn2+=(1.-mask[i,j])*(mask_b[i,j])
				xy+=np.absolute((ajust3[i,j])*((1.-mask[i,j])*(mask_b[i,j])))
				xn+=(ajust1[i,j])*((1.-mask[i,j])*(mask_b[i,j]))
		rff=(xy-0.8*sbk*nn2)/xn	
	##############################################################
	# INICIO DA ASSIMETRIA

	#IMG SMALL 
		xxc = X-5
		xxc2 = X+5
		yyc = Y-5
		yyc2 = Y+5
		data0 = ajust1[yyc:yyc2,xxc:xxc2]# img bcg
		data01 = mask[yyc:yyc2,xxc:xxc2]# mask normal 
		data02 = mask_b[yyc:yyc2,xxc:xxc2]# mask bcg
		
		# CALCULO DA ASSIMETRIA 10X10
		dxb = 0 
		dyb = 0
		speb = 0 
		for a in range(1000):
			matrix = ([1,0],[0,1])
			dx = np.random.normal(dxb,0.1)
			dy = np.random.normal(dyb,0.1)			
			data30 = ssn.interpolation.affine_transform(data0,matrix,offset=[dx,dy],mode='constant')# img bcg
			data31 = ssn.interpolation.affine_transform(data02,matrix,offset=[dx,dy],mode='constant')# mask bcg
			data32 = ssn.interpolation.affine_transform(data01,matrix,offset=[dx,dy],mode='constant')# mask normal
			data30r= np.rot90((data30),2)#img bcg rot
			data31r=np.rot90((data31),2)#mask bcg rot
			data32r=np.rot90((data32),2)#mask normal rot
	#############
			vas=[]#bcg normal
			var=[]#bcg rotacionada
			for i in range(0,yyc2-yyc):
				for j in range(0,xxc2-xxc):
					if data30[i,j] != 0.0 and data30r[i,j] != 0.0:
						if data32[i,j] == 0 and data31[i,j] == 1 and data32r[i,j] == 0 and data31r[i,j] == 1:
							vas.append(data30[i,j])
							var.append(data30r[i,j])
			spe = (1-sst.spearmanr(vas,var)[0])
			if spe < speb or speb == 0:
				speb = spe
				dxb = dx
				dyb = dy
	################################################################
	# ASSIMETRIA GRANDE
		matrix = ([1,0],[0,1])
		data30 = ssn.interpolation.affine_transform(ajust1,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# img bcg
		data31 = ssn.interpolation.affine_transform(mask_b,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask bcg
		data32 = ssn.interpolation.affine_transform(mask,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask normal
		data30r= np.rot90((data30),2)#img bcg rot
		data31r=np.rot90((data31),2)#mask bcg rot
		data32r=np.rot90((data32),2)#mask normal rot

		vas=[]#bcg normal
		var=[]#bcg rotacionada
		datat=copy.deepcopy(data30)
		datatr=copy.deepcopy(data30r)	
		res=0
		vas1=0
		nn2=0
		for i in range(0,ajust1.shape[0]):
			for j in range(0,ajust1.shape[1]):
				if data30[i,j] != 0.0 and data30r[i,j] != 0.0:
					if data32[i,j] == 0 and data32r[i,j] == 0 and (data31[i,j] == 1  or data31r[i,j] == 1):
						vas.append(data30[i,j])
						var.append(data30r[i,j])
						datat[i,j]=1
						datatr[i,j]=1
						res+=np.absolute(data30[i,j]-data30r[i,j])
						vas1+=(data30[i,j])
						nn2+=1.
		A1=((res-1.127*sbk*nn2)/vas1)/2.
		A0 = (1-sst.spearmanr(vas,var)[0])

		par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\t %f \t %f \t %f \t %f\n'%(cluster,chi2,re,mag,n,axis,pa,rs,magd,axisd,pad,float(rff),float(A0),float(A1),delta,gama))
		par.close()
		info_err.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(cluster,re_err,mag_err,n_err,axis_err,pa_err,rs_err,magd_err,axisd_err,pad_err))

		call('rm bcg_r.fits',shell=True)
		call('rm sigma-r.fits',shell=True)
		call('rm galfit.*',shell=True)
		call('rm bcg_r_psf*',shell=True)
		call('rm check*',shell=True)
	else:
		par.write('%s \t fail \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \n'%(cluster))
	os.chdir('../')
	return
	
'''
HEADER DE TUDO QUE SAI DO ARQUIVO pargal_*.dat aux_pargal_*.dat (RESPECTIVAMENTE):

0-ID
1-CHI2
2-RAIO EFETIVO
3-MAGNITUDE
4-N DE SERSIC
5-AXIS
6-ANGULO
7-RFF
8-ASSIMETRIA(SPEARMAN)
9-ASSIMETRIA(ABRAHAM)

0-ID
1-RAIO EFETIVO(INCERTEZA)
2-MAGNITUDE(INCERTEZA)
3-N DE SERSIC(INCERTEZA)
4-AXIS(INCERTEZA)
5-ANGULO(INCERTEZA)
'''
######################################################################

ok=[]
with open('pargal_L07_exp.dat','r') as inp2:
	for item in inp2.readlines():	
		ok.append(item.split()[0])
with open('data_indiv_clean.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_clean.dat','r')

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	cluster=ll1[0]
	run=str(ll1[2]).zfill(6)
	camcol=str(ll1[3])
	field=str(ll1[4]).zfill(4)
	if cluster in ok:
		pass
	else:
		print cluster

		header = pyfits.getheader(cluster+'/ajust-bcg-r.fits',1)
		data = pyfits.getdata(cluster+'/ajust-bcg-r.fits',1)
		header2 = pyfits.getheader(cluster+'/ajust-bcg-r.fits',2)

		EXPTIME=float(header['EXPTIME'])
		NMGY=float(header['NMGY'])
		GAIN=float(header['GAIN'])

	##############################################################
		re = float(header2['1_RE'].split()[0].replace('*',''))
		n = float(header2['1_N'].split()[0].replace('*',''))
		axis = float(header2['1_AR'].split()[0].replace('*',''))
		pa = float(header2['1_PA'].split()[0].replace('*',''))
		xc=float(header2['1_XC'].split()[0].replace('*',''))
		yc=float(header2['1_YC'].split()[0].replace('*',''))
		mag = float(header2['1_MAG'].split()[0].replace('*',''))
		magb= mag +2.5*log10(2.) 
		zeropoints=float(header2['MAGZPT'])
		
		pyfits.writeto(cluster+'/bcg_r.fits',data,header=header,clobber=True)
########################################################## 
# inicio do sigma.fits:

		mask = pyfits.getdata(cluster+'/bcg_r_mask.fits')
		mask_b = pyfits.getdata(cluster+'/bcg_r_mask_b.fits')
					
		sigma = copy.deepcopy(data)
		sig = sigma*GAIN
		vec = []
		for i in range(0,data.shape[0]):
			for o in range(0,data.shape[1]):
				if mask[i,o] == 0 and mask_b[i,o] == 0 : 
					vec.append(sig[i,o]) 	
		z = np.sum(np.power(vec-np.average(vec),2))/(len(vec)-1.)
		sig1 = (np.sqrt(np.absolute(sig)+z))/GAIN
		pyfits.writeto(cluster+'/sigma-r.fits',sig1,header=header,clobber=True)
#arquivo constr.all:

		ou3=open(cluster+'/constr.all','w')
		ou3.write(' 1 x -5 5 \n 1 y -5 5') 

# finish calculations and print input data to galfit
		ou1=open(cluster+'/feedme_exp.r','w')
		ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00   1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n\n # Object number: X MAIN SOURCE DISK \n 0) expdisk \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3)  %f \t 1 \t\t #  total   magnitude \n 4)  %f \t 1 \t\t#  Rs[pix] \n 9) %f \t 1 \t\t #  axis ratio (b/a) \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'%('bcg_r.fits','ajust-bcg-exp.fits','sigma-r.fits','bcg_r_psf_b.fits','bcg_r_mask.fits',1,data.shape[0],1,data.shape[1],zeropoints,xc,yc,mag,re/4.,axis,pa,xc,yc,magb,re,axis,pa))
		ou1.close()

		ou3.close()
		lnt(ll1[0],int(xc),int(yc))
###########################
