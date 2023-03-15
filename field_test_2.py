from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp
import numpy as np
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as scp 
import scipy.interpolate as sci
import photutils.isophote as phi
import photutils.aperture as php
import numpy.ma as ma
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")


######
isoimage = fits.open('/home/andre/Projetos/L07/1000/ajust-bcg-r.fits', memmap=True)[1].data

if os.path.isfile('Projetos/isophotest/L07_coef/1000/iso_table.dat'):
	temp=[[] for i in range(17)]
	iso_table=open('Projetos/isophotest/L07_coef/1000/iso_table.dat','r')
	for item in iso_table.readlines():
		if len(item.split()) == 3:
			extval_ellip=float(item.split()[0])
			extval_pa=float(item.split()[1])
			maxrad=float(item.split()[2])
		else:
			for i in range(17):
				temp[i].append(float(item.split()[i]))

	vec=[]
	for i in range(17):
		vec.append(np.asarray(temp[i]))
	x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err=vec	
	print(len(x0))

xx=[]
yy=[]
for i in range(10):
	isogal= phi.EllipseGeometry(x0=isoimage.shape[1]/2., y0=isoimage.shape[0]/2, sma=20+(i*2), eps=0.8,pa=np.pi/180.)
	aper = php.EllipticalAperture((isogal.x0, isogal.y0), isogal.sma,isogal.sma * (1 - isogal.eps),isogal.pa)
	xx.append(aper)
	test = phi.Ellipse(isoimage, isogal)
	yy.append(test)


fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(isoimage,vmin=0,vmax=1000,origin='lower')
for i in range(len(xx)):
	print(yy[i][0])
	xx[i].plot(color='white')
plt.xlabel('X (pix)')
plt.ylabel('Y (pix)')
plt.tight_layout()
plt.show()

'''
geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                           pa=20.0 * np.pi / 180.0)


'''

