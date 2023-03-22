from math import sin,cos,tan,pi,floor,log10,sqrt,atan2
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

#cluster0,sky,chisq_galfit1,skyvalue2,sigmasky3,ellipgrad4,pagrad5,extval_ellip/ell06,extval_pa-pa07,chirat_18,chirat_29,chirat_310,chisq_s_111,chisq_s_212,chisq_s_313,spopt[0]14,spopt[1]15,spopt[2]16,sepopt[0]17,sepopt[1]18,sepopt[2]19,sepopt[3]20,sepopt[4]21,	vpopt[0]22,vpopt[1]23,fixspopt[0]24,fixspopt[1]25,fixspopt[2]26,fixsepopt[0]27,fixsepopt[1]28,fixsepopt[2]29,fixsepopt[3]30,fixsepopt[4]31,fixvpopt[0]32,fixvpopt[1]32,pixspopt[0]33,pixspopt[1]34,pixspopt[2]35,
#pixsepopt[0]36,pixsepopt[1]37,pixsepopt[2]38,pixsepopt[3]39,pixsepopt[4]40,pixvpopt[0]41,pixvpopt[1]42

def graph(cluster,tipo):
	ajust1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[1].data
	ajust2 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[2].data
	header = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[2].header
	header1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[1].header
	mask = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask.fits', memmap=True)[0].data
	maskb = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask_b.fits', memmap=True)[0].data
	inp1=open('L07_maskcor/'+cluster+'/iso_table.dat','r')

	xc=float(header['1_XC'].split()[0].replace('*',''))
	yc=float(header['1_YC'].split()[0].replace('*',''))
	re = float(header['1_RE'].split()[0].replace('*',''))
	mag = float(header['1_MAG'].split()[0].replace('*',''))
	n = float(header['1_N'].split()[0].replace('*',''))
	sky=float(header['2_SKY'].split()[0].replace('*',''))
	pa0=(float(header['1_PA'].split()[0].replace('*',''))+90.)*np.pi/180.
	ell0=1.-float(header['1_AR'].split()[0].replace('*',''))
	chisq_galfit=float(header['CHI2NU'])
	NMGY=float(header1['NMGY'])
	EXPTIME=float(header1['EXPTIME'])
	
	maxrad=float(inp1.readline().split()[2])
	extval_ellip=float(inp1.readline().split()[0])
	extval_pa=float(inp1.readline().split()[1])

	vec=[[] for i in range(9)]
	for item in inp1.readlines()[0:]:	
		vec[0].append(float(item.split()[0]))
		vec[1].append(float(item.split()[1]))
		vec[2].append(float(item.split()[2]))
		vec[3].append(float(item.split()[3]))
		vec[4].append(float(item.split()[4]))
		vec[5].append(float(item.split()[5]))
		vec[6].append(float(item.split()[6]))
		vec[7].append(float(item.split()[7]))
		vec[8].append(float(item.split()[8]))
				
	x0=np.asarray(vec[0])
	y0=np.asarray(vec[1])
	sma=np.asarray(vec[2])
	pa=np.asarray(vec[3])
	eps=np.asarray(vec[4])
	intens=np.asarray(vec[5])
	ellip_err=np.asarray(vec[6])
	pa_err=np.asarray(vec[7])
	int_err=np.asarray(vec[8])	
	mag_iso = 22.5+(2.5*np.log10(1./NMGY)) -2.5*np.log10(intens) - 2.5*(log10(EXPTIME)) + 2.5*(log10(0.1569166))

	# PLOT DE PARAMETROS GERAIS RAIOX(MAGNITUDE,ELIPTICIDADE,PA)

	plt.figure(figsize=(9, 3))
	plt.subplots_adjust(hspace=0.35, wspace=0.35)
	plt.subplot(1, 3, 1)
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=2*2.5*np.log10(np.exp(1.))*int_err/intens,fmt='o',markersize=4,color='k',ecolor='0.5')
	plt.xlabel(r'$R$ (pix)')
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.,np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.])
	plt.gca().invert_yaxis()
	plt.subplot(1, 3, 2)
	plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=4, color='k',ecolor='0.5')
	plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
	plt.xlabel(r'$R$ (pix)')
	plt.ylabel(r'$e$')
	plt.subplot(1, 3, 3)
	plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=4, color='k',ecolor='0.5')
	plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	plt.xlabel(r'$R$ (pix)')
	plt.ylabel('PA (deg)')
	plt.tight_layout()
	plt.savefig('L07_maskcor/'+cluster+'/iso_info.png')	
	plt.close()
			
	plt.figure()
	plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=4, color='k')
	plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
	plt.xlabel(r'$R$ (pix)')
	plt.ylabel(r'$e$')
	plt.savefig('L07_maskcor/'+cluster+'/elipticidade.png')
			
	plt.figure()
	plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=4, color='k')
	plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	plt.xlabel(r'$R$ (pix)')
	plt.ylabel('PA (deg)')
	plt.savefig('L07_maskcor/'+cluster+'/position_angle.png')
	return

with open('iso_geral_values_maskcor_2.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('iso_geral_values_maskcor_2.dat','r')

chosen=[1001,1090,1140,1171,2009,2075,2167,3051,3152,3296,3583]

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	#print(ll1[0],ik)
	if int(ll1[0]) in chosen:
		print(ll1[0],ll1[17],ll1[18],ll1[19],ll1[20],ll1[21],ll1[29],ll1[38])
	'''
	if os.path.isfile('L07_maskcor/'+ll1[0]+'/iso_table.dat'):
		graph(ll1[0],tipo[0])
	if os.path.isfile('L07_maskcor/'+ll1[0]+'/iso_table.dat'):
		graph(ll1[0],tipo[1])
	if os.path.isfile('L07_maskcor/'+ll1[0]+'/iso_table.dat'):
		graph(ll1[0],tipo[2])
	'''
