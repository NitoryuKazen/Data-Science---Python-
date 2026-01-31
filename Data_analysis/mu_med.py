import os
import os.path
import sys

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# import matplotlib
# matplotlib.use('TkAgg')
from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt
from subprocess import call
from scipy.stats import pearsonr,iqr
from scipy.stats import gaussian_kde as kde
from sklearn.mixture import GaussianMixture
# from kneed import KneeLocator as kl
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy.special as scs
from photutils.aperture import EllipticalAperture, aperture_photometry
from astropy.io import fits
def mue_médio(sample,amostra,redshift,rodada): #RODAR SOMENTE NO LAB
	
	ok=[]
	with open(f'mue_med_dimm_{sample}.dat','r') as inp2:
		for item in inp2.readlines():   
			ok.append(item.split()[0])
	for cluster in amostra:
		if cluster in ok:
			pass
		else:
			z_gal=redshift[amostra==cluster][0].astype(float)
			print(z_gal)
			dimm_cosm=np.power(1+z_gal,4.)
			sersic_model,sersic_header = fits.getdata(f'../{sample}/{cluster}/{rodada}/ajust-sersic-llh-{rodada}.fits',header=True,ext=2)
			sersic_duplo_model,sersic_duplo_header = fits.getdata(f'../{sample}/{cluster}/{rodada}/ajust-sersic-duplo-llh-{rodada}.fits',header=True,ext=2)
			model_header=fits.open(f'../../{sample}/{cluster}/ajust-bcg-r.fits')[2].header
			stamp_header=fits.open(f'../../{sample}/{cluster}/ajust-bcg-r.fits')[1].header

			model_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY']
			
			###
			EXPTIME=float(stamp_header['EXPTIME'])
			magzero=float(model_header['MAGZPT'])+2.5*np.log10(EXPTIME)
			###

			#SERSIC
			xc,yc=float(sersic_header[model_names[0]].split()[0]),float(sersic_header[model_names[1]].split()[0])
			re = float(sersic_header[model_names[3]].split()[0])
			q=float(sersic_header[model_names[6]].split()[0])
			a = re 
			b = re * q
			theta = float(sersic_header[model_names[5]].split()[0])*np.pi/180.

			re_arcsec=0.396*re
			area_re = np.pi*(re_arcsec**2)*q 
			
			aperture = EllipticalAperture((xc, yc), a, b, theta)
			phot_table = aperture_photometry(sersic_model, aperture)
			flux_re = phot_table['aperture_sum'][0]*dimm_cosm

			mag_tot = -2.5*np.log10(flux_re) + magzero
			mu_mean_s = mag_tot + 2.5*np.log10(area_re)

			##SERSIC DUPLO
			
			#COMP 1
			xc,yc=float(sersic_duplo_header[f'{model_names[0]}_1'].split()[0]),float(sersic_duplo_header[f'{model_names[1]}_1'].split()[0])
			re = float(sersic_duplo_header[f'{model_names[3]}_1'].split()[0])
			q=float(sersic_duplo_header[f'{model_names[6]}_1'].split()[0])
			a = re 
			b = re * q
			theta = float(sersic_duplo_header[f'{model_names[5]}_1'].split()[0])*np.pi/180.

			re_arcsec=0.396*re
			area_re = np.pi*(re_arcsec**2)*q 
			
			aperture = EllipticalAperture((xc, yc), a, b, theta)
			phot_table = aperture_photometry(sersic_duplo_model, aperture)
			flux_re = phot_table['aperture_sum'][0]*dimm_cosm

			mag_tot = -2.5*np.log10(flux_re) + magzero
			mu_mean_1 = mag_tot + 2.5*np.log10(area_re)

			#COMP 2
			xc,yc=float(sersic_duplo_header[f'{model_names[0]}_1'].split()[0]),float(sersic_duplo_header[f'{model_names[1]}_1'].split()[0])
			re = float(sersic_duplo_header[f'{model_names[3]}_2'].split()[0])
			q=float(sersic_duplo_header[f'{model_names[6]}_2'].split()[0])
			a = re 
			b = re * q
			theta = float(sersic_duplo_header[f'{model_names[5]}_2'].split()[0])*np.pi/180.

			re_arcsec=0.396*re
			area_re = np.pi*(re_arcsec**2)*q 
			
			aperture = EllipticalAperture((xc, yc), a, b, theta)
			phot_table = aperture_photometry(sersic_duplo_model, aperture)
			flux_re = phot_table['aperture_sum'][0]*dimm_cosm

			mag_tot = -2.5*np.log10(flux_re) + magzero
			mu_mean_2 = mag_tot + 2.5*np.log10(area_re)

			output=open(f'mue_med_dimm_{sample}.dat','a')
			output.write(f'{cluster} {mu_mean_s} {mu_mean_1} {mu_mean_2}\n')
			output.close()
			print(cluster)
	return

def musersic(re,n,mtot):
	bn = 2.*n-1/3.+4./(405.*n)+46./(25515*n**2)+131./(1148175*n**3)-2194697./(30690717750*n**4)
	mue = mtot + 5*np.log10(re) + 2.5*np.log10(2*pi*n*np.exp(bn)*scs.gamma(2*n)/np.power(bn,2*n))
	return mue

def dist_pc_old(re,z):

	h0=70 #kpc(km/s)
	q0=-0.55#omega_m = 0.3 e omega_v=0.7
	c=299792.458 #km/s
	re=re*0.396
	re_rad=np.divide(re*np.pi,648000.)

	hubble_law=c*z/h0
	taylor_q0z=1+np.divide((1-q0)*z,2)
	dist=np.multiply(hubble_law,taylor_q0z)
	re_kpc=np.multiply(dist,re_rad)*1000.
	if re_kpc != 0.:
		x=np.log10(re_kpc)
	else:
		x=0
	return x
def dist_pc(re, z):
	h0 = 70          # km/s/Mpc
	q0 = -0.55       # omega_m = 0.3 e omega_v = 0.7
	c = 299792.458   # km/s

	re = re * 0.396
	re_rad = re * np.pi / 648000.  # converte para radianos

	hubble_law = c * z / h0
	taylor_q0z = 1 + ((1 - q0) * z / 2)
	dist = hubble_law * taylor_q0z

	re_kpc = dist * re_rad * 1000.  # converte para pc

	# evita log10(0)
	x = np.where(re_kpc != 0, np.log10(re_kpc), 0)

	return x
def lnt(cluster):
	
	ajust1 = fits.getdata(f'L07/{cluster}/ajust-sersic-llh.fits',1)
	ajust3 = fits.getdata(f'L07/{cluster}/ajust-sersic-llh.fits',3)

	mask = fits.getdata(f'../L07/{cluster}/bcg_r_mask.fits')
	mask_b = fits.getdata(f'../L07/{cluster}/bcg_r_mask_b.fits')

	##############################################################
	#CALCULO DO RFF
	
	sbk=np.std(ajust1[np.where((mask_b == 0) & (mask==0))])
	nn2=len(ajust1[np.where((mask_b == 1) & (mask==0))])
	xy = np.sum(np.absolute(ajust3[np.where((mask_b == 1) & (mask==0))]))
	xn = np.sum(ajust1[np.where((mask_b == 1) & (mask==0))])

	rff=(xy-0.8*sbk*nn2)/xn
	return rff
def rff_modelo_duplo(cluster):
	
	bcg = fits.getdata(f'L07/{cluster}/observation/ajust-sersic-exp-llh-observation.fits',1)

	resid_se = fits.getdata(f'L07/{cluster}/observation/ajust-sersic-exp-llh-observation.fits',3)
	resid_ss = fits.getdata(f'L07/{cluster}/observation/ajust-sersic-duplo-llh-observation.fits',3)

	mask = fits.getdata(f'../L07/{cluster}/bcg_r_mask.fits')
	mask_b = fits.getdata(f'../L07/{cluster}/bcg_r_mask_b.fits')

	##############################################################
	#CALCULO DO RFF
	
	sbk=np.std(bcg[np.where((mask_b == 0) & (mask==0))])
	nn2=len(bcg[np.where((mask_b == 1) & (mask==0))])
	
	xy_se = np.sum(np.absolute(resid_se[np.where((mask_b == 1) & (mask==0))]))
	xn_se = np.sum(bcg[np.where((mask_b == 1) & (mask==0))])

	xy_ss = np.sum(np.absolute(resid_ss[np.where((mask_b == 1) & (mask==0))]))
	xn_ss = np.sum(bcg[np.where((mask_b == 1) & (mask==0))])

	rff_se=(xy_se-0.8*sbk*nn2)/xn_se
	rff_ss=(xy_ss-0.8*sbk*nn2)/xn_ss

	return rff_se,rff_ss


####################################################################
sample='L07'

header_eta=['cluster','redshift']

data_eta_temp=np.loadtxt(f'{sample}_clean_redshift.dat',dtype=str).T

data_eta=dict(zip(header_eta,data_eta_temp))
##########################
cluster=data_eta['cluster']
redshift=data_eta['redshift']
output=open(f'mue_med_dimm_{sample}.dat','a')
mue_médio(sample,cluster,redshift,'observation_SE_sky')



