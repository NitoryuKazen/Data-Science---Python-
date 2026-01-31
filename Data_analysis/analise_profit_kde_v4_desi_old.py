import os
import os.path
import sys

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
from itertools import combinations
from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
import pandas as pd
import scipy.optimize as scp
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
import matplotlib.patches as mpatches
from subprocess import call
from scipy.stats import pearsonr,iqr,f,chi2,ks_2samp
from scipy.stats import median_abs_deviation as mad
from scipy.stats import gaussian_kde as kde
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy.special as scs
from photutils.aperture import EllipticalAperture, aperture_photometry
import warnings
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from astropy.constants import G
warnings.filterwarnings("ignore",message='invalid value encountered in log10',category=RuntimeWarning)
def make_image(sample,bcg_list,idx_z):
	model_name='SE_sky'
	for galaxia in bcg_list:
		if os.path.isfile(f'{sample}_stats_observation_desi/p_value/{idx_z}/bcg_depot/{galaxia}.png'):
			pass
		else:
			print(galaxia,idx_z)
			bcg=fits.open(f'../{sample}/{galaxia}/observation_{model_name}/ajust-sersic-llh-observation_{model_name}.fits')[1].data

			model_s=fits.open(f'../{sample}/{galaxia}/observation_{model_name}/ajust-sersic-llh-observation_{model_name}.fits')[2].data
			model_ss=fits.open(f'../{sample}/{galaxia}/observation_{model_name}/ajust-sersic-duplo-llh-observation_{model_name}.fits')[2].data

			residuos_s=fits.open(f'../{sample}/{galaxia}/observation_{model_name}/ajust-sersic-llh-observation_{model_name}.fits')[3].data
			residuos_ss=fits.open(f'../{sample}/{galaxia}/observation_{model_name}/ajust-sersic-duplo-llh-observation_{model_name}.fits')[3].data

			fig,axs=plt.subplots(2,3,figsize=(20,10))
			plt.suptitle(f'{galaxia}')
			#plt.subplots_adjust(hspace=0.35, wspace=0.35)
			###
			axs[0,0].imshow(bcg,vmin=0,vmax=1000,cmap='gray')
			axs[0,0].invert_yaxis()

			axs[0,1].imshow(model_s,vmin=0,vmax=1000,cmap='gray')
			axs[0,1].invert_yaxis()

			axs[0,2].imshow(residuos_s,vmin=0,vmax=1000,cmap='gray')
			axs[0,2].invert_yaxis()

			axs[1,0].imshow(bcg,vmin=0,vmax=1000,cmap='gray')
			axs[1,0].invert_yaxis()

			axs[1,1].imshow(model_ss,vmin=0,vmax=1000,cmap='gray')
			axs[1,1].invert_yaxis()

			axs[1,2].imshow(residuos_ss,vmin=0,vmax=1000,cmap='gray')
			axs[1,2].invert_yaxis()

			plt.savefig(f'{sample}_stats_observation_desi/p_value/{idx_z}/bcg_depot/{galaxia}.png')
			plt.close()
	return
def mue_médio(sample,cluster,rodada): #RODAR SOMENTE NO LAB
	#diming flux*
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
	flux_re = phot_table['aperture_sum'][0]

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
	flux_re = phot_table['aperture_sum'][0]

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
	flux_re = phot_table['aperture_sum'][0]

	mag_tot = -2.5*np.log10(flux_re) + magzero
	mu_mean_2 = mag_tot + 2.5*np.log10(area_re)

	return cluster,mu_mean_s,mu_mean_1,mu_mean_2
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
def linfunc(x,a,b):
	return a*x+b
def diff(x,param):
	px=gauss(x,*param[:3]) - lognormal(x,*param[3:6])
	return px
def exp_func(x,a,b):
	return a*np.exp(x*(b))

def log_model(x, a, b):
	return a * np.log(x) + b  
def power_model(x, a, b):
	return a * np.power(-x, b)
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

def gauss(x,mu,sigma,amp):
	return amp*np.exp(-((x-mu)**2)/(2*(sigma)**2))
def lognormal(x,mu,sigma,amp):
	return (amp)*np.exp(-((np.log(x)-mu)**2)/(2*(sigma)**2))

def dlognorm(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
    px=amp1*np.exp(-((x-mu1)**2)/(2*(sigma1)**2))+amp2*np.exp(-(np.log(x)-mu2)**2/(2*sigma2**2))
    return px
def bigauss(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
	return gauss(x,mu1,sigma1,amp1) + gauss(x,mu2,sigma2,amp2)
def diff(x,param):
	px=gauss(x,*param[:3]) - lognormal(x,*param[3:6])
	return px
def bt_ratio(mag_bojo,mag_env):
	fluxo_bojo=	np.power(10,-0.4*mag_bojo)
	fluxo_envelope=np.power(10,-0.4*mag_env)
	ii = np.divide(fluxo_bojo,fluxo_bojo+fluxo_envelope)
	return ii
def bt_ratio_corr(mag_bojo,mag_env,corr_int,corr_ext):
	fluxo_bojo=	np.power(10,-0.4*mag_bojo)*corr_int
	fluxo_envelope=np.power(10,-0.4*mag_env)*corr_ext
	ii = np.divide(fluxo_bojo[fluxo_bojo!=0],(fluxo_bojo+fluxo_envelope)[fluxo_bojo!=0])
	# ii = np.divide(fluxo_bojo,(fluxo_bojo+fluxo_envelope))
	return ii

def zhao_optim(x):
	y=0.57+0.08*x+0.31*np.power(x,2)
	return y
def analise_kde(data_geral,sub_samples,label,save_str,limites):

	kde_sample=kde(data_geral[np.isfinite(data_geral)])
	bw_factor = kde_sample.factor
	kde_sub_samples=[]
	x=np.linspace(limites[0],limites[1],1000)

	fig,axs=plt.subplots(2,int(len(sub_samples)/2),figsize=(15,10),tight_layout=True)
	idx=np.arange(axs.size)
	mat_idx=np.unravel_index(idx,axs.shape)
	cores=['green','blue','red','orange']
	for i,vec in enumerate(sub_samples):
		id_axs=mat_idx[0][i],mat_idx[1][i]
		# kde_sub_samples.append(kde(vec[np.isfinite(vec)],bw_method=bw_factor))
		kde_subsample=kde(vec[np.isfinite(vec)],bw_method=bw_factor)
		axs[id_axs].plot(x,kde_subsample(x),color=cores[i],label=f'{label[1][i]}')
		axs[id_axs].set_xlabel(label[0])
		axs[id_axs].legend()
	plt.show()
	# plt.savefig(f'observation/{save_str}.png')
	plt.close()
	idx_vec=[0,1,2,3]
	fig,axs=plt.subplots(1,2,figsize=(15,5),tight_layout=True)
	for i,vec in enumerate(sub_samples[::2]):
		j=idx_vec[::2][i]
		# kde_sub_samples.append(kde(vec[np.isfinite(vec)],bw_method=bw_factor))
		kde_subsample=kde(vec[np.isfinite(vec)],bw_method=bw_factor)
		axs[i].plot(x,kde_subsample(x),color=cores[j],label=f'{label[1][j]}')
		axs[i].set_xlabel(label[0])
		axs[i].legend()

	plt.show()
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(15,5),tight_layout=True)
	for j,vec in enumerate(sub_samples[1::2]):
		i=idx_vec[1::2][j]
		# kde_sub_samples.append(kde(vec[np.isfinite(vec)],bw_method=bw_factor))
		kde_subsample=kde(vec[np.isfinite(vec)],bw_method=bw_factor)
		axs[j].plot(x,kde_subsample(x),color=cores[i],label=f'{label[1][i]}')
		axs[j].set_xlabel(label[0])
		axs[j].legend()
	plt.show()
	plt.close()

	# plt.axvline(x=float(vec_stats_s[0][0]),ymin=0,ymax=1,ls='--',color='green',label='Média =%.3f +/- %.3f'%(float(vec_stats_s[0][0]),float(vec_stats_s[0][3])))
	# plt.axvline(x=float(vec_stats_se[0][0]),ymin=0,ymax=1,ls='--',color='red',label='Média =%.3f +/- %.3f'%(float(vec_stats_se[0][0]),float(vec_stats_se[0][3])))
	return
def gmm_bootstrap(sample,delta_bic,savedir,n,n_bins):
	delta_bic_vec=delta_bic.reshape(-1,1)

	ncl=[1,2,3,4,5,6,7,8,9,10]
	path=f'{sample}_stats_observation_desi/redshift_gmm/{n_bins}_bins/'
	output=open(f'{path}{sample}_stats_{savedir}.dat','a')
	ajust_vec=[]
	labels_vec=[]
	for i in range(10):
		gmm_mix=GaussianMixture(n_components=ncl[i])
		ajuste=gmm_mix.fit(delta_bic_vec)
		labels = gmm_mix.predict(delta_bic_vec)
		ajust_vec.append(ajuste)
		labels_vec.append(labels)

	bics = [m.bic(delta_bic_vec) for m in ajust_vec]
	aics = [m.aic(delta_bic_vec) for m in ajust_vec]

	elbow_bic=kl(ncl,bics,curve='convex',direction='decreasing')
	elbow_aic=kl(ncl,aics,curve='convex',direction='decreasing')

	plt.figure()
	plt.plot(ncl, bics,c='blue', label='BIC')
	plt.plot(ncl, aics,c='orange', label='AIC')
	plt.axvline(elbow_bic.elbow,label='Elbow BIC')
	plt.axvline(elbow_aic.elbow,c='orange',linestyle='--',label='Elbow AIC')
	plt.xlabel('Número de famílias')
	plt.ylabel('Valor do critério')
	plt.legend()
	plt.savefig(f'{path}{savedir}/elbow_score_{n}.png')
	plt.close()

	for q in range(10):
		comps=ncl[q]
		comp_output=open(f'{path}{savedir}/n_{comps}_comp_stats.dat','a')
		bic_comp=bics[q]
		vec_ll=[]
		vec_min_ll=[]
		for w in range(comps):
			mask = (labels_vec[comps-1] == w)
			if np.any(mask):
				vec_ll.append(mask)
				vec_min_ll.append(np.min(delta_bic_vec[mask]))
			else:
				vec_ll.append(mask)
				vec_min_ll.append(np.nan)
		
		vec_ll=np.asarray(vec_ll)
		vec_min_ll=np.asarray(vec_min_ll)

		comp_idx_mask=np.isfinite(vec_min_ll)

		vec_ll=vec_ll[comp_idx_mask]
		vec_min_ll=vec_min_ll[comp_idx_mask]

		vec_ll=vec_ll[np.argsort(-vec_min_ll)]
		vec_min_ll=vec_min_ll[np.argsort(-vec_min_ll)]
		try:
			comp_output.write(f'{n} {comps} {bic_comp} {' '.join(str(vec_min_ll[g]) for g in range(comps))} \n')
			comp_output.close()
		except:
			comp_output.write(f'{n} nan nan nan \n')
			comp_output.close()
	##############################

	ll=[]
	min_ll=[]
	for k in range(elbow_bic.elbow):
		mask = (labels_vec[elbow_bic.elbow-1] == k)
		if np.any(mask):
			ll.append(mask)
			min_ll.append(np.min(delta_bic_vec[mask]))
		else:
			ll.append(mask)
			min_ll.append(np.nan)

	ll=np.asarray(ll)
	min_ll=np.asarray(min_ll)

	idx_mask=np.isfinite(min_ll)

	ll=ll[idx_mask]
	min_ll=min_ll[idx_mask]

	ll=ll[np.argsort(-min_ll)]
	min_ll=min_ll[np.argsort(-min_ll)]
	try:
		output.write(f'{n} {elbow_bic.elbow} {' '.join(str(min_ll[g]) for g in range(elbow_bic.elbow))} \n')
		output.close()
	except:
		output.write(f'{n} nan nan \n')
		output.close()
	########################################
	bins_delta=np.arange(-7500,1000,200)
	fig=plt.figure()
	ax=fig.add_subplot()
	cores=['green','blue','orange','red','purple','pink','gray']
	lines=['--','-.',':','--','-.']
	for h in range(elbow_bic.elbow):
		try:
			ax.hist(delta_bic[ll[h]], bins=bins_delta, color=cores[h], alpha=0.5, label=f'Grupo {h+1}')
			if h+1 != elbow_bic.elbow:
				ax.axvline(min_ll[h],c='black',linestyle=lines[h],label=f'{min_ll[h]}')
		except:
			ax.hist([], bins=bins_delta, color=cores[h], alpha=0.5, label=f'Grupo {h+1}')

	ax.set_xlabel(r'$\Delta BIC$')
	ax.set_xlim(-10000,5000)
	plt.legend()
	plt.savefig(f'{path}{savedir}/hist_try_{n}.png')
	plt.close(fig)

	return
def bic_clean(bic,max_llh,llh_cl):
	cl_llh=-llh_cl
	x_data=bic+2*max_llh
	cl_bic=x_data-2*cl_llh
	return cl_bic
def f_test(chi2_s,chi2_ss,ndof_s,ndof_ss):
	chi2nu_s = chi2_s / ndof_s
	chi2nu_ss = chi2_ss / ndof_ss
	F = chi2nu_s / chi2nu_ss
	dfn = ndof_s
	dfd = ndof_ss
	p_value = f.sf(F, dfn, dfd)
	return p_value
def two_line_model(params, x, y):
	slope1, intercept1, slope2, intercept2, x_div = params

	y_pred = np.zeros_like(x)
	mask1 = x <= x_div
	mask2 = x > x_div

	y_pred[mask1] = slope1 * x[mask1] + intercept1
	y_pred[mask2] = slope2 * x[mask2] + intercept2

	return y_pred
def chi_square(params, x, y, y_err=None):
	if y_err is None:
		y_err = np.ones_like(y)  # Assume equal errors if not provided

	y_pred = two_line_model(params, x, y)
	chi2_val = np.sum(((y - y_pred) / y_err) ** 2)
	return chi2_val
def find_optimal_division(x, y, y_err=None, initial_guess=None):
    if initial_guess is None:
        # x_min, x_max = -0.283,2.716#np.min(x), np.max(x)
        # x_mid = 0.7#0.5*(x_min+x_max)
        x_min, x_max =-0.3,2.5# np.min(x), np.max(x)
        x_mid = 0.65#0.5*(x_min+x_max)
        
        # Fit initial lines to left and right halves
        mask_left = x <= x_mid
        mask_right = x > x_mid
        
        if np.sum(mask_left) > 1:
            coeffs_left = np.polyfit(x[mask_left], y[mask_left], 1)
        else:
            coeffs_left = [-3, 0.0]
            
        if np.sum(mask_right) > 1:
            coeffs_right = np.polyfit(x[mask_right], y[mask_right], 1)
        else:
            coeffs_right = [-3, 0.0]
        
        initial_guess = [coeffs_left[0], coeffs_left[1], 
                        coeffs_right[0], coeffs_right[1], x_mid]
    
    # Constraints: x_div should be within data range
    bounds = [(None, None), (None, None), (None, None), (None, None), 
              (np.min(x) + 0.1, np.max(x) - 0.1)]
    
    # Minimize chi-square
    result = minimize(chi_square, initial_guess, args=(x, y, y_err),
                     method='Nelder-Mead', bounds=bounds)
    
    return result
def bootstrap_uncertainty(x, y, params, y_err=None, n_bootstrap=100):
    """
    Estimate uncertainties using bootstrap resampling
    """
    bootstrap_params = []
    n_points = len(x)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_points, n_points, replace=True)
        x_bs = x[indices]
        y_bs = y[indices]
        
        if y_err is not None:
            y_err_bs = y_err[indices]
        else:
            y_err_bs = None
        
        # Fit to bootstrap sample
        try:
            result = find_optimal_division(x_bs, y_bs, y_err_bs)
            if result.success:
                bootstrap_params.append(result.x)
        except:
            continue
    
    bootstrap_params = np.array(bootstrap_params)
    
    if len(bootstrap_params) > 0:
        uncertainties = np.std(bootstrap_params, axis=0)
        return uncertainties
    else:
        return np.zeros_like(params)
def ks_calc(vecs):
	n = len(vecs)
	results_p = [[] for i in range(n)]
	results_d = [[] for i in range(n)]

	for i in range(n):
		for j in range(n):
			D, p = ks_2samp(vecs[i], vecs[j])
			results_p[i].append(p)
			results_d[i].append(D)
	return results_p,results_d
def svc_calc_trio(vecs_x,vecs_y,name,outfile):
	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	coefs  = res.named_steps['svc'].coef_.ravel()
	inter  = res.named_steps['svc'].intercept_.ravel()

	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	name=name1,name2,name3
	save_vec = np.concatenate([[name1],[name2],[name3],[acc_check]])

	with open(outfile, "ab") as f:
	    np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_calc_dupla(vecs_x,vecs_y,name,outfile):

	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	coefs  = res.named_steps['svc'].coef_.ravel()
	inter  = res.named_steps['svc'].intercept_.ravel()

	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	name=name1,name2
	save_vec = np.concatenate([[name1],[name2],[acc_check]])

	with open(outfile, "ab") as f:
	    np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_calc(vecs_x,vecs_y):

	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	scaler = res.named_steps['standardscaler']
	svc    = res.named_steps['svc']

	coefs_pad = svc.coef_[0]     # coeficientes no espaço padronizado
	inter_pad = svc.intercept_[0]

	# Desfaz a padronização
	coefs = coefs_pad / scaler.scale_
	inter = inter_pad - np.sum(coefs_pad * scaler.mean_ / scaler.scale_)
	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	# name=name1,name2
	# save_vec = np.concatenate([[name1],[name2],[acc_check]])

	# with open(outfile, "ab") as f:
	#     np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_line_plot(x,a,b,c):
	y_plot = -(a*x+b)/c
	return y_plot

def make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place):
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')

	writer_2c = PillowWriter(fps=2)

	with writer_2c.saving(fig, save_place[0], dpi=100):
		for angle in np.arange(0,360,60):
			for angle2 in np.arange(0,180,30):
				ax.clear()
				ax.scatter(vec_data[0][lim_cd_small], vec_data[1][lim_cd_small],vec_data[2][lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
				ax.scatter(vec_data[0][lim_cd_big], vec_data[1][lim_cd_big], vec_data[2][lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='True cD')
				ax.set_zlim(zlim)
				if xlim != None:
					ax.set_xlim(xlim)
				if ylim != None:
					ax.set_xlim(ylim)
				ax.set_xlabel(vec_label[0])
				ax.set_ylabel(vec_label[1])
				ax.set_zlabel(vec_label[2])
				ax.legend()
				ax.view_init(elev=angle, azim=angle2)
				writer_2c.grab_frame()
		plt.close()

	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')

	writer_sample = PillowWriter(fps=2)

	with writer_sample.saving(fig, save_place[1], dpi=100):

		for angle in np.arange(0,360,60):
			for angle2 in np.arange(0,180,30):
				ax.clear()
				ax.scatter(vec_data[0][elip_lim], vec_data[1][elip_lim],vec_data[2][elip_lim],alpha=0.4, c='green', edgecolor='black', label='E')
				ax.scatter(vec_data[0][lim_cd_small], vec_data[1][lim_cd_small],vec_data[2][lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
				ax.scatter(vec_data[0][lim_cd_big], vec_data[1][lim_cd_big], vec_data[2][lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='True cD')
				ax.set_zlim(zlim)
				ax.set_xlabel(vec_label[0])
				ax.set_ylabel(vec_label[1])
				ax.set_zlabel(vec_label[2])
				ax.legend()
				ax.view_init(elev=angle, azim=angle2)
				writer_sample.grab_frame()
		plt.close()
	return
###################################################################################
###################################################################################
###################################################################################
#BLOCO DE ABERTURA DOS DADOS
sample=str(sys.argv[1])
#HEADERS
header_data_z=['cluster','redshift','morfologia']
header_eta=['cluster','assimetria']
header_data=np.loadtxt(f'profit_observation.header',dtype=str)
header_mue=['cluster','mue_med_s','mue_med_1','mue_med_2']
header_psf=['cluster','npix','psf_fwhm']
header_chi2=['cluster','chi2_s','chi2_ss']
header_casjobs=['cluster','magabs','logmass','age','metal','conc']
header_halpha=['cluster','halpha','halpha_err']
header_veldisp=['cluster','veldisp','veldisp_err']
header_photutils=np.loadtxt(f'graph_stats_WHL_sky.header',dtype=str)
header_erro_photutils=['cluster','grad_e_err','grad_pa_err','slope_a3_err','slope_a4_err','slope_b3_err','slope_b4_err','slope_disk_err','slope_gr_fix_log_err','slope_gr_fix_err','slope_gr_free_log_err','slope_gr_free_err']
if sample == 'WHL':
	header_200=['cluster','r200','richness','n200']
elif sample=='L07':
	header_200=['cluster','r200','vel']
header_corr=['cluster','corr_1','corr_2']
header_class=['cluster','bic_class']
#DATA FILES
data_chi2_temp=np.loadtxt(f'{sample}_files/{sample}_profit_desi_obs_chi2_desi.dat',dtype=str).T
data_psf_temp=np.loadtxt(f'{sample}_files/{sample}_psf_data_desi.dat',dtype=str).T
data_z_temp=np.loadtxt(f'{sample}_files/{sample}_clean_redshift.dat',dtype=str).T
data_eta_temp=np.loadtxt(f'{sample}_files/{sample}_ass_desi.dat').T
# data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky.dat',dtype=str).T
data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_desi_obs.dat',dtype=str).T
data_mue_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_{sample}_desi.dat',dtype=str).T
data_mue_comp_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_comp_{sample}_desi.dat',dtype=str).T
data_casjobs_temp=np.loadtxt(f'{sample}_files/casjobs_data_clean_{sample}.dat',dtype=str).T
data_halpha_temp=np.loadtxt(f'{sample}_files/casjobs_halpha_{sample}.dat',dtype=str).T
data_veldisp_temp=np.loadtxt(f'{sample}_files/casjobs_veldisp_{sample}.dat',dtype=str).T
# data_photutils_temp=np.loadtxt(f'{sample}_files/graph_stats_{sample}_sky_v2.dat',dtype=str).T
data_photutils_temp=np.loadtxt(f'{sample}_files/graph_stats_{sample}_sky_desi.dat',dtype=str).T
data_simul_s_temp=np.loadtxt(f'{sample}_files/{sample}_profit_simulation_SE_sky.dat',dtype=str).T
data_simul_ss_temp=np.loadtxt(f'{sample}_files/{sample}_profit_simulation_s_duplo_SE_sky.dat',dtype=str).T
data_erro_photutils_temp=np.loadtxt(f'{sample}_files/graph_errors_{sample}_sky.dat',dtype=str).T
data_200_temp=np.loadtxt(f'{sample}_files/{sample}_r200.dat',dtype=str).T
data_corr_temp=np.loadtxt(f'{sample}_files/{sample}_corr_bt_desi.dat',dtype=str).T
data_class_temp=np.loadtxt(f'{sample}_files/{sample}_bic_classification.dat',dtype=str).T
###############
data_z=dict(zip(header_data_z,data_z_temp))
data_eta=dict(zip(header_eta,data_eta_temp))
data_obs=dict(zip(header_data,data_obs_temp))
data_psf=dict(zip(header_psf,data_psf_temp))
data_chi2=dict(zip(header_chi2,data_chi2_temp))
data_casjobs=dict(zip(header_casjobs,data_casjobs_temp))
data_halpha=dict(zip(header_halpha,data_halpha_temp))
data_veldisp=dict(zip(header_veldisp,data_veldisp_temp))
data_photutils=dict(zip(header_photutils,data_photutils_temp))
data_simul_s=dict(zip(header_data,data_simul_s_temp))
data_simul_ss=dict(zip(header_data,data_simul_ss_temp))
data_mue=dict(zip(header_mue,data_mue_temp))
data_mue_comp=dict(zip(header_mue,data_mue_comp_temp))
data_erro_photutils=dict(zip(header_erro_photutils,data_erro_photutils_temp))
data_200=dict(zip(header_200,data_200_temp))
data_corr=dict(zip(header_corr,data_corr_temp))
data_class=dict(zip(header_class,data_class_temp))
rff_ss=np.loadtxt(f'{sample}_files/{sample}_rff_duplo_desi.dat',dtype=float,usecols=[1]).T

mue_med_s,mue_med_1,mue_med_2=data_mue['mue_med_s'].astype(float),data_mue['mue_med_1'].astype(float),data_mue['mue_med_2'].astype(float)
mue_med_s,mue_med_comp_1,mue_med_comp_2=data_mue_comp['mue_med_s'].astype(float),data_mue_comp['mue_med_1'].astype(float),data_mue_comp['mue_med_2'].astype(float)
re1,re2=data_obs['RE_1'].astype(float),data_obs['RE_2'].astype(float)
n1,n2=data_obs['NSER_1'].astype(float),data_obs['NSER_2'].astype(float)

lim_finite=np.isfinite(mue_med_s) & np.isfinite(mue_med_comp_1) & np.isfinite(mue_med_comp_2)

n1_lim=(n1==0.5) | (n1>9.9)
n2_lim=(n2==0.5) | (n2>14.9)

re1_lim=(re1<=1.05)
re2_lim=(re2<=1.05)

cut_lim=~(n1_lim | n2_lim | re1_lim | re2_lim)
if sample == 'WHL':
	ra,dec=np.loadtxt(f'data_indiv_WHL_clean.dat',dtype=float,usecols=[1,2])[lim_finite & cut_lim].T
elif sample== 'L07':
	ra,dec=np.loadtxt(f'L07_files/data_indiv_clean_L07.dat',dtype=float,usecols=[5,6])[lim_finite & cut_lim].T

##############
cluster=data_obs['cluster'][lim_finite & cut_lim]

rff_s=data_obs['rff'].astype(float)[lim_finite & cut_lim]
rff_ss=rff_ss[lim_finite & cut_lim]
ass=data_eta['assimetria'].astype(float)[lim_finite & cut_lim]
eta=rff_s-ass
rff_ratio=np.divide(rff_ss,rff_s)

#####
bic_sersic_obs=data_obs['BIC_s'].astype(float)[lim_finite & cut_lim]
bic_sersic_duplo_obs=data_obs['BIC_ss'].astype(float)[lim_finite & cut_lim]
delta_bic_obs=bic_sersic_duplo_obs - bic_sersic_obs
####

llh_sersic_obs_clean=data_obs['CL_LLH_s'].astype(float)[lim_finite & cut_lim]
llh_sersic_duplo_obs_clean=data_obs['CL_LLH_ss'].astype(float)[lim_finite & cut_lim]
llh_sersic_obs=data_obs['MAX_LLH_s'].astype(float)[lim_finite & cut_lim]
llh_sersic_duplo_obs=data_obs['MAX_LLH_ss'].astype(float)[lim_finite & cut_lim]

llh_sersic_sim=data_simul_s['MAX_LLH_s'].astype(float)
llh_sersic_duplo_sim=data_simul_s['MAX_LLH_ss'].astype(float)

bic_sersic_obs_clean=bic_clean(bic_sersic_obs,llh_sersic_obs,llh_sersic_obs_clean)
bic_sersic_duplo_obs_clean=bic_clean(bic_sersic_duplo_obs,llh_sersic_duplo_obs,llh_sersic_duplo_obs_clean)
delta_bic_obs_clean=bic_sersic_duplo_obs_clean - bic_sersic_obs_clean

chi2_s=data_chi2['chi2_s'].astype(float)[lim_finite & cut_lim]
chi2_ss=data_chi2['chi2_ss'].astype(float)[lim_finite & cut_lim]

chi2_ratio=chi2_s/chi2_ss

psf_factor=np.pi*np.power(data_psf['psf_fwhm'].astype(float),2.)[lim_finite & cut_lim]

region=data_psf['npix'].astype(float)[lim_finite & cut_lim]

n_res=region/psf_factor

k1,k2=9,15

n_dof_s=n_res-k1-1
n_dof_ss=n_res-k2-1

p_value=f_test(chi2_s,chi2_ss,n_dof_s,n_dof_ss)

delta_llh =data_simul_s['MAX_LLH_ss'].astype(float) - data_simul_s['MAX_LLH_s'].astype(float)

bic_sersic_sim=data_simul_s['BIC_s'].astype(float)[lim_finite & cut_lim]
bic_sersic_sim_duplo=data_simul_s['BIC_ss'].astype(float)[lim_finite & cut_lim]
delta_bic_sim=bic_sersic_sim_duplo - bic_sersic_sim

bic_sersic_sim_ss=data_simul_ss['BIC_s'].astype(float)
bic_sersic_sim_duplo_ss=data_simul_ss['BIC_ss'].astype(float)
delta_bic_sim_ss=bic_sersic_sim_duplo_ss - bic_sersic_sim_ss

redshift=data_z['redshift'].astype(float)[lim_finite & cut_lim]

mag1,mag2=data_obs['MAG_1'].astype(float)[lim_finite & cut_lim],data_obs['MAG_2'].astype(float)[lim_finite & cut_lim]
re1,re2=data_obs['RE_1'].astype(float)[lim_finite & cut_lim],data_obs['RE_2'].astype(float)[lim_finite & cut_lim]
n1,n2=data_obs['NSER_1'].astype(float)[lim_finite & cut_lim],data_obs['NSER_2'].astype(float)[lim_finite & cut_lim]

e1,e2,e_s=data_obs['AXRAT_1'].astype(float)[lim_finite & cut_lim],data_obs['AXRAT_2'].astype(float)[lim_finite & cut_lim],data_obs['AXRAT'].astype(float)[lim_finite & cut_lim]
box1,box2,box_s=data_obs['BOX_1'].astype(float)[lim_finite & cut_lim],data_obs['BOX_2'].astype(float)[lim_finite & cut_lim],data_obs['BOX'].astype(float)[lim_finite & cut_lim]

corr_1,corr_2=data_corr['corr_1'].astype(float)[lim_finite & cut_lim],data_corr['corr_2'].astype(float)[lim_finite & cut_lim]

mag_s,re_s,n_s=data_obs['MAG'].astype(float)[lim_finite & cut_lim],data_obs['RE'].astype(float)[lim_finite & cut_lim],data_obs['NSER'].astype(float)[lim_finite & cut_lim]
sky=data_obs['SKY_s'].astype(float)[lim_finite & cut_lim]

mue_med_s=mue_med_s[lim_finite & cut_lim]
mue_med_1=mue_med_1[lim_finite & cut_lim]
mue_med_2=mue_med_2[lim_finite & cut_lim]

mue_med_comp_1=mue_med_comp_1[lim_finite & cut_lim]
mue_med_comp_2=mue_med_comp_2[lim_finite & cut_lim]

mag_env=np.where(re1>re2,mag1,mag2)
mag_bojo=np.where(re1>re2,mag2,mag1)

re_env=np.where(re1>re2,re1,re2)
re_bojo=np.where(re1>re2,re2,re1)

n_env=np.where(re1>re2,n1,n2)
n_bojo=np.where(re1>re2,n2,n1)

e_env=np.where(re1>re2,e1,e2)
e_bojo=np.where(re1>re2,e2,e1)

box_env=np.where(re1>re2,box1,box2)
box_bojo=np.where(re1>re2,box2,box1)

bt_vec_12=bt_ratio(mag1,mag2)
re_ratio_12=np.divide(re1,re2)
n_ratio_12=np.divide(n1,n2)
axrat_ratio_12=np.divide(e1,e2)

bt_vec_corr=bt_ratio_corr(mag1,mag2,corr_1,corr_2)
bt_vec_ss=bt_ratio(mag_bojo,mag_env)
re_ratio_ss=np.divide(re_bojo,re_env)
n_ratio_ss=np.divide(n_bojo,n_env)

re_env_kpc=dist_pc(re_env,redshift)
re_bojo_kpc=dist_pc(re_bojo,redshift)
re_s_kpc=dist_pc(re_s,redshift)

re_1_kpc=dist_pc(re1,redshift)
re_2_kpc=dist_pc(re2,redshift)

mue_1=musersic(re1,n1,mag1)
mue_2=musersic(re2,n2,mag2)

mue_env=musersic(re_env,n_env,mag_env)
mue_bojo=musersic(re_bojo,n_bojo,mag_bojo)
mue_s=musersic(re_s,n_s,mag_s)

mue_med_env=np.where(re1>re2,mue_med_1,mue_med_2)
mue_med_bojo=np.where(re1>re2,mue_med_2,mue_med_1)

magabs=data_casjobs['magabs'].astype(float)[lim_finite & cut_lim]
h_line=data_halpha['halpha'].astype(float)[lim_finite & cut_lim]
vel_disp=data_veldisp['veldisp'].astype(float)[lim_finite & cut_lim]
slope_gr=data_photutils['slope_gr_fix_log'].astype(float)[lim_finite & cut_lim]

lim_casjobs=magabs!=0
lim_halpha=h_line!=0
lim_veldisp=vel_disp!=0
lim_photutils=np.isfinite(slope_gr)

if sample=='L07':
	tipo_morf=data_z['morfologia'][lim_finite & cut_lim]
	e_cut=tipo_morf=='E'
	cd_cut=tipo_morf=='cD'
	ecd_cut=tipo_morf=='E/cD'
	cde_cut=tipo_morf=='cD/E'

	e_cut_casjobs=e_cut[lim_casjobs]
	cd_cut_casjobs=cd_cut[lim_casjobs]
	ecd_cut_casjobs=ecd_cut[lim_casjobs]
	cde_cut_casjobs=cde_cut[lim_casjobs]

	e_cut_photutils=e_cut[lim_photutils]
	cd_cut_photutils=cd_cut[lim_photutils]
	ecd_cut_photutils=ecd_cut[lim_photutils]
	cde_cut_photutils=cde_cut[lim_photutils]

	e_cut_halpha=e_cut[lim_halpha]
	cd_cut_halpha=cd_cut[lim_halpha]
	ecd_cut_halpha=ecd_cut[lim_halpha]
	cde_cut_halpha=cde_cut[lim_halpha]

	e_cut_veldisp=e_cut[lim_veldisp]
	cd_cut_veldisp=cd_cut[lim_veldisp]
	ecd_cut_veldisp=ecd_cut[lim_veldisp]
	cde_cut_veldisp=cde_cut[lim_veldisp]

magabs_temp=data_casjobs['magabs'].astype(float)[lim_finite & cut_lim]
starmass_temp=data_casjobs['logmass'].astype(float)[lim_finite & cut_lim]
age_temp=data_casjobs['age'].astype(float)[lim_finite & cut_lim]
conc_temp=data_casjobs['conc'].astype(float)[lim_finite & cut_lim]

magabs=data_casjobs['magabs'].astype(float)[lim_finite & cut_lim][lim_casjobs]
starmass=data_casjobs['logmass'].astype(float)[lim_finite & cut_lim][lim_casjobs]
age=data_casjobs['age'].astype(float)[lim_finite & cut_lim][lim_casjobs]
conc=data_casjobs['conc'].astype(float)[lim_finite & cut_lim][lim_casjobs]
h_line=data_halpha['halpha'].astype(float)[lim_finite & cut_lim][lim_halpha]
vel_disp=data_veldisp['veldisp'].astype(float)[lim_finite & cut_lim][lim_veldisp]

grad_e=data_photutils['ellipgrad'].astype(float)[lim_finite & cut_lim][lim_photutils]
grad_pa=np.abs(data_photutils['pagrad'].astype(float)[lim_finite & cut_lim][lim_photutils])
chi2_s=data_photutils['chi_s'].astype(float)[lim_finite & cut_lim][lim_photutils]
chi2_ss=data_photutils['chi_ss'].astype(float)[lim_finite & cut_lim][lim_photutils]

ie_s=data_photutils['ie_s'].astype(float)[lim_finite & cut_lim][lim_photutils]
n_1d_s=data_photutils['n_s'].astype(float)[lim_finite & cut_lim][lim_photutils]

ie_1d_1=data_photutils['ie_1'].astype(float)[lim_finite & cut_lim][lim_photutils]
re_1d_1=data_photutils['re_1'].astype(float)[lim_finite & cut_lim][lim_photutils]
n_1d_1=data_photutils['n_1'].astype(float)[lim_finite & cut_lim][lim_photutils]
ie_1d_2=data_photutils['ie_2'].astype(float)[lim_finite & cut_lim][lim_photutils]
re_1d_2=data_photutils['re_2'].astype(float)[lim_finite & cut_lim][lim_photutils]
n_1d_2=data_photutils['n_2'].astype(float)[lim_finite & cut_lim][lim_photutils]

med_a3=data_photutils['med_a3'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_a3=data_photutils['slope_slow_a3'].astype(float)[lim_finite & cut_lim][lim_photutils]

med_a4=data_photutils['med_a4'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_a4=data_photutils['slope_slow_a4'].astype(float)[lim_finite & cut_lim][lim_photutils]

med_b3=data_photutils['med_b3'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_b3=data_photutils['slope_slow_b3'].astype(float)[lim_finite & cut_lim][lim_photutils]

med_b4=data_photutils['med_b4'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_b4=data_photutils['slope_slow_b4'].astype(float)[lim_finite & cut_lim][lim_photutils]

med_disk=data_photutils['med_diskness'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_disk=data_photutils['slope_slow_diskness'].astype(float)[lim_finite & cut_lim][lim_photutils]

slope_gr=data_photutils['slope_gr_fix_log'].astype(float)[lim_finite & cut_lim][lim_photutils]

slope_a3_err=data_erro_photutils['slope_a3_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_a4_err=data_erro_photutils['slope_a4_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_b3_err=data_erro_photutils['slope_b3_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_b4_err=data_erro_photutils['slope_b4_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_disk_err=data_erro_photutils['slope_disk_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
slope_gr_err=data_erro_photutils['slope_gr_fix_log_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
grad_e_err=data_erro_photutils['grad_e_err'].astype(float)[lim_finite & cut_lim][lim_photutils]
grad_pa_err=data_erro_photutils['grad_pa_err'].astype(float)[lim_finite & cut_lim][lim_photutils]

slope_a3_err=np.ones_like(grad_e_err)
slope_a4_err=np.ones_like(grad_e_err)
slope_b3_err=np.ones_like(grad_e_err)
slope_b4_err=np.ones_like(grad_e_err)
slope_disk_err=np.ones_like(grad_e_err)
slope_gr_err=np.ones_like(grad_e_err)
grad_e_err=np.ones_like(grad_e_err)
grad_pa_err=np.ones_like(grad_pa_err)

if sample=='WHL':
	r200=data_200['r200'].astype(float)[lim_finite & cut_lim]
	cl_rich=data_200['richness'].astype(float)[lim_finite & cut_lim]

	m200=-1.49+1.17*np.log10(cl_rich)
elif sample == 'L07':
	r200=data_200['r200'].astype(float)[lim_finite & cut_lim]
	vel=data_200['vel'].astype(float)[lim_finite & cut_lim]
	m200=np.log10((3*(vel**3)*r200)/G.value)

cmap=plt.colormaps['hot']
cmap_r=plt.colormaps['hot_r']

lim_cd_big=data_class['bic_class'][lim_finite & cut_lim]=='cD'
lim_cd_small=data_class['bic_class'][lim_finite & cut_lim]=='E(EL)'
elip_lim=data_class['bic_class'][lim_finite & cut_lim]=='E'

cd_lim=(lim_cd_big) | (lim_cd_small)

cd_lim_casjobs=cd_lim[lim_casjobs]
cd_lim_halpha=cd_lim[lim_halpha]
cd_lim_veldisp=cd_lim[lim_veldisp]
cd_lim_photutils=cd_lim[lim_photutils]

elip_lim_casjobs=elip_lim[lim_casjobs]
elip_lim_halpha=elip_lim[lim_halpha]
elip_lim_veldisp=elip_lim[lim_veldisp]
elip_lim_photutils=elip_lim[lim_photutils]

cd_lim_casjobs_small=lim_cd_small[lim_casjobs]
cd_lim_casjobs_big=lim_cd_big[lim_casjobs]

cd_lim_halpha_small=lim_cd_small[lim_halpha]
cd_lim_halpha_big=lim_cd_big[lim_halpha]

cd_lim_veldisp_small=lim_cd_small[lim_veldisp]
cd_lim_veldisp_big=lim_cd_big[lim_veldisp]

cd_lim_photutils_small=lim_cd_small[lim_photutils]
cd_lim_photutils_big=lim_cd_big[lim_photutils]

###################################################################################
###################################################################################
###################################################################################
#INVESTIGAÇÃO DO KORMENDY

os.makedirs(f'{sample}_stats_observation_desi/p_value',exist_ok=True)

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_s_kpc,mue_med_s,marker='d',edgecolor='black',label='Sérsic',color='blue')
x1s,x2s=axs.set_xlim()
y1s,y2s=axs.set_ylim()
axs.yaxis.set_inverted(True)
plt.close()

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_1_kpc,mue_med_1,marker='s',edgecolor='black',label='comp1',color='green')
axs.scatter(re_2_kpc,mue_med_2,marker='o',edgecolor='black',label='comp2',color='red')
axs.scatter(re_s_kpc,mue_med_s,marker='d',edgecolor='black',label='Sérsic',color='blue')
x1ss,x2ss=axs.set_xlim()
y1ss,y2ss=axs.set_ylim()
x1ss,x2ss=(-0.28322921653227484,2.716867381942037)
y1ss,y2ss=(16.824997655983005,26.727550943781317)
axs.yaxis.set_inverted(True)
plt.close()


xlabel=r'$\log_{10} R_e (Kpc)$'
ylabel=r'$<\mu_e>$'

label_cd='cD'
label_elip='E'
label_elip_el='E(EL)'
linha_cd_label='Linha cD'
linha_elip_label='Linha E'
linha_elip_el_label='Linha E(EL)'


label_bojo='Comp 1 cD.'
label_env='Comp 2 cD.'
label_bojo_el='Comp 1 E(EL)'
label_env_el='Comp 2 E(EL)'

linha_interna_label='Linha comp 1 cD'
linha_externa_label='Linha comp 2 cD'

linha_interna_el_label='Linha comp 1 E(EL)'
linha_externa_el_label='Linha comp 2 E(EL)'
####

#GERAL
linspace_re= comp_12_linspace = np.linspace(min(re_s_kpc),max(re_s_kpc),100)

##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & cD)
re_sersic_cd_kpc,re_sersic_kpc=re_s_kpc[cd_lim],re_s_kpc[elip_lim]
mue_sersic_cd,mue_sersic=mue_med_s[cd_lim],mue_med_s[elip_lim]

[alpha_cd,beta_cd],cov_cd = np.polyfit(re_sersic_cd_kpc,mue_sersic_cd,1,cov=True)
[alpha_s,beta_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

linha_cd = alpha_cd * comp_12_linspace + beta_cd
linha_sersic = alpha_s * comp_12_linspace + beta_s

alpha_cd_label,beta_cd_label = format(alpha_cd,'.3'),format(beta_cd,'.3') 
alpha_ser_label,beta_ser_label = format(alpha_s,'.3'),format(beta_s,'.3')

fig,axs=plt.subplots(1,1,sharey=True,sharex=True,figsize=(10,10))
axs.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
axs.plot(linspace_re, linha_cd, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_ser_label}$'+'\n'+fr'$\beta={beta_ser_label}$')
axs.set_ylim(y2s,y1s)
axs.set_xlim(x1s,x2s)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()

plt.savefig(f'{sample}_stats_observation_desi/p_value/rel_kormendy_cd_e.png')
plt.close()
#CONJUNTO - CORTE POR DELTA BIC
re_intern,re_extern,re_sersic_kpc=re_1_kpc[cd_lim],re_2_kpc[cd_lim],re_s_kpc[elip_lim]
mue_intern,mue_extern,mue_sersic=mue_med_1[cd_lim],mue_med_2[cd_lim],mue_med_s[elip_lim]

[alpha_med_1,beta_med_1],cov_1 = np.polyfit(re_intern,mue_intern,1,cov=True)
[alpha_med_2,beta_med_2],cov_2 = np.polyfit(re_extern,mue_extern,1,cov=True)
[alpha_med_s,beta_med_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

linha_interna = alpha_med_1 * comp_12_linspace + beta_med_1
linha_externa = alpha_med_2 * comp_12_linspace + beta_med_2
linha_sersic = alpha_med_s * comp_12_linspace + beta_med_s

alpha_med_int,beta_med_int = format(alpha_med_1,'.3'),format(beta_med_1,'.3') 
alpha_med_ext,beta_med_ext = format(alpha_med_2,'.3'),format(beta_med_2,'.3')
alpha_med_ser,beta_med_ser = format(alpha_med_s,'.3'),format(beta_med_s,'.3')

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
axs.plot(linspace_re, linha_externa, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()

plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
plt.close()

#####################################################
lim_small_center=re_intern<0.673
lim_big_center=~lim_small_center

re_sersic_cd_kpc_small,re_sersic_cd_kpc_big=re_sersic_cd_kpc[lim_small_center],re_sersic_cd_kpc[lim_big_center]
mue_sersic_cd_small,mue_sersic_cd_big=mue_sersic_cd[lim_small_center],mue_sersic_cd[lim_big_center]

[alpha_cd_small,beta_cd_small],cov_cd_small = np.polyfit(re_sersic_cd_kpc_small,mue_sersic_cd_small,1,cov=True)
[alpha_cd_big,beta_cd_big],cov_cd_big = np.polyfit(re_sersic_cd_kpc_big,mue_sersic_cd_big,1,cov=True)

linha_cd_small = alpha_cd_small * comp_12_linspace + beta_cd_small
linha_cd_big = alpha_cd_big * comp_12_linspace + beta_cd_big

alpha_cd_label_small,beta_cd_label_small = format(alpha_cd_small,'.3'),format(beta_cd_small,'.3') 
alpha_cd_label_big,beta_cd_label_big = format(alpha_cd_big,'.3'),format(beta_cd_big,'.3') 
#####
re_intern_small,re_extern_small=re_intern[lim_small_center],re_extern[lim_small_center]
mue_intern_small,mue_extern_small=mue_intern[lim_small_center],mue_extern[lim_small_center]

[alpha_med_1_small,beta_med_1_small],cov_1_small = np.polyfit(re_intern_small,mue_intern_small,1,cov=True)
[alpha_med_2_small,beta_med_2_small],cov_2_small = np.polyfit(re_extern_small,mue_extern_small,1,cov=True)

linha_intern_small = alpha_med_1_small * comp_12_linspace + beta_med_1_small
linha_extern_small = alpha_med_2_small * comp_12_linspace + beta_med_2_small

alpha_med_int_small,beta_med_int_small = format(alpha_med_1_small,'.3'),format(beta_med_1_small,'.3') 
alpha_med_ext_small,beta_med_ext_small = format(alpha_med_2_small,'.3'),format(beta_med_2_small,'.3')
##
re_intern_big,re_extern_big=re_intern[lim_big_center],re_extern[lim_big_center]
mue_intern_big,mue_extern_big=mue_intern[lim_big_center],mue_extern[lim_big_center]

[alpha_med_1_big,beta_med_1_big],cov_1_big = np.polyfit(re_intern_big,mue_intern_big,1,cov=True)
[alpha_med_2_big,beta_med_2_big],cov_2_big = np.polyfit(re_extern_big,mue_extern_big,1,cov=True)

linha_intern_big = alpha_med_1_big * comp_12_linspace + beta_med_1_big
linha_extern_big = alpha_med_2_big * comp_12_linspace + beta_med_2_big

alpha_med_int_big,beta_med_int_big = format(alpha_med_1_big,'.3'),format(beta_med_1_big,'.3') 
alpha_med_ext_big,beta_med_ext_big = format(alpha_med_2_big,'.3'),format(beta_med_2_big,'.3')

if sample=='L07':
	# e_cut=tipo_morf=='E'
	# cd_cut=tipo_morf=='cD'
	# ecd_cut=tipo_morf=='E/cD'
	# cde_cut=tipo_morf=='cD/E'

	########################################
	#GERAL
	sample_count=len(cluster)
	e_sample=len(cluster[e_cut])+len(cluster[ecd_cut])
	cd_sample=len(cluster[cd_cut])+len(cluster[cde_cut])
	inc_sample=len(cluster[ecd_cut])+len(cluster[cde_cut])

	s_all=len(cluster[elip_lim])
	s_elip=len(cluster[e_cut & elip_lim])
	s_cds=len(cluster[cd_cut & elip_lim])
	s_inc=len(cluster[cde_cut & elip_lim])+len(cluster[ecd_cut & elip_lim])

	ss_all=len(cluster[cd_lim])
	ss_elip=len(cluster[e_cut & cd_lim])
	ss_cds=len(cluster[cd_cut & cd_lim])
	ss_inc=len(cluster[cde_cut & cd_lim])+len(cluster[ecd_cut & cd_lim])

	####################################

	#POR CENTAGENS
	#percentuais totais
	s_budget=100.*(s_all/sample_count)
	ss_budget=100.*(ss_all/sample_count)
	#acertos/erro
	#S
	s_elip_budget=100.*(s_elip/e_sample)
	s_cd_budget=100.*(s_cds/cd_sample)
	s_misc_budget=100.*(s_inc/inc_sample)
	#SS
	ss_cd_budget=100.*(ss_cds/cd_sample)
	ss_elip_budget=100.*(ss_elip/e_sample)
	ss_misc_budget=100.*(ss_inc/inc_sample)

	vec=np.asarray([[ss_elip,ss_inc,ss_cds],[s_elip,s_inc,s_cds]])
	labelsx = ['cD','E/cD & cD/E','E']
	labelsy = ['True cD','True E']
	plt.figure(figsize=(6, 4))
	sns.heatmap(vec, annot=True, fmt='d', cmap='Reds',
	            xticklabels=np.flip(labelsx), yticklabels=labelsy)
	plt.savefig('L07_stats_observation_desi/confusion_matrix_morf_2class.png')
	plt.close()

	vec=np.asarray([[ss_elip_budget,ss_misc_budget,ss_cd_budget],[s_elip_budget,s_misc_budget,s_cd_budget]])
	labelsx = ['cD','E/cD & cD/E','E']
	labelsy = ['True cD','True E']
	plt.figure(figsize=(6, 4))
	sns.heatmap(vec, annot=True, fmt='.2f', cmap='Reds',
	            xticklabels=np.flip(labelsx), yticklabels=labelsy)
	plt.savefig('L07_stats_observation_desi/confusion_matrix_morf_2class_porcentag.png')
	plt.close()
	################################################
	#E,cD & E(EL)

	#E(EL)
	ss_small_all=len(cluster[lim_cd_small])
	ss_small_elip=len(cluster[e_cut & lim_cd_small])
	ss_small_cds=len(cluster[cd_cut & lim_cd_small])
	ss_small_inc=len(cluster[cde_cut & lim_cd_small])+len(cluster[ecd_cut & lim_cd_small])

	#cD
	ss_big_all=len(cluster[lim_cd_big])
	ss_big_elip=len(cluster[e_cut & lim_cd_big])
	ss_big_cds=len(cluster[cd_cut & lim_cd_big])
	ss_big_inc=len(cluster[cde_cut & lim_cd_big])+len(cluster[ecd_cut & lim_cd_big])

	####################################

	#POR CENTAGENS
	#percentuais totais
	s_budget=100.*(s_all/sample_count)
	ss_small_budget=100.*(ss_small_all/sample_count)
	ss_big_budget=100.*(ss_big_all/sample_count)

	#acertos/erro
	#E(EL)
	ss_small_cd_budget=100.*(ss_small_cds/cd_sample)
	ss_small_elip_budget=100.*(ss_small_elip/e_sample)
	ss_small_misc_budget=100.*(ss_small_inc/inc_sample)

	#cD
	ss_big_cd_budget=100.*(ss_big_cds/cd_sample)
	ss_big_elip_budget=100.*(ss_big_elip/e_sample)
	ss_big_misc_budget=100.*(ss_big_inc/inc_sample)

	vec=np.asarray([[ss_small_elip,ss_small_inc,ss_small_cds],[ss_big_elip,ss_big_inc,ss_big_cds],[s_elip,s_inc,s_cds]])
	labelsx = ['cD','E/cD & cD/E','E']
	labelsy = ['True E(EL)','True cD','True E']
	plt.figure(figsize=(6, 4))
	sns.heatmap(vec, annot=True, fmt='d', cmap='Reds',
	            xticklabels=np.flip(labelsx), yticklabels=labelsy)
	plt.savefig('L07_stats_observation_desi/confusion_matrix_morf_3class.png')
	plt.close()

	vec=np.asarray([[ss_small_elip_budget,ss_small_misc_budget,ss_small_cd_budget],[ss_big_elip_budget,ss_big_misc_budget,ss_big_cd_budget],[s_elip_budget,s_misc_budget,s_cd_budget]])
	labelsx = ['cD','E/cD & cD/E','E']
	labelsy = ['True E(EL)','True cD','True E']
	plt.figure(figsize=(6, 4))
	sns.heatmap(vec, annot=True, fmt='.3f', cmap='Reds',
	            xticklabels=np.flip(labelsx), yticklabels=labelsy)
	plt.savefig('L07_stats_observation_desi/confusion_matrix_morf_3class_porcentag.png')
	plt.close()
###################################################################################
###################################################################################
###################################################################################
#DEFINIÇÃO DOS PARÂMETROS/LIMITES/DISTRIBUIÇÕES DE INTERESSE E FATORES PARA OS PLOTS KDE
names_simples=['2C','E(EL)','cD','E']
names_comps=['2C','E(EL)','cD','E']
names_morf=['cD','E/cD & cD/E','E']
cores=['black','blue','red','green']
line_width=[4,1,1,1]
alpha_vec=[0.4,1,1,1]


#################################################
#PARÂMETROS DA RELAÇÃO DE KORMENDY

#RAIO EFETIVO
re_kde_entry=np.append(re_sersic_kpc,[re_intern,re_extern])
kde_re=kde(re_kde_entry)
re_factor = kde_re.factor
re_linspace=np.linspace(x2ss,x1ss,3000)

#BRILHO EFETIVO MÉDIO
mue_kde_entry=np.append(mue_sersic,[mue_intern,mue_extern])
kde_mue=kde(mue_kde_entry)
mue_factor = kde_mue.factor
mue_linspace=np.linspace(y2ss,y1ss,3000)

#SERSIC SIMPLES -- E(EL),cD,E
#RAIO EFETIVO
re_kde_cd_kpc_small,re_kde_cd_kpc_big,re_kde_cd_kpc=kde(re_sersic_cd_kpc_small,bw_method=re_factor),kde(re_sersic_cd_kpc_big,bw_method=re_factor),kde(re_sersic_cd_kpc,bw_method=re_factor)
#BRILHO EFETIVO MÉDIO
mue_kde_cd_kpc_small,mue_kde_cd_kpc_big,mue_kde_cd_kpc=kde(mue_sersic_cd_small,bw_method=mue_factor),kde(mue_sersic_cd_big,bw_method=mue_factor),kde(mue_sersic_cd,bw_method=mue_factor)

#GERAL(SEM SEPARAR EL DAS cDS) -- COMPONENTES -- INTERNO,EXTERNO,SERSIC
re_kde_intern,re_kde_extern,re_kde_sersic_kpc=kde(re_intern,bw_method=re_factor),kde(re_extern,bw_method=re_factor),kde(re_sersic_kpc,bw_method=re_factor)
mue_kde_intern,mue_kde_extern,mue_kde_sersic_kpc=kde(mue_intern,bw_method=mue_factor),kde(mue_extern,bw_method=mue_factor),kde(mue_sersic,bw_method=mue_factor)

##
#ELIPTICAS (EXTRA-LIGHT) -- COMPONENTES -- INTERNO,EXTERNO
re_kde_intern_small,re_kde_extern_small=kde(re_intern[lim_small_center],bw_method=re_factor),kde(re_extern[lim_small_center],bw_method=re_factor)
mue_kde_intern_small,mue_kde_extern_small=kde(mue_intern[lim_small_center],bw_method=mue_factor),kde(mue_extern[lim_small_center],bw_method=mue_factor)
##
#cDs CONVENCIONAIS -- COMPONENTES -- INTERNO,EXTERNO
re_kde_intern_big,re_kde_extern_big=kde(re_intern[lim_big_center],bw_method=re_factor),kde(re_extern[lim_big_center],bw_method=re_factor)
mue_kde_intern_big,mue_kde_extern_big=kde(mue_intern[lim_big_center],bw_method=mue_factor),kde(mue_extern[lim_big_center],bw_method=mue_factor)
##
##################################
#INDICE DE SÉRSIC 
##SÉRSIC -- cD,E(EL),cD(std),E
n_sersic_cd,n_sersic_cd_small,n_sersic_cd_big,n_sersic_e=n_s[cd_lim],n_s[lim_cd_small],n_s[lim_cd_big],n_s[elip_lim]
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL)
n_interno_cd,n_externo_cd,n_interno_cd_big,n_externo_cd_big,n_interno_small,n_externo_small,n_interno_elip,n_externo_elip=n1[cd_lim],n2[cd_lim],n1[lim_cd_big],n2[lim_cd_big],n1[lim_cd_small],n2[lim_cd_small],n1[elip_lim],n2[elip_lim]

##
ks_n_sersic=ks_calc([n_sersic_cd,n_sersic_cd_small,n_sersic_cd_big,n_sersic_e])
ks_n_intern=ks_calc([n_interno_cd,n_interno_small,n_interno_cd_big,n_interno_elip])
ks_n_extern=ks_calc([n_externo_cd,n_externo_small,n_externo_cd_big,n_externo_elip])
##
vec_ks_n_sersic=ks_n_sersic[0][1][2],ks_n_sersic[0][1][3],ks_n_sersic[0][2][3]
vec_ks_n_intern=ks_n_intern[0][1][2],ks_n_intern[0][1][3],ks_n_intern[0][2][3]
vec_ks_n_extern=ks_n_extern[0][1][2],ks_n_extern[0][1][3],ks_n_extern[0][2][3]
#
vec_ks_n_2c=ks_n_sersic[0][0][1],ks_n_sersic[0][0][2],ks_n_sersic[0][0][3]
vec_ks_n_intern_2c=ks_n_intern[0][0][1],ks_n_intern[0][0][2],ks_n_intern[0][0][3]
vec_ks_n_extern_2c=ks_n_extern[0][1][2],ks_n_extern[0][0][2],ks_n_extern[0][0][3]
##

vec_med_n_sersic=med_n_sersic_cd,med_n_sersic_cd_small,med_n_sersic_cd_big,med_n_sersic_e=np.average(n_sersic_cd),np.average(n_sersic_cd_small),np.average(n_sersic_cd_big),np.average(n_sersic_e)
vec_med_n_interno=med_n_interno_cd,med_n_interno_small,med_n_interno_cd_big,med_n_interno_elip=np.average(n_interno_cd),np.average(n_interno_small),np.average(n_interno_cd_big),np.average(n_interno_elip)
vec_med_n_externo=med_n_interno_cd,med_n_externo_small,med_n_externo_cd_big,med_n_externo_elip=np.average(n_externo_cd),np.average(n_externo_small),np.average(n_externo_cd_big),np.average(n_externo_elip)

##
n_kde_entry=np.append(n_s,[n1,n2])
kde_n=kde(n_kde_entry)
n_factor = kde_n.factor
n_linspace=np.linspace(min(n_kde_entry),max(n_kde_entry),3000)
##
#IMAGENS INDICE DE SÉRSIC
##SÉRSIC -- cD,E(EL),cD(std),E - KDE
vec_kde_n_simples=n_sersic_kde_cd,n_sersic_kde_cd_small,n_sersic_kde_cd_big,n_sersic_kde_e=kde(n_sersic_cd,bw_method=n_factor),kde(n_sersic_cd_small,bw_method=n_factor),kde(n_sersic_cd_big,bw_method=n_factor),kde(n_sersic_e,bw_method=n_factor)
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
vec_kde_n_intern=n_interno_kde_cd,n_interno_kde_small,n_interno_kde_cd_big,n_interno_kde_e=kde(n_interno_cd,bw_method=n_factor),kde(n_interno_small,bw_method=n_factor),kde(n_interno_cd_big,bw_method=n_factor),kde(n_interno_elip,bw_method=n_factor)
vec_kde_n_extern=n_externo_kde_cd,n_externo_kde_small,n_externo_kde_cd_big,n_externo_kde_e=kde(n_externo_cd,bw_method=n_factor),kde(n_externo_small,bw_method=n_factor),kde(n_externo_cd_big,bw_method=n_factor),kde(n_externo_elip,bw_method=n_factor)
##
os.makedirs(f'{sample}_stats_observation_desi/test_ks/indice_sersic',exist_ok=True)
#MODELO SIMPLES
plt.figure()
plt.title('Indice de Sérsic - Modelos Simples - p_value')
sns.heatmap(ks_n_sersic[0],xticklabels=names_simples,yticklabels=names_simples,annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Modelos Simples - D_value')
sns.heatmap(ks_n_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_simples_dvalue.png')
plt.close()

##
#MODELO COMPOSTO - COMPONENTE INTERNO
plt.figure()
plt.title('Indice de Sérsic - Componente interno - p_value')
sns.heatmap(ks_n_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Componente interno - D_value')
sns.heatmap(ks_n_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_interno_dvalue.png')
plt.close()

##
#MODELO COMPOSTO - COMPONENTE EXTERNO
plt.figure()
plt.title('Indice de Sérsic - Componente externo - p_value')
sns.heatmap(ks_n_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Componente externo - D_value')
sns.heatmap(ks_n_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Modelos Simples')
for i,dist in enumerate(vec_kde_n_simples):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_n_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_sersic[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_n_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_sersic[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_n_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_n_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Componente interno')
for i,dist in enumerate(vec_kde_n_intern):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_n_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_interno[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_n_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_intern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_n_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_intern_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_n_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_intern_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Componente externo')
for i,dist in enumerate(vec_kde_n_extern):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_n_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_n_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_extern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_n_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_extern_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_n_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_extern_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Componente externo vs Modelo simples')
axs.plot(n_linspace,n_sersic_kde_e(n_linspace),color='green',label=f'{names_simples[-1]}')
axs.axvline(vec_med_n_sersic[3],color='green',ls='--',label=fr'$\mu = {vec_med_n_sersic[3]:.3f}$')
axs.plot(n_linspace,n_externo_kde_small(n_linspace),color='blue',label=f'{names_simples[1]}')
axs.axvline(vec_med_n_externo[1],color='blue',ls='--',label=fr'$\mu = {vec_med_n_externo[1]:.3f}$')
axs.legend()
axs.set_xlabel(r'$n$')
ks_e_el=ks_2samp(n_sersic_e,n_externo_small)[1]
info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_EL_E_kde.png')
plt.close()

if sample == 'L07':
	##SÉRSIC -- cD,E(EL),cD(std),E
	#CLASSIFICAÇÃO ZHAO PURA
	n_sersic_cd_zhao,n_sersic_e_zhao=n_s[cd_cut],n_s[e_cut]	
	##
	#CLASSIFICAÇÃO NOSSA / SUBGRUPOS MORFOLOGICOS DO ZHAO
	n_sersic_cd_E,n_sersic_cd_cD,n_sersic_cd_misc=n_s[cd_lim & e_cut],n_s[cd_lim & cd_cut],n_s[cd_lim & (ecd_cut | cde_cut)]
	n_sersic_cd_small_E,n_sersic_cd_small_cD,n_sersic_cd_small_misc=n_s[lim_cd_small & e_cut],n_s[lim_cd_small & cd_cut],n_s[lim_cd_small & (ecd_cut | cde_cut)]
	n_sersic_cd_big_E,n_sersic_cd_big_cD,n_sersic_cd_big_misc=n_s[lim_cd_big & e_cut],n_s[lim_cd_big & cd_cut],n_s[lim_cd_big & (ecd_cut | cde_cut)]
	n_sersic_e_E,n_sersic_e_cD,n_sersic_e_misc=n_s[elip_lim & e_cut],n_s[elip_lim & cd_cut],n_s[elip_lim & (ecd_cut | cde_cut)]

	##COMPONENTES -- INTERNO/EXTERNO cD(any),INTERNO/EXTERNO E DAS CLASSIFICAÇÕES ZHAO
	n_interno_cd_zhao,n_externo_cd_zhao,n_interno_e_zhao,n_externo_e_zhao=n1[cd_cut],n2[cd_cut],n1[e_cut],n2[e_cut]
	#
	#COMPONENTES -- INTERNO/EXTERNO NOSSO COM SUBDIVISÃO POR ZHAO 
	n_interno_cd_E,n_interno_cd_cD,n_interno_cd_misc=n1[cd_lim & e_cut],n1[cd_lim & cd_cut],n1[cd_lim & (ecd_cut | cde_cut)]
	n_externo_cd_E,n_externo_cd_cD,n_externo_cd_misc=n2[cd_lim & e_cut],n2[cd_lim & cd_cut],n2[cd_lim & (ecd_cut | cde_cut)]
	#
	n_interno_cd_big_E,n_interno_cd_big_cD,n_interno_cd_big_misc=n1[lim_cd_big & e_cut],n1[lim_cd_big & cd_cut],n1[lim_cd_big & (ecd_cut | cde_cut)]
	n_externo_cd_big_E,n_externo_cd_big_cD,n_externo_cd_big_misc=n2[lim_cd_big & e_cut],n2[lim_cd_big & cd_cut],n2[lim_cd_big & (ecd_cut | cde_cut)]
	#
	n_interno_cd_small_E,n_interno_cd_small_cD,n_interno_cd_small_misc=n1[lim_cd_small & e_cut],n1[lim_cd_small & cd_cut],n1[lim_cd_small & (ecd_cut | cde_cut)]
	n_externo_cd_small_E,n_externo_cd_small_cD,n_externo_cd_small_misc=n2[lim_cd_small & e_cut],n2[lim_cd_small & cd_cut],n2[lim_cd_small & (ecd_cut | cde_cut)]
	#
	n_interno_elip_E,n_interno_elip_cD,n_interno_elip_misc=n1[elip_lim & e_cut],n1[elip_lim & cd_cut],n1[elip_lim & (ecd_cut | cde_cut)]
	n_externo_elip_E,n_externo_elip_cD,n_externo_elip_misc=n2[elip_lim & e_cut],n2[elip_lim & cd_cut],n2[elip_lim & (ecd_cut | cde_cut)]
	
	###
	ks_n_zhao=ks_2samp(n_sersic_cd,n_sersic_e)[1],ks_2samp(n_sersic_cd_zhao,n_sersic_e_zhao)[1]
	ks_n_intern_zhao=ks_2samp(n_interno_cd,n_interno_elip)[1],ks_2samp(n_interno_cd_zhao,n_interno_e_zhao)[1]
	ks_n_extern_zhao=ks_2samp(n_externo_cd,n_externo_elip)[1],ks_2samp(n_externo_cd_zhao,n_externo_e_zhao)[1]
	comp_ks_zhao=np.vstack([ks_n_zhao,ks_n_intern_zhao,ks_n_extern_zhao])
	###
	ks_n_cD_zhao=ks_2samp(n_sersic_cd_cD,n_sersic_e_cD)[1],ks_2samp(n_sersic_cd_big_cD,n_sersic_e_cD)[1],ks_2samp(n_sersic_cd_big_cD,n_sersic_cd_small_cD)[1],ks_2samp(n_sersic_e_cD,n_sersic_cd_small_cD)[1]
	ks_n_intern_cD_zhao=ks_2samp(n_interno_cd_cD,n_interno_elip_cD)[1],ks_2samp(n_interno_cd_big_cD,n_interno_elip_cD)[1],ks_2samp(n_interno_cd_big_cD,n_interno_cd_small_cD)[1],ks_2samp(n_interno_elip_cD,n_interno_cd_small_cD)[1]
	ks_n_extern_cD_zhao=ks_2samp(n_externo_cd_cD,n_externo_elip_cD)[1],ks_2samp(n_externo_cd_big_cD,n_externo_elip_cD)[1],ks_2samp(n_externo_cd_big_cD,n_externo_cd_small_cD)[1],ks_2samp(n_externo_elip_cD,n_externo_cd_small_cD)[1]

	vec_med_n_sersic_cD=med_n_sersic_cd_cD,med_n_sersic_cd_small_cD,med_n_sersic_cd_big_cD,med_n_sersic_e_cD=np.average(n_sersic_cd_cD),np.average(n_sersic_cd_small_cD),np.average(n_sersic_cd_big_cD),np.average(n_sersic_e_cD)
	vec_med_n_interno_cD=med_n_interno_cd_cD,med_n_interno_small_cD,med_n_interno_cd_big_cD,med_n_interno_elip_cD=np.average(n_interno_cd_cD),np.average(n_interno_cd_small_cD),np.average(n_interno_cd_big_cD),np.average(n_interno_elip_cD)
	vec_med_n_externo_cD=med_n_interno_cd_cD,med_n_externo_small_cD,med_n_externo_cd_big_cD,med_n_externo_elip_cD=np.average(n_externo_cd_cD),np.average(n_externo_cd_small_cD),np.average(n_externo_cd_big_cD),np.average(n_externo_elip_cD)

	#IMAGENS INDICE DE SÉRSIC
	##SÉRSIC -- cD,E(EL),cD(std),E - KDE
	vec_kde_n_simples_zhao=n_sersic_kde_cd_cD,n_sersic_kde_cd_small_cD,n_sersic_kde_cd_big_cD,n_sersic_kde_e_cD=kde(n_sersic_cd_cD,bw_method=n_factor),kde(n_sersic_cd_small_cD,bw_method=n_factor),kde(n_sersic_cd_big_cD,bw_method=n_factor),kde(n_sersic_e_cD,bw_method=n_factor)
	n_sersic_kde_e_misc=kde(n_sersic_e_misc,bw_method=n_factor)	
	##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
	vec_kde_n_intern_cD=n_interno_kde_cd_cD,n_interno_kde_small_cD,n_interno_kde_cd_big_cD,n_interno_kde_e_cD=kde(n_interno_cd_cD,bw_method=n_factor),kde(n_interno_cd_small_cD,bw_method=n_factor),kde(n_interno_cd_big_cD,bw_method=n_factor),kde(n_interno_elip_cD,bw_method=n_factor)
	vec_kde_n_extern_cD=n_externo_kde_cd_cD,n_externo_kde_small_cD,n_externo_kde_cd_big_cD,n_externo_kde_e_cD=kde(n_externo_cd_cD,bw_method=n_factor),kde(n_externo_cd_small_cD,bw_method=n_factor),kde(n_externo_cd_big_cD,bw_method=n_factor),kde(n_externo_elip_cD,bw_method=n_factor)

	fig,ax=plt.subplots()
	plt.title('Indice de Sérsic - Comparação de classificações')
	sns.heatmap(comp_ks_zhao,fmt='.3e',xticklabels=['Nossa','Zhao'],yticklabels=['cD vs. E','cD vs. E','cD vs. E'], annot=True, cmap='Reds')
	ax.xaxis.tick_top()
	ax_right = ax.twinx()
	ax_right.set_ylim(ax.get_ylim())
	ax_right.set_yticks(ax.get_yticks())
	ax_right.set_yticklabels(['n',r'$n_1$',r'$n_2$'])
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_n_simples_zhao):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_n_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_n_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_n_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_n_intern_cD):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_n_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_n_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_intern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_n_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_n_extern_cD):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_n_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_n_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_extern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_n_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Componente externo vs Modelo simples - cD Zhao')
	axs.plot(n_linspace,n_sersic_kde_e_cD(n_linspace),color='green',label=f'{names_simples[-1]}')
	axs.axvline(vec_med_n_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_n_sersic_cD[3]:.3f}$')
	axs.plot(n_linspace,n_externo_kde_small_cD(n_linspace),color='blue',label=f'{names_simples[1]}')
	axs.axvline(vec_med_n_externo_cD[1],color='blue',ls='--',label=fr'$\mu = {vec_med_n_externo_cD[1]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	ks_e_el=ks_2samp(n_sersic_e_cD,n_externo_cd_small_E)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_externo_EL_E_kde_cD_zhao.png')
	plt.close()

	#MODELO SIMPLES - ELIP cD X MISC

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - cD vs. E/cD & cD/E - Zhao')
	axs.plot(n_linspace,n_sersic_kde_e_cD(n_linspace),color='green',label=f'{names_simples[-1]} cD')
	axs.axvline(vec_med_n_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_n_sersic_cD[3]:.3f}$')
	axs.plot(n_linspace,n_sersic_kde_e_misc(n_linspace),color='black',label=f'{names_simples[-1]} E/cD & cD/E')
	axs.axvline(np.average(n_sersic_e_misc),color='black',ls='--',label=fr'$\mu = {np.average(n_sersic_e_misc):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	ks_e_cd_misc=ks_2samp(n_sersic_e_cD,n_sersic_e_misc)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_cd_misc:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/indice_sersic/n_simples_E_kde_cD_misc_zhao.png')
	plt.close()
#####################################################
#RAZÃO AXIAL
##SÉRSIC -- cD,E(EL),cD(std),E
axrat_sersic_cd,axrat_sersic_cd_small,axrat_sersic_cd_big,axrat_sersic_e=e_s[cd_lim],e_s[lim_cd_small],e_s[lim_cd_big],e_s[elip_lim]
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL)
axrat_interno_cd,axrat_externo_cd,axrat_interno_cd_big,axrat_externo_cd_big,axrat_interno_small,axrat_externo_small,axrat_interno_elip,axrat_externo_elip=e1[cd_lim],e2[cd_lim],e1[lim_cd_big],e2[lim_cd_big],e1[lim_cd_small],e2[lim_cd_small],e1[elip_lim],e2[elip_lim]

ks_axrat_sersic=ks_calc([axrat_sersic_cd,axrat_sersic_cd_small,axrat_sersic_cd_big,axrat_sersic_e])
ks_axrat_intern=ks_calc([axrat_interno_cd,axrat_interno_small,axrat_interno_cd_big,axrat_interno_elip])
ks_axrat_extern=ks_calc([axrat_externo_cd,axrat_externo_small,axrat_externo_cd_big,axrat_interno_elip])
##
vec_ks_axrat_sersic=ks_axrat_sersic[0][1][2],ks_axrat_sersic[0][1][3],ks_axrat_sersic[0][2][3]
vec_ks_axrat_intern=ks_axrat_intern[0][1][2],ks_axrat_intern[0][1][3],ks_axrat_intern[0][2][3]
vec_ks_axrat_extern=ks_axrat_extern[0][1][2],ks_axrat_extern[0][1][3],ks_axrat_extern[0][2][3]
#
vec_ks_axrat_2c=ks_axrat_sersic[0][0][1],ks_axrat_sersic[0][0][2],ks_axrat_sersic[0][0][3]
vec_ks_axrat_intern_2c=ks_axrat_intern[0][0][1],ks_axrat_intern[0][0][2],ks_axrat_intern[0][0][3]
vec_ks_axrat_extern_2c=ks_axrat_extern[0][1][2],ks_axrat_extern[0][0][2],ks_axrat_extern[0][0][3]
##
vec_med_axrat_sersic=med_axrat_sersic_cd,med_axrat_sersic_cd_small,med_axrat_sersic_cd_big,med_axrat_sersic_e=np.average(axrat_sersic_cd),np.average(axrat_sersic_cd_small),np.average(axrat_sersic_cd_big),np.average(axrat_sersic_e)
vec_med_axrat_interno=med_axrat_interno_cd,med_axrat_interno_small,med_axrat_interno_cd_big,med_axrat_interno_elip=np.average(axrat_interno_cd),np.average(axrat_interno_small),np.average(axrat_interno_cd_big),np.average(axrat_interno_elip)
vec_med_axrat_externo=med_axrat_interno_cd,med_axrat_externo_small,med_axrat_externo_cd_big,med_axrat_externo_elip=np.average(axrat_externo_cd),np.average(axrat_externo_small),np.average(axrat_externo_cd_big),np.average(axrat_externo_elip)
#
axrat_kde_entry=np.append(e_s,[e1,e2])
kde_axrat=kde(axrat_kde_entry)
axrat_factor = kde_axrat.factor
axrat_linspace=np.linspace(min(axrat_kde_entry),max(axrat_kde_entry),3000)

##SÉRSIC -- cD,E(EL),cD(std),E
axrat_sersic_kde_cd,axrat_sersic_kde_cd_small,axrat_sersic_kde_cd_big,axrat_sersic_kde_e=kde(axrat_sersic_cd,bw_method=axrat_factor),kde(axrat_sersic_cd_small,bw_method=axrat_factor),kde(axrat_sersic_cd_big,bw_method=axrat_factor),kde(axrat_sersic_e,bw_method=axrat_factor)
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL)
axrat_interno_kde_cd,axrat_externo_kde_cd,axrat_interno_kde_cd_big,axrat_externo_kde_cd_big,axrat_interno_kde_small,axrat_externo_kde_small,axrat_interno_kde_elip,axrat_externo_kde_elip=kde(axrat_interno_cd,bw_method=axrat_factor),kde(axrat_externo_cd,bw_method=axrat_factor),kde(axrat_interno_cd_big,bw_method=axrat_factor),kde(axrat_externo_cd_big,bw_method=axrat_factor),kde(axrat_interno_small,bw_method=axrat_factor),kde(axrat_externo_small,bw_method=axrat_factor),kde(axrat_interno_elip,bw_method=axrat_factor),kde(axrat_externo_elip,bw_method=axrat_factor)

vec_kde_axrat_simples=axrat_sersic_kde_cd,axrat_sersic_kde_cd_small,axrat_sersic_kde_cd_big,axrat_sersic_kde_e
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
vec_kde_axrat_intern=axrat_interno_kde_cd,axrat_interno_kde_small,axrat_interno_kde_cd_big,axrat_interno_kde_elip
vec_kde_axrat_extern=axrat_externo_kde_cd,axrat_externo_kde_small,axrat_externo_kde_cd_big,axrat_externo_kde_elip

#IMAGENS RAZÃO AXIAL
os.makedirs(f'{sample}_stats_observation_desi/test_ks/ax_ratio',exist_ok=True)

##
plt.figure()
plt.title('Razão Axial - Modelos Simples - p_value')
sns.heatmap(ks_axrat_sersic[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Modelos Simples - D_value')
sns.heatmap(ks_axrat_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_simples_dvalue.png')
plt.close()

##
plt.figure()
plt.title('Razão Axial - Componente interno - p_value')
sns.heatmap(ks_axrat_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Componente interno - D_value')
sns.heatmap(ks_axrat_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_interno_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Razão Axial - Componente externo - p_value')
sns.heatmap(ks_axrat_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Componente externo - D_value')
sns.heatmap(ks_axrat_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Razão Axial - Modelos Simples')
for i,dist in enumerate(vec_kde_axrat_simples):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_axrat_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_sersic[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_axrat_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_sersic[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_axrat_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_axrat_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Razão Axial - Componente interno')
for i,dist in enumerate(vec_kde_axrat_intern):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_axrat_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_interno[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_axrat_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_intern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_axrat_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_intern_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_axrat_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_intern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Razão Axial - Componente externo')
for i,dist in enumerate(vec_kde_axrat_extern):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_axrat_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_axrat_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_extern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_axrat_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_extern_2c[0]:.3e}\n' f'K-S(2C,True cD)={vec_ks_axrat_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_extern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Razão Axial - Componente externo vs Modelo simples')
axs.plot(axrat_linspace,axrat_sersic_kde_e(axrat_linspace),color='green',label=f'{names_simples[-1]}')
axs.axvline(vec_med_axrat_sersic[3],color='green',ls='--',label=fr'$\mu = {vec_med_axrat_sersic[3]:.3f}$')
axs.plot(axrat_linspace,axrat_externo_kde_small(axrat_linspace),color='blue',label=f'{names_simples[1]}')
axs.axvline(vec_med_axrat_externo[1],color='blue',ls='--',label=fr'$\mu = {vec_med_axrat_externo[1]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
ks_e_el=ks_2samp(axrat_sersic_e,axrat_externo_small)[1]
info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_EL_E_kde.png')
plt.close()

if sample == 'L07':
	##SÉRSIC -- cD,E(EL),cD(std),E
	#CLASSIFICAÇÃO ZHAO PURA
	axrat_sersic_cd_zhao,axrat_sersic_e_zhao=e_s[cd_cut],e_s[e_cut]	
	##
	#CLASSIFICAÇÃO NOSSA / SUBGRUPOS MORFOLOGICOS DO ZHAO
	axrat_sersic_cd_E,axrat_sersic_cd_cD,axrat_sersic_cd_misc=e_s[cd_lim & e_cut],e_s[cd_lim & cd_cut],e_s[cd_lim & (ecd_cut | cde_cut)]
	axrat_sersic_cd_small_E,axrat_sersic_cd_small_cD,axrat_sersic_cd_small_misc=e_s[lim_cd_small & e_cut],e_s[lim_cd_small & cd_cut],e_s[lim_cd_small & (ecd_cut | cde_cut)]
	axrat_sersic_cd_big_E,axrat_sersic_cd_big_cD,axrat_sersic_cd_big_misc=e_s[lim_cd_big & e_cut],e_s[lim_cd_big & cd_cut],e_s[lim_cd_big & (ecd_cut | cde_cut)]
	axrat_sersic_e_E,axrat_sersic_e_cD,axrat_sersic_e_misc=e_s[elip_lim & e_cut],e_s[elip_lim & cd_cut],e_s[elip_lim & (ecd_cut | cde_cut)]

	##COMPONENTES -- INTERNO/EXTERNO cD(any),INTERNO/EXTERNO E DAS CLASSIFICAÇÕES ZHAO
	axrat_interno_cd_zhao,axrat_externo_cd_zhao,axrat_interno_e_zhao,axrat_externo_e_zhao=e1[cd_cut],e2[cd_cut],e1[e_cut],e2[e_cut]
	#
	#COMPONENTES -- INTERNO/EXTERNO NOSSO COM SUBDIVISÃO POR ZHAO 
	axrat_interno_cd_E,axrat_interno_cd_cD,axrat_interno_cd_misc=e1[cd_lim & e_cut],e1[cd_lim & cd_cut],e1[cd_lim & (ecd_cut | cde_cut)]
	axrat_externo_cd_E,axrat_externo_cd_cD,axrat_externo_cd_misc=e2[cd_lim & e_cut],e2[cd_lim & cd_cut],e2[cd_lim & (ecd_cut | cde_cut)]
	#
	axrat_interno_cd_big_E,axrat_interno_cd_big_cD,axrat_interno_cd_big_misc=e1[lim_cd_big & e_cut],e1[lim_cd_big & cd_cut],e1[lim_cd_big & (ecd_cut | cde_cut)]
	axrat_externo_cd_big_E,axrat_externo_cd_big_cD,axrat_externo_cd_big_misc=e2[lim_cd_big & e_cut],e2[lim_cd_big & cd_cut],e2[lim_cd_big & (ecd_cut | cde_cut)]
	#
	axrat_interno_cd_small_E,axrat_interno_cd_small_cD,axrat_interno_cd_small_misc=e1[lim_cd_small & e_cut],e1[lim_cd_small & cd_cut],e1[lim_cd_small & (ecd_cut | cde_cut)]
	axrat_externo_cd_small_E,axrat_externo_cd_small_cD,axrat_externo_cd_small_misc=e2[lim_cd_small & e_cut],e2[lim_cd_small & cd_cut],e2[lim_cd_small & (ecd_cut | cde_cut)]
	#
	axrat_interno_elip_E,axrat_interno_elip_cD,axrat_interno_elip_misc=e1[elip_lim & e_cut],e1[elip_lim & cd_cut],e1[elip_lim & (ecd_cut | cde_cut)]
	axrat_externo_elip_E,axrat_externo_elip_cD,axrat_externo_elip_misc=e2[elip_lim & e_cut],e2[elip_lim & cd_cut],e2[elip_lim & (ecd_cut | cde_cut)]
	
	###
	ks_axrat_zhao=ks_2samp(axrat_sersic_cd,axrat_sersic_e)[1],ks_2samp(axrat_sersic_cd_zhao,axrat_sersic_e_zhao)[1]
	ks_axrat_intern_zhao=ks_2samp(axrat_interno_cd,axrat_interno_elip)[1],ks_2samp(axrat_interno_cd_zhao,axrat_interno_e_zhao)[1]
	ks_axrat_extern_zhao=ks_2samp(axrat_externo_cd,axrat_externo_elip)[1],ks_2samp(axrat_externo_cd_zhao,axrat_externo_e_zhao)[1]
	comp_ks_zhao=np.vstack([ks_axrat_zhao,ks_axrat_intern_zhao,ks_axrat_extern_zhao])
	###
	ks_axrat_cD_zhao=ks_2samp(axrat_sersic_cd_cD,axrat_sersic_e_cD)[1],ks_2samp(axrat_sersic_cd_big_cD,axrat_sersic_e_cD)[1],ks_2samp(axrat_sersic_cd_big_cD,axrat_sersic_cd_small_cD)[1],ks_2samp(axrat_sersic_e_cD,axrat_sersic_cd_small_cD)[1]
	ks_axrat_intern_cD_zhao=ks_2samp(axrat_interno_cd_cD,axrat_interno_elip_cD)[1],ks_2samp(axrat_interno_cd_big_cD,axrat_interno_elip_cD)[1],ks_2samp(axrat_interno_cd_big_cD,axrat_interno_cd_small_cD)[1],ks_2samp(axrat_interno_elip_cD,axrat_interno_cd_small_cD)[1]
	ks_axrat_extern_cD_zhao=ks_2samp(axrat_externo_cd_cD,axrat_externo_elip_cD)[1],ks_2samp(axrat_externo_cd_big_cD,axrat_externo_elip_cD)[1],ks_2samp(axrat_externo_cd_big_cD,axrat_externo_cd_small_cD)[1],ks_2samp(axrat_externo_elip_cD,axrat_externo_cd_small_cD)[1]

	vec_med_axrat_sersic_cD=med_axrat_sersic_cd_cD,med_axrat_sersic_cd_small_cD,med_axrat_sersic_cd_big_cD,med_axrat_sersic_e_cD=np.average(axrat_sersic_cd_cD),np.average(axrat_sersic_cd_small_cD),np.average(axrat_sersic_cd_big_cD),np.average(axrat_sersic_e_cD)
	vec_med_axrat_interno_cD=med_axrat_interno_cd_cD,med_axrat_interno_small_cD,med_axrat_interno_cd_big_cD,med_axrat_interno_elip_cD=np.average(axrat_interno_cd_cD),np.average(axrat_interno_cd_small_cD),np.average(axrat_interno_cd_big_cD),np.average(axrat_interno_elip_cD)
	vec_med_axrat_externo_cD=med_axrat_interno_cd_cD,med_axrat_externo_small_cD,med_axrat_externo_cd_big_cD,med_axrat_externo_elip_cD=np.average(axrat_externo_cd_cD),np.average(axrat_externo_cd_small_cD),np.average(axrat_externo_cd_big_cD),np.average(axrat_externo_elip_cD)

	#IMAGENS INDICE DE SÉRSIC
	##SÉRSIC -- cD,E(EL),cD(std),E - KDE
	vec_kde_axrat_simples_zhao=axrat_sersic_kde_cd_cD,axrat_sersic_kde_cd_small_cD,axrat_sersic_kde_cd_big_cD,axrat_sersic_kde_e_cD=kde(axrat_sersic_cd_cD,bw_method=axrat_factor),kde(axrat_sersic_cd_small_cD,bw_method=axrat_factor),kde(axrat_sersic_cd_big_cD,bw_method=axrat_factor),kde(axrat_sersic_e_cD,bw_method=axrat_factor)
	axrat_sersic_kde_e_misc=kde(axrat_sersic_e_misc,bw_method=axrat_factor)	
	##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
	vec_kde_axrat_intern_cD=axrat_interno_kde_cd_cD,axrat_interno_kde_small_cD,axrat_interno_kde_cd_big_cD,axrat_interno_kde_e_cD=kde(axrat_interno_cd_cD,bw_method=axrat_factor),kde(axrat_interno_cd_small_cD,bw_method=axrat_factor),kde(axrat_interno_cd_big_cD,bw_method=axrat_factor),kde(axrat_interno_elip_cD,bw_method=axrat_factor)
	vec_kde_axrat_extern_cD=axrat_externo_kde_cd_cD,axrat_externo_kde_small_cD,axrat_externo_kde_cd_big_cD,axrat_externo_kde_e_cD=kde(axrat_externo_cd_cD,bw_method=axrat_factor),kde(axrat_externo_cd_small_cD,bw_method=axrat_factor),kde(axrat_externo_cd_big_cD,bw_method=axrat_factor),kde(axrat_externo_elip_cD,bw_method=axrat_factor)

	fig,ax=plt.subplots()
	plt.title('Razão Axial Sérsic - Comparação de classificações')
	sns.heatmap(comp_ks_zhao,fmt='.3e',xticklabels=['Nossa','Zhao'],yticklabels=['cD vs. E','cD vs. E','cD vs. E'], annot=True, cmap='Reds')
	ax.xaxis.tick_top()
	ax_right = ax.twinx()
	ax_right.set_ylim(ax.get_ylim())
	ax_right.set_yticks(ax.get_yticks())
	ax_right.set_yticklabels(['q',r'$q_1$',r'$q_2$'])
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_simples_zhao):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_axrat_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$q$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_axrat_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_axrat_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_intern_cD):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_axrat_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$q_1$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_axrat_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_intern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_axrat_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_extern_cD):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_axrat_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$q_2$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_axrat_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_extern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_axrat_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Componente externo vs Modelo simples - cD Zhao')
	axs.plot(axrat_linspace,axrat_sersic_kde_e_cD(axrat_linspace),color='green',label=f'{names_simples[-1]}')
	axs.axvline(vec_med_axrat_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_axrat_sersic_cD[3]:.3f}$')
	axs.plot(axrat_linspace,axrat_externo_kde_small_cD(axrat_linspace),color='blue',label=f'{names_simples[1]}')
	axs.axvline(vec_med_axrat_externo_cD[1],color='blue',ls='--',label=fr'$\mu = {vec_med_axrat_externo_cD[1]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$q$')
	ks_e_el=ks_2samp(axrat_sersic_e_cD,axrat_externo_cd_small_E)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_externo_EL_E_kde_cD_zhao.png')
	plt.close()

	#MODELO SIMPLES - ELIP cD X MISC

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - cD vs. E/cD & cD/E - Zhao')
	axs.plot(axrat_linspace,axrat_sersic_kde_e_cD(axrat_linspace),color='green',label=f'{names_simples[-1]} cD')
	axs.axvline(vec_med_axrat_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_axrat_sersic_cD[3]:.3f}$')
	axs.plot(axrat_linspace,axrat_sersic_kde_e_misc(axrat_linspace),color='black',label=f'{names_simples[-1]} E/cD & cD/E')
	axs.axvline(np.average(axrat_sersic_e_misc),color='black',ls='--',label=fr'$\mu = {np.average(axrat_sersic_e_misc):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$q$')
	ks_e_cd_misc=ks_2samp(axrat_sersic_e_cD,axrat_sersic_e_misc)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_cd_misc:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/ax_ratio/axrat_simples_E_kde_cD_misc_zhao.png')
	plt.close()

###################################################################################
#BOX
##SÉRSIC -- cD,E(EL),cD(std),E
box_sersic_cd,box_sersic_cd_small,box_sersic_cd_big,box_sersic_e=box_s[cd_lim],box_s[lim_cd_small],box_s[lim_cd_big],box_s[elip_lim]
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL)
box_interno_cd,box_externo_cd,box_interno_cd_big,box_externo_cd_big,box_interno_small,box_externo_small,box_interno_elip,box_externo_elip=box1[cd_lim],box2[cd_lim],box1[lim_cd_big],box2[lim_cd_big],box1[lim_cd_small],box2[lim_cd_small],box1[elip_lim],box2[elip_lim]

ks_box_sersic=ks_calc([box_sersic_cd,box_sersic_cd_small,box_sersic_cd_big,box_sersic_e])
ks_box_intern=ks_calc([box_interno_cd,box_interno_small,box_interno_cd_big,box_interno_elip])
ks_box_extern=ks_calc([box_externo_cd,box_externo_small,box_externo_cd_big,box_externo_elip])

box_kde_entry=np.append(box_s,[box1,box2])
kde_box=kde(box_kde_entry)
box_factor = kde_box.factor
box_linspace=np.linspace(min(box_kde_entry),max(box_kde_entry),3000)

vec_ks_box_sersic=ks_box_sersic[0][1][2],ks_box_sersic[0][1][3],ks_box_sersic[0][2][3]
vec_ks_box_intern=ks_box_intern[0][1][2],ks_box_intern[0][1][3],ks_box_intern[0][2][3]
vec_ks_box_extern=ks_box_extern[0][1][2],ks_box_extern[0][1][3],ks_box_extern[0][2][3]
#
vec_ks_box_2c=ks_box_sersic[0][0][1],ks_box_sersic[0][0][2],ks_box_sersic[0][0][3]
vec_ks_box_intern_2c=ks_box_intern[0][0][1],ks_box_intern[0][0][2],ks_box_intern[0][0][3]
vec_ks_box_extern_2c=ks_box_extern[0][1][2],ks_box_extern[0][0][2],ks_box_extern[0][0][3]
##
vec_med_box_sersic=med_box_sersic_cd,med_box_sersic_cd_small,med_box_sersic_cd_big,med_box_sersic_e=np.average(box_sersic_cd),np.average(box_sersic_cd_small),np.average(box_sersic_cd_big),np.average(box_sersic_e)
vec_med_box_interno=med_box_interno_cd,med_box_interno_small,med_box_interno_cd_big,med_box_interno_elip=np.average(box_interno_cd),np.average(box_interno_small),np.average(box_interno_cd_big),np.average(box_interno_elip)
vec_med_box_externo=med_box_interno_cd,med_box_externo_small,med_box_externo_cd_big,med_box_externo_elip=np.average(box_externo_cd),np.average(box_externo_small),np.average(box_externo_cd_big),np.average(box_externo_elip)

##SÉRSIC -- cD,E(EL),cD(std),E
box_sersic_kde_cd,box_sersic_kde_cd_small,box_sersic_kde_cd_big,box_sersic_kde_e=kde(box_sersic_cd,bw_method=box_factor),kde(box_sersic_cd_small,bw_method=box_factor),kde(box_sersic_cd_big,bw_method=box_factor),kde(box_sersic_e,bw_method=box_factor)
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL)
box_interno_kde_cd,box_externo_kde_cd,box_interno_kde_cd_big,box_externo_kde_cd_big,box_interno_kde_small,box_externo_kde_small,box_interno_kde_elip,box_externo_kde_elip=kde(box_interno_cd,bw_method=box_factor),kde(box_externo_cd,bw_method=box_factor),kde(box_interno_cd_big,bw_method=box_factor),kde(box_externo_cd_big,bw_method=box_factor),kde(box_interno_small,bw_method=box_factor),kde(box_externo_small,bw_method=box_factor),kde(box_interno_elip,bw_method=box_factor),kde(box_externo_elip,bw_method=box_factor)

vec_kde_box_simples=box_sersic_kde_cd,box_sersic_kde_cd_small,box_sersic_kde_cd_big,box_sersic_kde_e
##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
vec_kde_box_intern=box_interno_kde_cd,box_interno_kde_small,box_interno_kde_cd_big,box_interno_kde_elip
vec_kde_box_extern=box_externo_kde_cd,box_externo_kde_small,box_externo_kde_cd_big,box_externo_kde_elip

#IMAGENS BOXINESS
os.makedirs(f'{sample}_stats_observation_desi/test_ks/box',exist_ok=True)

##
plt.figure()
plt.title('Boxiness - Modelos Simples - p_value')
sns.heatmap(ks_box_sersic[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Modelos Simples - D_value')
sns.heatmap(ks_box_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_simples_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Boxiness - Componente interno - p_value')
sns.heatmap(ks_box_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Componente interno - D_value')
sns.heatmap(ks_box_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_interno_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Boxiness - Componente externo - p_value')
sns.heatmap(ks_box_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Componente externo - D_value')
sns.heatmap(ks_box_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Boxiness - Modelos Simples')
for i,dist in enumerate(vec_kde_box_simples):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_box_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_sersic[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_box_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_sersic[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_box_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,True cD) ={vec_ks_box_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Boxiness - Componente interno')
for i,dist in enumerate(vec_kde_box_intern):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_box_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_interno[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_box_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_intern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_box_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,True cD) ={vec_ks_box_intern_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_intern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Boxiness - Componente externo')
for i,dist in enumerate(vec_kde_box_extern):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_box_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_box_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_extern[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_box_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,True cD) ={vec_ks_box_extern_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_extern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Boxiness - Componente externo vs Modelo simples')
axs.plot(box_linspace,box_sersic_kde_e(box_linspace),color='green',label=f'{names_simples[-1]}')
axs.axvline(vec_med_box_sersic[3],color='green',ls='--',label=fr'$\mu = {vec_med_box_sersic[3]:.3f}$')
axs.plot(box_linspace,box_externo_kde_small(box_linspace),color='blue',label=f'{names_simples[1]}')
axs.axvline(vec_med_box_externo[1],color='blue',ls='--',label=fr'$\mu = {vec_med_box_externo[1]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
ks_e_el=ks_2samp(box_sersic_e,box_externo_small)[1]
info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_EL_E_kde.png')
plt.close()
if sample == 'L07':
	##SÉRSIC -- cD,E(EL),cD(std),E
	#CLASSIFICAÇÃO ZHAO PURA
	box_sersic_cd_zhao,box_sersic_e_zhao=box_s[cd_cut],box_s[e_cut]	
	##
	#CLASSIFICAÇÃO NOSSA / SUBGRUPOS MORFOLOGICOS DO ZHAO
	box_sersic_cd_E,box_sersic_cd_cD,box_sersic_cd_misc=box_s[cd_lim & e_cut],box_s[cd_lim & cd_cut],box_s[cd_lim & (ecd_cut | cde_cut)]
	box_sersic_cd_small_E,box_sersic_cd_small_cD,box_sersic_cd_small_misc=box_s[lim_cd_small & e_cut],box_s[lim_cd_small & cd_cut],box_s[lim_cd_small & (ecd_cut | cde_cut)]
	box_sersic_cd_big_E,box_sersic_cd_big_cD,box_sersic_cd_big_misc=box_s[lim_cd_big & e_cut],box_s[lim_cd_big & cd_cut],box_s[lim_cd_big & (ecd_cut | cde_cut)]
	box_sersic_e_E,box_sersic_e_cD,box_sersic_e_misc=box_s[elip_lim & e_cut],box_s[elip_lim & cd_cut],box_s[elip_lim & (ecd_cut | cde_cut)]

	##COMPONENTES -- INTERNO/EXTERNO cD(any),INTERNO/EXTERNO E DAS CLASSIFICAÇÕES ZHAO
	box_interno_cd_zhao,box_externo_cd_zhao,box_interno_e_zhao,box_externo_e_zhao=box1[cd_cut],box2[cd_cut],box1[e_cut],box2[e_cut]
	#
	#COMPONENTES -- INTERNO/EXTERNO NOSSO COM SUBDIVISÃO POR ZHAO 
	box_interno_cd_E,box_interno_cd_cD,box_interno_cd_misc=box1[cd_lim & e_cut],box1[cd_lim & cd_cut],box1[cd_lim & (ecd_cut | cde_cut)]
	box_externo_cd_E,box_externo_cd_cD,box_externo_cd_misc=box2[cd_lim & e_cut],box2[cd_lim & cd_cut],box2[cd_lim & (ecd_cut | cde_cut)]
	#
	box_interno_cd_big_E,box_interno_cd_big_cD,box_interno_cd_big_misc=box1[lim_cd_big & e_cut],box1[lim_cd_big & cd_cut],box1[lim_cd_big & (ecd_cut | cde_cut)]
	box_externo_cd_big_E,box_externo_cd_big_cD,box_externo_cd_big_misc=box2[lim_cd_big & e_cut],box2[lim_cd_big & cd_cut],box2[lim_cd_big & (ecd_cut | cde_cut)]
	#
	box_interno_cd_small_E,box_interno_cd_small_cD,box_interno_cd_small_misc=box1[lim_cd_small & e_cut],box1[lim_cd_small & cd_cut],box1[lim_cd_small & (ecd_cut | cde_cut)]
	box_externo_cd_small_E,box_externo_cd_small_cD,box_externo_cd_small_misc=box2[lim_cd_small & e_cut],box2[lim_cd_small & cd_cut],box2[lim_cd_small & (ecd_cut | cde_cut)]
	#
	box_interno_elip_E,box_interno_elip_cD,box_interno_elip_misc=box1[elip_lim & e_cut],box1[elip_lim & cd_cut],box1[elip_lim & (ecd_cut | cde_cut)]
	box_externo_elip_E,box_externo_elip_cD,box_externo_elip_misc=box2[elip_lim & e_cut],box2[elip_lim & cd_cut],box2[elip_lim & (ecd_cut | cde_cut)]
	
	###
	ks_box_zhao=ks_2samp(box_sersic_cd,box_sersic_e)[1],ks_2samp(box_sersic_cd_zhao,box_sersic_e_zhao)[1]
	ks_box_intern_zhao=ks_2samp(box_interno_cd,box_interno_elip)[1],ks_2samp(box_interno_cd_zhao,box_interno_e_zhao)[1]
	ks_box_extern_zhao=ks_2samp(box_externo_cd,box_externo_elip)[1],ks_2samp(box_externo_cd_zhao,box_externo_e_zhao)[1]
	comp_ks_zhao=np.vstack([ks_box_zhao,ks_box_intern_zhao,ks_box_extern_zhao])
	###
	ks_box_cD_zhao=ks_2samp(box_sersic_cd_cD,box_sersic_e_cD)[1],ks_2samp(box_sersic_cd_big_cD,box_sersic_e_cD)[1],ks_2samp(box_sersic_cd_big_cD,box_sersic_cd_small_cD)[1],ks_2samp(box_sersic_e_cD,box_sersic_cd_small_cD)[1]
	ks_box_intern_cD_zhao=ks_2samp(box_interno_cd_cD,box_interno_elip_cD)[1],ks_2samp(box_interno_cd_big_cD,box_interno_elip_cD)[1],ks_2samp(box_interno_cd_big_cD,box_interno_cd_small_cD)[1],ks_2samp(box_interno_elip_cD,box_interno_cd_small_cD)[1]
	ks_box_extern_cD_zhao=ks_2samp(box_externo_cd_cD,box_externo_elip_cD)[1],ks_2samp(box_externo_cd_big_cD,box_externo_elip_cD)[1],ks_2samp(box_externo_cd_big_cD,box_externo_cd_small_cD)[1],ks_2samp(box_externo_elip_cD,box_externo_cd_small_cD)[1]

	vec_med_box_sersic_cD=med_box_sersic_cd_cD,med_box_sersic_cd_small_cD,med_box_sersic_cd_big_cD,med_box_sersic_e_cD=np.average(box_sersic_cd_cD),np.average(box_sersic_cd_small_cD),np.average(box_sersic_cd_big_cD),np.average(box_sersic_e_cD)
	vec_med_box_interno_cD=med_box_interno_cd_cD,med_box_interno_small_cD,med_box_interno_cd_big_cD,med_box_interno_elip_cD=np.average(box_interno_cd_cD),np.average(box_interno_cd_small_cD),np.average(box_interno_cd_big_cD),np.average(box_interno_elip_cD)
	vec_med_box_externo_cD=med_box_interno_cd_cD,med_box_externo_small_cD,med_box_externo_cd_big_cD,med_box_externo_elip_cD=np.average(box_externo_cd_cD),np.average(box_externo_cd_small_cD),np.average(box_externo_cd_big_cD),np.average(box_externo_elip_cD)

	#IMAGENS INDICE DE SÉRSIC
	##SÉRSIC -- cD,E(EL),cD(std),E - KDE
	vec_kde_box_simples_zhao=box_sersic_kde_cd_cD,box_sersic_kde_cd_small_cD,box_sersic_kde_cd_big_cD,box_sersic_kde_e_cD=kde(box_sersic_cd_cD,bw_method=box_factor),kde(box_sersic_cd_small_cD,bw_method=box_factor),kde(box_sersic_cd_big_cD,bw_method=box_factor),kde(box_sersic_e_cD,bw_method=box_factor)
	box_sersic_kde_e_misc=kde(box_sersic_e_misc,bw_method=box_factor)	
	##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
	vec_kde_box_intern_cD=box_interno_kde_cd_cD,box_interno_kde_small_cD,box_interno_kde_cd_big_cD,box_interno_kde_e_cD=kde(box_interno_cd_cD,bw_method=box_factor),kde(box_interno_cd_small_cD,bw_method=box_factor),kde(box_interno_cd_big_cD,bw_method=box_factor),kde(box_interno_elip_cD,bw_method=box_factor)
	vec_kde_box_extern_cD=box_externo_kde_cd_cD,box_externo_kde_small_cD,box_externo_kde_cd_big_cD,box_externo_kde_e_cD=kde(box_externo_cd_cD,bw_method=box_factor),kde(box_externo_cd_small_cD,bw_method=box_factor),kde(box_externo_cd_big_cD,bw_method=box_factor),kde(box_externo_elip_cD,bw_method=box_factor)

	fig,ax=plt.subplots()
	plt.title('Boxiness Sérsic - Comparação de classificações')
	sns.heatmap(comp_ks_zhao,fmt='.3e',xticklabels=['Nossa','Zhao'],yticklabels=['cD vs. E','cD vs. E','cD vs. E'], annot=True, cmap='Reds')
	ax.xaxis.tick_top()
	ax_right = ax.twinx()
	ax_right.set_ylim(ax.get_ylim())
	ax_right.set_yticks(ax.get_yticks())
	ax_right.set_yticklabels([r'$a_4/a$',r'$(a_4/a)_1$',r'$(a_4/a)_2$'])
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_box_simples_zhao):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_box_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4/a$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_box_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_box_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_box_intern_cD):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_box_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$(a_4/a)_1$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_box_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_intern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_box_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_box_extern_cD):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_box_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$(a_4/a)_2$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_box_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_extern_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_box_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO E(EL) X ELIP

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness Sérsic - Componente externo vs Modelo simples - cD Zhao')
	axs.plot(box_linspace,box_sersic_kde_e_cD(box_linspace),color='green',label=f'{names_simples[-1]}')
	axs.axvline(vec_med_box_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_box_sersic_cD[3]:.3f}$')
	axs.plot(box_linspace,box_externo_kde_small_cD(box_linspace),color='blue',label=f'{names_simples[1]}')
	axs.axvline(vec_med_box_externo_cD[1],color='blue',ls='--',label=fr'$\mu = {vec_med_box_externo_cD[1]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$a_4/a$')
	ks_e_el=ks_2samp(box_sersic_e_cD,box_externo_cd_small_E)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_el:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_externo_EL_E_kde_cD_zhao.png')
	plt.close()

	#MODELO SIMPLES - ELIP cD X MISC

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness Sérsic - cD vs. E/cD & cD/E - Zhao')
	axs.plot(box_linspace,box_sersic_kde_e_cD(box_linspace),color='green',label=f'{names_simples[-1]} cD')
	axs.axvline(vec_med_box_sersic_cD[3],color='green',ls='--',label=fr'$\mu = {vec_med_box_sersic_cD[3]:.3f}$')
	axs.plot(box_linspace,box_sersic_kde_e_misc(box_linspace),color='black',label=f'{names_simples[-1]} E/cD & cD/E')
	axs.axvline(np.average(box_sersic_e_misc),color='black',ls='--',label=fr'$\mu = {np.average(box_sersic_e_misc):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$a_4/a$')
	ks_e_cd_misc=ks_2samp(box_sersic_e_cD,box_sersic_e_misc)[1]
	info_labels = (f'K-S(E,E(EL)) ={ks_e_cd_misc:.3e}')
	fig.text(0.7, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/box/box_simples_E_kde_cD_misc_zhao.png')
	plt.close()
##############################################################
#RFFs/RFFss
##SÉRSIC -- cD,E(EL),cD(std),E
rff_ratio_cd,rff_ratio_cd_small,rff_ratio_cd_big,rff_ratio_e=rff_ratio[cd_lim],rff_ratio[lim_cd_small],rff_ratio[lim_cd_big],rff_ratio[elip_lim]

#RFFss/RFFs
kde_rff_ratio=kde(rff_ratio)
rff_ratio_factor = kde_rff_ratio.factor
rff_ratio_linspace=np.linspace(min(rff_ratio),max(rff_ratio),3000)
##SÉRSIC -- cD,E(EL),cD(std),E
rff_ratio_kde_cd,rff_ratio_kde_cd_small,rff_ratio_kde_cd_big,rff_ratio_kde_e=kde(rff_ratio_cd,bw_method=rff_ratio_factor),kde(rff_ratio_cd_small,bw_method=rff_ratio_factor),kde(rff_ratio_cd_big,bw_method=rff_ratio_factor),kde(rff_ratio_e,bw_method=rff_ratio_factor)
if sample=='L07':
	os.makedirs(f'{sample}_stats_observation_desi/test_ks/rff_ratio',exist_ok=True)
	rff_ratio_cd_cD,rff_ratio_cd_small_cD,rff_ratio_cd_big_cD,rff_ratio_e_cD=rff_ratio[cd_lim & cd_cut],rff_ratio[lim_cd_small & cd_cut],rff_ratio[lim_cd_big & cd_cut],rff_ratio[elip_lim & cd_cut]
	rff_ratio_e_E=rff_ratio[elip_lim & e_cut]
	rff_ratio_e_misc=rff_ratio[elip_lim & (ecd_cut | cde_cut)]
	vec_rff_ratio_kde_zhao=rff_ratio_cd_cD_kde,rff_ratio_cd_small_cD_kde,rff_ratio_cd_big_cD_kde,rff_ratio_e_cD_kde=kde(rff_ratio_cd_cD,bw_method=rff_ratio_factor),kde(rff_ratio_cd_small_cD,bw_method=rff_ratio_factor),kde(rff_ratio_cd_big_cD,bw_method=rff_ratio_factor),kde(rff_ratio_e_cD,bw_method=rff_ratio_factor)
	rff_ratio_e_E_kde=kde(rff_ratio[elip_lim & e_cut],bw_method=rff_ratio_factor)
	rff_ratio_e_misc_kde=kde(rff_ratio[elip_lim & (ecd_cut | cde_cut)],bw_method=rff_ratio_factor)

	vec_rff_ratio_elip_kde_zhao=rff_ratio_e_cD_kde,rff_ratio_e_misc_kde,rff_ratio_e_E_kde

	vec_med_rff_ratio=np.average(rff_ratio_cd_cD),np.average(rff_ratio_cd_small_cD),np.average(rff_ratio_cd_big_cD),np.average(rff_ratio_e_cD)
	vec_med_rff_ratio_elip_zhao=np.average(rff_ratio_e_cD),np.average(rff_ratio_e_misc),np.average(rff_ratio_e_E)
	fig,axs=plt.subplots(1,1)
	for i,dist in enumerate(vec_rff_ratio_kde_zhao):
		axs.plot(rff_ratio_linspace,dist(rff_ratio_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_rff_ratio[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_rff_ratio[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$RFF_{ss}/RFF_s$')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/rff_ratio/rff_ratio_morf_cd.png')
	plt.close()

	fig,axs=plt.subplots(1,1)
	for i,dist in enumerate(vec_rff_ratio_elip_kde_zhao):
		axs.plot(rff_ratio_linspace,dist(rff_ratio_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_rff_ratio_elip_zhao[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_rff_ratio_elip_zhao[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$RFF_{ss}/RFF_s$')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/rff_ratio/rff_ratio_elip_all_morf.png')
	plt.close()
############################################################
#A3
##SÉRSIC -- cD,E(EL),cD(std),E
##média
a3_med_cd,a3_med_cd_small,a3_med_cd_big,a3_med_e=med_a3[cd_lim_photutils],med_a3[cd_lim_photutils_small],med_a3[cd_lim_photutils_big],med_a3[elip_lim_photutils]
##slope
slope_a3_cd,slope_a3_cd_small,slope_a3_cd_big,slope_a3_e=slope_a3[cd_lim_photutils],slope_a3[cd_lim_photutils_small],slope_a3[cd_lim_photutils_big],slope_a3[elip_lim_photutils]
slope_a3_cd_err,slope_a3_cd_small_err,slope_a3_cd_big_err,slope_a3_e_err=slope_a3_err[cd_lim_photutils],slope_a3_err[cd_lim_photutils_small],slope_a3_err[cd_lim_photutils_big],slope_a3_err[elip_lim_photutils]

ks_a3=ks_calc([a3_med_cd,a3_med_cd_small,a3_med_cd_big,a3_med_e])
ks_slope_a3=ks_calc([slope_a3_cd,slope_a3_cd_small,slope_a3_cd_big,slope_a3_e])

vec_ks_a3=ks_a3[0][1][2],ks_a3[0][1][3],ks_a3[0][2][3]
vec_ks_slope_a3=ks_slope_a3[0][1][2],ks_slope_a3[0][1][3],ks_slope_a3[0][2][3]

vec_ks_a3_2c=ks_a3[0][0][1],ks_a3[0][0][2],ks_a3[0][0][3]
vec_ks_slope_a3_2c=ks_slope_a3[0][0][1],ks_slope_a3[0][0][2],ks_slope_a3[0][0][3]

##média
kde_a3=kde(med_a3)
a3_factor = kde_a3.factor
if sample == 'WHL':
	a3_linspace=np.linspace(-0.06,0.06,3000)
elif sample == 'L07':
	a3_linspace=np.linspace(-0.04,0.06,3000)
##slope
kde_slope_a3=kde(slope_a3,weights=slope_a3_err)
slope_a3_factor = kde_slope_a3.factor
if sample == 'WHL':
	slope_a3_linspace=np.linspace(-0.02,0.02,3000)
elif sample == 'L07':
	slope_a3_linspace=np.linspace(-0.01,0.01,3000)

##média
a3_med_kde_cd,a3_med_kde_cd_small,a3_med_kde_cd_big,a3_med_kde_e=kde(a3_med_cd,bw_method=a3_factor),kde(a3_med_cd_small,bw_method=a3_factor),kde(a3_med_cd_big,bw_method=a3_factor),kde(a3_med_e,bw_method=a3_factor)
##slope
slope_a3_kde_cd,slope_a3_kde_cd_small,slope_a3_kde_cd_big,slope_a3_kde_e=kde(slope_a3_cd,bw_method=slope_a3_factor,weights=slope_a3_cd_err),kde(slope_a3_cd_small,bw_method=slope_a3_factor,weights=slope_a3_cd_small_err),kde(slope_a3_cd_big,bw_method=slope_a3_factor,weights=slope_a3_cd_big_err),kde(slope_a3_e,bw_method=slope_a3_factor,weights=slope_a3_e_err)

vec_med_a3=med_a3_cd,med_a3_cd_small,med_a3_cd_big,med_a3_e=np.average(a3_med_cd),np.average(a3_med_cd_small),np.average(a3_med_cd_big),np.average(a3_med_e)
vec_med_slope_a3=med_slope_a3_cd,med_slope_a3_cd_small,med_slope_a3_cd_big,med_slope_a3_e=np.average(slope_a3_cd,weights=slope_a3_cd_err),np.average(slope_a3_cd_small,weights=slope_a3_cd_small_err),np.average(slope_a3_cd_big,weights=slope_a3_cd_big_err),np.average(slope_a3_e,weights=slope_a3_e_err)

vec_kde_a3=a3_med_kde_cd,a3_med_kde_cd_small,a3_med_kde_cd_big,a3_med_kde_e
vec_kde_slope_a3=slope_a3_kde_cd,slope_a3_kde_cd_small,slope_a3_kde_cd_big,slope_a3_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/a3',exist_ok=True)
##
plt.figure()
plt.title('a3 - p_value')
sns.heatmap(ks_a3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_pvalue.png')
plt.close()

plt.figure()
plt.title('a3 - D_value')
sns.heatmap(ks_a3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope a3 - p_value')
sns.heatmap(ks_slope_a3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_pvalue.png')
plt.close()

plt.figure()
plt.title('slope a3 - D_value')
sns.heatmap(ks_slope_a3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('a3 médio')
for i,dist in enumerate(vec_kde_a3):
	axs.plot(a3_linspace,dist(a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_a3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$a_3$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_a3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_a3[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_a3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_a3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_a3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_a3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope a3')
for i,dist in enumerate(vec_kde_slope_a3):
	axs.plot(slope_a3_linspace,dist(slope_a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_a3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha a_3$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_slope_a3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_a3[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_slope_a3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_a3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_a3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_a3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_simples_kde.png')
plt.close()

if sample=='L07':

	#media - E,cD,E/cD DO ZHAO
	a3_med_cd_zhao,a3_med_e_zhao,a3_med_misc_zhao=med_a3[cd_cut_photutils],med_a3[e_cut_photutils],med_a3[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - E,cD,E/cD DO ZHAO
	slope_a3_cd_zhao,slope_a3_e_zhao,slope_a3_misc_zhao=slope_a3[cd_cut_photutils],slope_a3[e_cut_photutils],slope_a3[(ecd_cut_photutils | cde_cut_photutils)]
	#media - SUBGRUPO cD DO ZHAO
	a3_med_cd_cD,a3_med_cd_small_cD,a3_med_cd_big_cD,a3_med_e_cD=med_a3[cd_lim_photutils & cd_cut_photutils],med_a3[cd_lim_photutils_small & cd_cut_photutils],med_a3[cd_lim_photutils_big & cd_cut_photutils],med_a3[elip_lim_photutils & cd_cut_photutils]
	a3_med_e_E,a3_med_e_misc=med_a3[elip_lim_photutils & e_cut_photutils],med_a3[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	slope_a3_cd_cD,slope_a3_cd_small_cD,slope_a3_cd_big_cD,slope_a3_e_cD=slope_a3[cd_lim_photutils & cd_cut_photutils],slope_a3[cd_lim_photutils_small & cd_cut_photutils],slope_a3[cd_lim_photutils_big & cd_cut_photutils],slope_a3[elip_lim_photutils & cd_cut_photutils]
	slope_a3_e_E,slope_a3_e_misc=slope_a3[elip_lim_photutils & e_cut_photutils],slope_a3[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##média - KDE
	a3_med_kde_cd_zhao,a3_med_kde_e_zhao=kde(a3_med_cd_zhao,bw_method=a3_factor),kde(a3_med_e_zhao,bw_method=a3_factor)
	vec_kde_a3_med_cD=a3_med_kde_cd_cD,a3_med_kde_cd_small_cD,a3_med_kde_cd_big_cD,a3_med_kde_e_cD=kde(a3_med_cd_cD,bw_method=a3_factor),kde(a3_med_cd_small_cD,bw_method=a3_factor),kde(a3_med_cd_big_cD,bw_method=a3_factor),kde(a3_med_e_cD,bw_method=a3_factor)
	vec_kde_a3_med_e=a3_med_kde_e_cD,a3_med_kde_e_misc,a3_med_kde_e_E=kde(a3_med_e_cD,bw_method=a3_factor),kde(a3_med_e_misc,bw_method=a3_factor),kde(a3_med_e_E,bw_method=a3_factor)
	##slope - KDE
	slope_a3_kde_cd_zhao,slope_a3_kde_e_zhao=kde(slope_a3_cd_zhao,bw_method=slope_a3_factor),kde(slope_a3_e_zhao,bw_method=slope_a3_factor)
	vec_kde_slope_a3_cD=slope_a3_kde_cd_cD,slope_a3_kde_cd_small_cD,slope_a3_kde_cd_big_cD,slope_a3_kde_e_cD=kde(slope_a3_cd_cD,bw_method=slope_a3_factor),kde(slope_a3_cd_small_cD,bw_method=slope_a3_factor),kde(slope_a3_cd_big_cD,bw_method=slope_a3_factor),kde(slope_a3_e_cD,bw_method=slope_a3_factor)
	vec_kde_slope_a3_e=slope_a3_kde_e_cD,slope_a3_kde_e_misc,slope_a3_kde_e_E=kde(slope_a3_e_cD,bw_method=slope_a3_factor),kde(slope_a3_e_misc,bw_method=slope_a3_factor),kde(slope_a3_e_E,bw_method=slope_a3_factor)

	#MÉDIAS
	vec_med_a3_cD=med_a3_cd_cD,med_a3_cd_small_cD,med_a3_cd_big_cD,med_a3_e_cD=np.average(a3_med_cd_cD),np.average(a3_med_cd_small_cD),np.average(a3_med_cd_big_cD),np.average(a3_med_e_cD)
	vec_med_a3_E=med_a3_e_cD,med_a3_e_misc,med_a3_e_E=np.average(a3_med_e_cD),np.average(a3_med_e_misc),np.average(a3_med_e_E)
	#
	vec_med_slope_a3_cD=med_slope_a3_cd,med_slope_a3_cd_small,med_slope_a3_cd_big,med_slope_a3_e=np.average(slope_a3_cd_cD),np.average(slope_a3_cd_small_cD),np.average(slope_a3_cd_big_cD),np.average(slope_a3_e_cD)
	vec_med_slope_a3_E=med_slope_a3_e_cD,med_slope_a3_e_misc,med_slope_a3_e_E=np.average(slope_a3_e_cD),np.average(slope_a3_e_misc),np.average(slope_a3_e_E)
	#

	ks_med_a3_zhao=ks_2samp(a3_med_cd,a3_med_e)[1],ks_2samp(a3_med_cd_zhao,a3_med_e_zhao)[1]
	ks_med_a3_cD_zhao=ks_2samp(a3_med_cd_cD,a3_med_e_cD)[1],ks_2samp(a3_med_cd_big_cD,a3_med_e_cD)[1],ks_2samp(a3_med_cd_big_cD,a3_med_cd_small_cD)[1],ks_2samp(a3_med_e_cD,a3_med_cd_small_cD)[1]
	ks_med_a3_e_zhao=ks_2samp(a3_med_e_E,a3_med_e_misc)[1],ks_2samp(a3_med_e_cD,a3_med_e_E)[1],ks_2samp(a3_med_e_cD,a3_med_e_misc)[1]

	ks_slope_a3_zhao=ks_2samp(slope_a3_cd,slope_a3_e)[1],ks_2samp(slope_a3_cd_zhao,slope_a3_e_zhao)[1]
	ks_slope_a3_cD_zhao=ks_2samp(slope_a3_cd_cD,slope_a3_e_cD)[1],ks_2samp(slope_a3_cd_big_cD,slope_a3_e_cD)[1],ks_2samp(slope_a3_cd_big_cD,slope_a3_cd_small_cD)[1],ks_2samp(slope_a3_e_cD,slope_a3_cd_small_cD)[1]
	ks_slope_a3_e_zhao=ks_2samp(slope_a3_e_E,slope_a3_e_misc)[1],ks_2samp(slope_a3_e_cD,slope_a3_e_E)[1],ks_2samp(slope_a3_e_cD,slope_a3_e_misc)[1]

	##
	#VALORES MÉDIOS DE a3
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a3 médio - comparação cD vs E - Nossa vs Zhao')
	axs.plot(a3_linspace,a3_med_kde_e_zhao(a3_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(a3_med_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(a3_med_e_zhao):.3e}$')
	axs.plot(a3_linspace,a3_med_kde_cd_zhao(a3_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(a3_med_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(a3_med_cd_zhao):.3e}$')

	axs.plot(a3_linspace,a3_med_kde_e(a3_linspace),color='green',label='E[Kaipper]')
	axs.axvline(np.average(a3_med_e),color='green',ls='--',label=fr'$\mu_E = {np.average(a3_med_e):.3e}$')
	axs.plot(a3_linspace,a3_med_kde_cd(a3_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(a3_med_cd),color='red',ls='--',label=fr'$\mu_cD = {np.average(a3_med_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_med_a3_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_med_a3_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a3 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_a3_med_cD):
		axs.plot(a3_linspace,dist(a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_a3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_med_a3_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_a3_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_med_a3_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_a3_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_med_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a3 médio - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_a3_med_e):
		axs.plot(a3_linspace,dist(a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_a3_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a3_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(E/cD,E) ={ks_med_a3_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_med_a3_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_med_a3_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/a3_simples_kde_e_morf_zhao.png')
	plt.close()

	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a3 - comparação cD vs E - Nossa vs Zhao')
	axs.plot(slope_a3_linspace,slope_a3_kde_e_zhao(slope_a3_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(slope_a3_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(slope_a3_e_zhao):.3e}$')
	axs.plot(slope_a3_linspace,slope_a3_kde_cd_zhao(slope_a3_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(slope_a3_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(slope_a3_cd_zhao):.3e}$')

	axs.plot(slope_a3_linspace,slope_a3_kde_e(slope_a3_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(slope_a3_e),color='red',ls='--',label=fr'$\mu_E = {np.average(slope_a3_e):.3e}$')
	axs.plot(slope_a3_linspace,slope_a3_kde_cd(slope_a3_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(slope_a3_cd),color='red',label=fr'$\mu_cD = {np.average(slope_a3_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_slope_a3_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_slope_a3_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a3 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_a3_cD):
		axs.plot(slope_a3_linspace,dist(slope_a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_a3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_slope_a3_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_a3_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_slope_a3_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_a3_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a3 - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_slope_a3_e):
		axs.plot(slope_a3_linspace,dist(slope_a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_slope_a3_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a3_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(E/cD,E) ={ks_slope_a3_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_slope_a3_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_slope_a3_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a3/slope_a3_kde_e_morf_zhao.png')
	plt.close()

######################################################
#A4
a4_med_cd,a4_med_cd_small,a4_med_cd_big,a4_med_e=med_a4[cd_lim_photutils],med_a4[cd_lim_photutils_small],med_a4[cd_lim_photutils_big],med_a4[elip_lim_photutils]
##slope
slope_a4_cd,slope_a4_cd_small,slope_a4_cd_big,slope_a4_e=slope_a4[cd_lim_photutils],slope_a4[cd_lim_photutils_small],slope_a4[cd_lim_photutils_big],slope_a4[elip_lim_photutils]
slope_a4_cd_err,slope_a4_cd_small_err,slope_a4_cd_big_err,slope_a4_e_err=slope_a4_err[cd_lim_photutils],slope_a4_err[cd_lim_photutils_small],slope_a4_err[cd_lim_photutils_big],slope_a4_err[elip_lim_photutils]

ks_a4=ks_calc([a4_med_cd,a4_med_cd_small,a4_med_cd_big,a4_med_e])
ks_slope_a4=ks_calc([slope_a4_cd,slope_a4_cd_small,slope_a4_cd_big,slope_a4_e])

vec_ks_a4=ks_a4[0][1][2],ks_a4[0][1][3],ks_a4[0][2][3]
vec_ks_slope_a4=ks_slope_a4[0][1][2],ks_slope_a4[0][1][3],ks_slope_a4[0][2][3]

vec_ks_a4_2c=ks_a4[0][0][1],ks_a4[0][0][2],ks_a4[0][0][3]
vec_ks_slope_a4_2c=ks_slope_a4[0][0][1],ks_slope_a4[0][0][2],ks_slope_a4[0][0][3]

##média
kde_a4=kde(med_a4)
a4_factor = kde_a4.factor

if sample == 'WHL':
	a4_linspace=np.linspace(-0.041,max(med_a4),3000)
elif sample == 'L07':
	a4_linspace=np.linspace(-0.03,0.025,3000)
##slope
kde_slope_a4=kde(slope_a4,weights=slope_a4_err)
slope_a4_factor = kde_slope_a4.factor

if sample == 'WHL':
	slope_a4_linspace=np.linspace(-0.015,0.015,3000)
elif sample == 'L07':
	slope_a4_linspace=np.linspace(-0.01,0.01,3000)

##média
a4_med_kde_cd,a4_med_kde_cd_small,a4_med_kde_cd_big,a4_med_kde_e=kde(a4_med_cd,bw_method=a4_factor),kde(a4_med_cd_small,bw_method=a4_factor),kde(a4_med_cd_big,bw_method=a4_factor),kde(a4_med_e,bw_method=a4_factor)
##slope
slope_a4_kde_cd,slope_a4_kde_cd_small,slope_a4_kde_cd_big,slope_a4_kde_e=kde(slope_a4_cd,bw_method=slope_a4_factor,weights=slope_a4_cd_err),kde(slope_a4_cd_small,bw_method=slope_a4_factor,weights=slope_a4_cd_small_err),kde(slope_a4_cd_big,bw_method=slope_a4_factor,weights=slope_a4_cd_big_err),kde(slope_a4_e,bw_method=slope_a4_factor,weights=slope_a4_e_err)

vec_med_a4=med_a4_cd,med_a4_cd_small,med_a4_cd_big,med_a4_e=np.average(a4_med_cd),np.average(a4_med_cd_small),np.average(a4_med_cd_big),np.average(a4_med_e)
vec_med_slope_a4=med_slope_a4_cd,med_slope_a4_cd_small,med_slope_a4_cd_big,med_slope_a4_e=np.average(slope_a4_cd,weights=slope_a4_cd_err),np.average(slope_a4_cd_small,weights=slope_a4_cd_small_err),np.average(slope_a4_cd_big,weights=slope_a4_cd_big_err),np.average(slope_a4_e,weights=slope_a4_e_err)

vec_kde_a4=a4_med_kde_cd,a4_med_kde_cd_small,a4_med_kde_cd_big,a4_med_kde_e
vec_kde_slope_a4=slope_a4_kde_cd,slope_a4_kde_cd_small,slope_a4_kde_cd_big,slope_a4_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/a4',exist_ok=True)

##
plt.figure()
plt.title('a4 - p_value')
sns.heatmap(ks_a4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_pvalue.png')
plt.close()

plt.figure()
plt.title('a4 - D_value')
sns.heatmap(ks_a4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope a4 - p_value')
sns.heatmap(ks_slope_a4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_pvalue.png')
plt.close()

plt.figure()
plt.title('slope a4 - D_value')
sns.heatmap(ks_slope_a4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('a4 médio')
for i,dist in enumerate(vec_kde_a4):
	axs.plot(a4_linspace,dist(a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_a4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a4[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$a_4$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_a4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_a4[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_a4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_a4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_a4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_a4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope a4')
for i,dist in enumerate(vec_kde_slope_a4):
	axs.plot(slope_a4_linspace,dist(slope_a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_a4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a4[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha a_4$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_slope_a4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_a4[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_slope_a4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_a4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_a4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_a4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_simples_kde.png')
plt.close()

if sample=='L07':

	#media - E,cD,E/cD DO ZHAO
	a4_med_cd_zhao,a4_med_e_zhao,a4_med_misc_zhao=med_a4[cd_cut_photutils],med_a4[e_cut_photutils],med_a4[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - E,cD,E/cD DO ZHAO
	slope_a4_cd_zhao,slope_a4_e_zhao,slope_a4_misc_zhao=slope_a4[cd_cut_photutils],slope_a4[e_cut_photutils],slope_a4[(ecd_cut_photutils | cde_cut_photutils)]
	#media - SUBGRUPO cD DO ZHAO
	a4_med_cd_cD,a4_med_cd_small_cD,a4_med_cd_big_cD,a4_med_e_cD=med_a4[cd_lim_photutils & cd_cut_photutils],med_a4[cd_lim_photutils_small & cd_cut_photutils],med_a4[cd_lim_photutils_big & cd_cut_photutils],med_a4[elip_lim_photutils & cd_cut_photutils]
	a4_med_e_E,a4_med_e_misc=med_a4[elip_lim_photutils & e_cut_photutils],med_a4[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	slope_a4_cd_cD,slope_a4_cd_small_cD,slope_a4_cd_big_cD,slope_a4_e_cD=slope_a4[cd_lim_photutils & cd_cut_photutils],slope_a4[cd_lim_photutils_small & cd_cut_photutils],slope_a4[cd_lim_photutils_big & cd_cut_photutils],slope_a4[elip_lim_photutils & cd_cut_photutils]
	slope_a4_e_E,slope_a4_e_misc=slope_a4[elip_lim_photutils & e_cut_photutils],slope_a4[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##média - KDE
	a4_med_kde_cd_zhao,a4_med_kde_e_zhao=kde(a4_med_cd_zhao,bw_method=a4_factor),kde(a4_med_e_zhao,bw_method=a4_factor)
	vec_kde_a4_med_cD=a4_med_kde_cd_cD,a4_med_kde_cd_small_cD,a4_med_kde_cd_big_cD,a4_med_kde_e_cD=kde(a4_med_cd_cD,bw_method=a4_factor),kde(a4_med_cd_small_cD,bw_method=a4_factor),kde(a4_med_cd_big_cD,bw_method=a4_factor),kde(a4_med_e_cD,bw_method=a4_factor)
	vec_kde_a4_med_e=a4_med_kde_e_cD,a4_med_kde_e_misc,a4_med_kde_e_E=kde(a4_med_e_cD,bw_method=a4_factor),kde(a4_med_e_misc,bw_method=a4_factor),kde(a4_med_e_E,bw_method=a4_factor)
	##slope - KDE
	slope_a4_kde_cd_zhao,slope_a4_kde_e_zhao=kde(slope_a4_cd_zhao,bw_method=slope_a4_factor),kde(slope_a4_e_zhao,bw_method=slope_a4_factor)
	vec_kde_slope_a4_cD=slope_a4_kde_cd_cD,slope_a4_kde_cd_small_cD,slope_a4_kde_cd_big_cD,slope_a4_kde_e_cD=kde(slope_a4_cd_cD,bw_method=slope_a4_factor),kde(slope_a4_cd_small_cD,bw_method=slope_a4_factor),kde(slope_a4_cd_big_cD,bw_method=slope_a4_factor),kde(slope_a4_e_cD,bw_method=slope_a4_factor)
	vec_kde_slope_a4_e=slope_a4_kde_e_cD,slope_a4_kde_e_misc,slope_a4_kde_e_E=kde(slope_a4_e_cD,bw_method=slope_a4_factor),kde(slope_a4_e_misc,bw_method=slope_a4_factor),kde(slope_a4_e_E,bw_method=slope_a4_factor)

	#MÉDIAS
	vec_med_a4_cD=med_a4_cd_cD,med_a4_cd_small_cD,med_a4_cd_big_cD,med_a4_e_cD=np.average(a4_med_cd_cD),np.average(a4_med_cd_small_cD),np.average(a4_med_cd_big_cD),np.average(a4_med_e_cD)
	vec_med_a4_E=med_a4_e_cD,med_a4_e_misc,med_a4_e_E=np.average(a4_med_e_cD),np.average(a4_med_e_misc),np.average(a4_med_e_E)
	#
	vec_med_slope_a4_cD=med_slope_a4_cd,med_slope_a4_cd_small,med_slope_a4_cd_big,med_slope_a4_e=np.average(slope_a4_cd_cD),np.average(slope_a4_cd_small_cD),np.average(slope_a4_cd_big_cD),np.average(slope_a4_e_cD)
	vec_med_slope_a4_E=med_slope_a4_e_cD,med_slope_a4_e_misc,med_slope_a4_e_E=np.average(slope_a4_e_cD),np.average(slope_a4_e_misc),np.average(slope_a4_e_E)
	#

	ks_med_a4_zhao=ks_2samp(a4_med_cd,a4_med_e)[1],ks_2samp(a4_med_cd_zhao,a4_med_e_zhao)[1]
	ks_med_a4_cD_zhao=ks_2samp(a4_med_cd_cD,a4_med_e_cD)[1],ks_2samp(a4_med_cd_big_cD,a4_med_e_cD)[1],ks_2samp(a4_med_cd_big_cD,a4_med_cd_small_cD)[1],ks_2samp(a4_med_e_cD,a4_med_cd_small_cD)[1]
	ks_med_a4_e_zhao=ks_2samp(a4_med_e_E,a4_med_e_misc)[1],ks_2samp(a4_med_e_cD,a4_med_e_E)[1],ks_2samp(a4_med_e_cD,a4_med_e_misc)[1]

	ks_slope_a4_zhao=ks_2samp(slope_a4_cd,slope_a4_e)[1],ks_2samp(slope_a4_cd_zhao,slope_a4_e_zhao)[1]
	ks_slope_a4_cD_zhao=ks_2samp(slope_a4_cd_cD,slope_a4_e_cD)[1],ks_2samp(slope_a4_cd_big_cD,slope_a4_e_cD)[1],ks_2samp(slope_a4_cd_big_cD,slope_a4_cd_small_cD)[1],ks_2samp(slope_a4_e_cD,slope_a4_cd_small_cD)[1]
	ks_slope_a4_e_zhao=ks_2samp(slope_a4_e_E,slope_a4_e_misc)[1],ks_2samp(slope_a4_e_cD,slope_a4_e_E)[1],ks_2samp(slope_a4_e_cD,slope_a4_e_misc)[1]

	##
	#VALORES MÉDIOS DE a4
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a4 médio - comparação cD vs E - Nossa vs Zhao')
	axs.plot(a4_linspace,a4_med_kde_e_zhao(a4_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(a4_med_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(a4_med_e_zhao):.3e}$')
	axs.plot(a4_linspace,a4_med_kde_cd_zhao(a4_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(a4_med_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(a4_med_cd_zhao):.3e}$')

	axs.plot(a4_linspace,a4_med_kde_e(a4_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(a4_med_e),color='red',ls='--',label=fr'$\mu_E = {np.average(a4_med_e):.3e}$')
	axs.plot(a4_linspace,a4_med_kde_cd(a4_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(a4_med_cd),color='red',label=fr'$\mu_cD = {np.average(a4_med_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_med_a4_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_med_a4_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a4 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_a4_med_cD):
		axs.plot(a4_linspace,dist(a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_a4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_med_a4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_a4_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_med_a4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_a4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_med_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a4 médio - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_a4_med_e):
		axs.plot(a4_linspace,dist(a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_a4_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a4_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(E/cD,E) ={ks_med_a4_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_med_a4_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_med_a4_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/a4_simples_kde_e_morf_zhao.png')
	plt.close()

	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a4 - comparação cD vs E - Nossa vs Zhao')
	axs.plot(slope_a4_linspace,slope_a4_kde_e_zhao(slope_a4_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(slope_a4_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(slope_a4_e_zhao):.3e}$')
	axs.plot(slope_a4_linspace,slope_a4_kde_cd_zhao(slope_a4_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(slope_a4_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(slope_a4_cd_zhao):.3e}$')

	axs.plot(slope_a4_linspace,slope_a4_kde_e(slope_a4_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(slope_a4_e),color='red',ls='--',label=fr'$\mu_E = {np.average(slope_a4_e):.3e}$')
	axs.plot(slope_a4_linspace,slope_a4_kde_cd(slope_a4_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(slope_a4_cd),color='red',label=fr'$\mu_cD = {np.average(slope_a4_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_slope_a4_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_slope_a4_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a4 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_a4_cD):
		axs.plot(slope_a4_linspace,dist(slope_a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_a4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_slope_a4_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_a4_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_slope_a4_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_a4_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a4 - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_slope_a4_e):
		axs.plot(slope_a4_linspace,dist(slope_a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_slope_a4_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a4_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(E/cD,E) ={ks_slope_a4_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_slope_a4_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_slope_a4_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/a4/slope_a4_kde_e_morf_zhao.png')
	plt.close()
##################################################################
#B3
b3_med_cd,b3_med_cd_small,b3_med_cd_big,b3_med_e=med_b3[cd_lim_photutils],med_b3[cd_lim_photutils_small],med_b3[cd_lim_photutils_big],med_b3[elip_lim_photutils]
##slope
slope_b3_cd,slope_b3_cd_small,slope_b3_cd_big,slope_b3_e=slope_b3[cd_lim_photutils],slope_b3[cd_lim_photutils_small],slope_b3[cd_lim_photutils_big],slope_b3[elip_lim_photutils]
slope_b3_cd_err,slope_b3_cd_small_err,slope_b3_cd_big_err,slope_b3_e_err=slope_b3_err[cd_lim_photutils],slope_b3_err[cd_lim_photutils_small],slope_b3_err[cd_lim_photutils_big],slope_b3_err[elip_lim_photutils]

ks_b3=ks_calc([b3_med_cd,b3_med_cd_small,b3_med_cd_big,b3_med_e])
ks_slope_b3=ks_calc([slope_b3_cd,slope_b3_cd_small,slope_b3_cd_big,slope_b3_e])

vec_ks_b3=ks_b3[0][1][2],ks_b3[0][1][3],ks_b3[0][2][3]
vec_ks_slope_b3=ks_slope_b3[0][1][2],ks_slope_b3[0][1][3],ks_slope_b3[0][2][3]

vec_ks_b3_2c=ks_b3[0][0][1],ks_b3[0][0][2],ks_b3[0][0][3]
vec_ks_slope_b3_2c=ks_slope_b3[0][0][1],ks_slope_b3[0][0][2],ks_slope_b3[0][0][3]

##média
kde_b3=kde(med_b3)
b3_factor = kde_b3.factor
if sample == 'WHL':
	b3_linspace=np.linspace(-0.062,0.062,3000)
elif sample == 'L07':
	b3_linspace=np.linspace(-0.062,0.052,3000)
##slope
kde_slope_b3=kde(slope_b3,weights=slope_b3_err)
slope_b3_factor = kde_slope_b3.factor
if sample == 'WHL':
	slope_b3_linspace=np.linspace(-0.02,0.02,3000)
elif sample == 'L07':
	slope_b3_linspace=np.linspace(-0.01,0.01,3000)

##média
b3_med_kde_cd,b3_med_kde_cd_small,b3_med_kde_cd_big,b3_med_kde_e=kde(b3_med_cd,bw_method=b3_factor),kde(b3_med_cd_small,bw_method=b3_factor),kde(b3_med_cd_big,bw_method=b3_factor),kde(b3_med_e,bw_method=b3_factor)
##slope
slope_b3_kde_cd,slope_b3_kde_cd_small,slope_b3_kde_cd_big,slope_b3_kde_e=kde(slope_b3_cd,bw_method=slope_b3_factor,weights=slope_b3_cd_err),kde(slope_b3_cd_small,bw_method=slope_b3_factor,weights=slope_b3_cd_small_err),kde(slope_b3_cd_big,bw_method=slope_b3_factor,weights=slope_b3_cd_big_err),kde(slope_b3_e,bw_method=slope_b3_factor,weights=slope_b3_e_err)

vec_med_b3=med_b3_cd,med_b3_cd_small,med_b3_cd_big,med_b3_e=np.average(b3_med_cd),np.average(b3_med_cd_small),np.average(b3_med_cd_big),np.average(b3_med_e)
vec_med_slope_b3=med_slope_b3_cd,med_slope_b3_cd_small,med_slope_b3_cd_big,med_slope_b3_e=np.average(slope_b3_cd,weights=slope_b3_cd_err),np.average(slope_b3_cd_small,weights=slope_b3_cd_small_err),np.average(slope_b3_cd_big,weights=slope_b3_cd_big_err),np.average(slope_b3_e,weights=slope_b3_e_err)

vec_kde_b3=b3_med_kde_cd,b3_med_kde_cd_small,b3_med_kde_cd_big,b3_med_kde_e
vec_kde_slope_b3=slope_b3_kde_cd,slope_b3_kde_cd_small,slope_b3_kde_cd_big,slope_b3_kde_e


os.makedirs(f'{sample}_stats_observation_desi/test_ks/b3',exist_ok=True)

##
plt.figure()
plt.title('b3 - p_value')
sns.heatmap(ks_b3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_pvalue.png')
plt.close()

plt.figure()
plt.title('b3 - D_value')
sns.heatmap(ks_b3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope b3 - p_value')
sns.heatmap(ks_slope_b3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_pvalue.png')
plt.close()

plt.figure()
plt.title('slope b3 - D_value')
sns.heatmap(ks_slope_b3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('b3 médio')
for i,dist in enumerate(vec_kde_b3):
	axs.plot(b3_linspace,dist(b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_b3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$b_3$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_b3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_b3[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_b3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_b3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_b3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_b3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope b3')
for i,dist in enumerate(vec_kde_slope_b3):
	axs.plot(slope_b3_linspace,dist(slope_b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_b3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha b_3$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_slope_b3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_b3[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_slope_b3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_b3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_b3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_b3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_simples_kde.png')
plt.close()
if sample=='L07':

	#media - E,cD,E/cD DO ZHAO
	b3_med_cd_zhao,b3_med_e_zhao,b3_med_misc_zhao=med_b3[cd_cut_photutils],med_b3[e_cut_photutils],med_b3[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - E,cD,E/cD DO ZHAO
	slope_b3_cd_zhao,slope_b3_e_zhao,slope_b3_misc_zhao=slope_b3[cd_cut_photutils],slope_b3[e_cut_photutils],slope_b3[(ecd_cut_photutils | cde_cut_photutils)]
	#media - SUBGRUPO cD DO ZHAO
	b3_med_cd_cD,b3_med_cd_small_cD,b3_med_cd_big_cD,b3_med_e_cD=med_b3[cd_lim_photutils & cd_cut_photutils],med_b3[cd_lim_photutils_small & cd_cut_photutils],med_b3[cd_lim_photutils_big & cd_cut_photutils],med_b3[elip_lim_photutils & cd_cut_photutils]
	b3_med_e_E,b3_med_e_misc=med_b3[elip_lim_photutils & e_cut_photutils],med_b3[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	slope_b3_cd_cD,slope_b3_cd_small_cD,slope_b3_cd_big_cD,slope_b3_e_cD=slope_b3[cd_lim_photutils & cd_cut_photutils],slope_b3[cd_lim_photutils_small & cd_cut_photutils],slope_b3[cd_lim_photutils_big & cd_cut_photutils],slope_b3[elip_lim_photutils & cd_cut_photutils]
	slope_b3_e_E,slope_b3_e_misc=slope_b3[elip_lim_photutils & e_cut_photutils],slope_b3[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##média - KDE
	b3_med_kde_cd_zhao,b3_med_kde_e_zhao=kde(b3_med_cd_zhao,bw_method=b3_factor),kde(b3_med_e_zhao,bw_method=b3_factor)
	vec_kde_b3_med_cD=b3_med_kde_cd_cD,b3_med_kde_cd_small_cD,b3_med_kde_cd_big_cD,b3_med_kde_e_cD=kde(b3_med_cd_cD,bw_method=b3_factor),kde(b3_med_cd_small_cD,bw_method=b3_factor),kde(b3_med_cd_big_cD,bw_method=b3_factor),kde(b3_med_e_cD,bw_method=b3_factor)
	vec_kde_b3_med_e=b3_med_kde_e_cD,b3_med_kde_e_misc,b3_med_kde_e_E=kde(b3_med_e_cD,bw_method=b3_factor),kde(b3_med_e_misc,bw_method=b3_factor),kde(b3_med_e_E,bw_method=b3_factor)
	##slope - KDE
	slope_b3_kde_cd_zhao,slope_b3_kde_e_zhao=kde(slope_b3_cd_zhao,bw_method=slope_b3_factor),kde(slope_b3_e_zhao,bw_method=slope_b3_factor)
	vec_kde_slope_b3_cD=slope_b3_kde_cd_cD,slope_b3_kde_cd_small_cD,slope_b3_kde_cd_big_cD,slope_b3_kde_e_cD=kde(slope_b3_cd_cD,bw_method=slope_b3_factor),kde(slope_b3_cd_small_cD,bw_method=slope_b3_factor),kde(slope_b3_cd_big_cD,bw_method=slope_b3_factor),kde(slope_b3_e_cD,bw_method=slope_b3_factor)
	vec_kde_slope_b3_e=slope_b3_kde_e_cD,slope_b3_kde_e_misc,slope_b3_kde_e_E=kde(slope_b3_e_cD,bw_method=slope_b3_factor),kde(slope_b3_e_misc,bw_method=slope_b3_factor),kde(slope_b3_e_E,bw_method=slope_b3_factor)

	#MÉDIAS
	vec_med_b3_cD=med_b3_cd_cD,med_b3_cd_small_cD,med_b3_cd_big_cD,med_b3_e_cD=np.average(b3_med_cd_cD),np.average(b3_med_cd_small_cD),np.average(b3_med_cd_big_cD),np.average(b3_med_e_cD)
	vec_med_b3_E=med_b3_e_cD,med_b3_e_misc,med_b3_e_E=np.average(b3_med_e_cD),np.average(b3_med_e_misc),np.average(b3_med_e_E)
	#
	vec_med_slope_b3_cD=med_slope_b3_cd,med_slope_b3_cd_small,med_slope_b3_cd_big,med_slope_b3_e=np.average(slope_b3_cd_cD),np.average(slope_b3_cd_small_cD),np.average(slope_b3_cd_big_cD),np.average(slope_b3_e_cD)
	vec_med_slope_b3_E=med_slope_b3_e_cD,med_slope_b3_e_misc,med_slope_b3_e_E=np.average(slope_b3_e_cD),np.average(slope_b3_e_misc),np.average(slope_b3_e_E)
	#

	ks_med_b3_zhao=ks_2samp(b3_med_cd,b3_med_e)[1],ks_2samp(b3_med_cd_zhao,b3_med_e_zhao)[1]
	ks_med_b3_cD_zhao=ks_2samp(b3_med_cd_cD,b3_med_e_cD)[1],ks_2samp(b3_med_cd_big_cD,b3_med_e_cD)[1],ks_2samp(b3_med_cd_big_cD,b3_med_cd_small_cD)[1],ks_2samp(b3_med_e_cD,b3_med_cd_small_cD)[1]
	ks_med_b3_e_zhao=ks_2samp(b3_med_e_E,b3_med_e_misc)[1],ks_2samp(b3_med_e_cD,b3_med_e_E)[1],ks_2samp(b3_med_e_cD,b3_med_e_misc)[1]

	ks_slope_b3_zhao=ks_2samp(slope_b3_cd,slope_b3_e)[1],ks_2samp(slope_b3_cd_zhao,slope_b3_e_zhao)[1]
	ks_slope_b3_cD_zhao=ks_2samp(slope_b3_cd_cD,slope_b3_e_cD)[1],ks_2samp(slope_b3_cd_big_cD,slope_b3_e_cD)[1],ks_2samp(slope_b3_cd_big_cD,slope_b3_cd_small_cD)[1],ks_2samp(slope_b3_e_cD,slope_b3_cd_small_cD)[1]
	ks_slope_b3_e_zhao=ks_2samp(slope_b3_e_E,slope_b3_e_misc)[1],ks_2samp(slope_b3_e_cD,slope_b3_e_E)[1],ks_2samp(slope_b3_e_cD,slope_b3_e_misc)[1]

	##
	#VALORES MÉDIOS DE b3
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b3 médio - comparação cD vs E - Nossa vs Zhao')
	axs.plot(b3_linspace,b3_med_kde_e_zhao(b3_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(b3_med_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(b3_med_e_zhao):.3e}$')
	axs.plot(b3_linspace,b3_med_kde_cd_zhao(b3_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(b3_med_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(b3_med_cd_zhao):.3e}$')

	axs.plot(b3_linspace,b3_med_kde_e(b3_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(b3_med_e),color='red',ls='--',label=fr'$\mu_E = {np.average(b3_med_e):.3e}$')
	axs.plot(b3_linspace,b3_med_kde_cd(b3_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(b3_med_cd),color='red',label=fr'$\mu_cD = {np.average(b3_med_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_med_b3_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_med_b3_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b3 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_b3_med_cD):
		axs.plot(b3_linspace,dist(b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_b3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_med_b3_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_b3_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_med_b3_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_b3_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_med_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b3 médio - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_b3_med_e):
		axs.plot(b3_linspace,dist(b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_b3_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b3_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(E/cD,E) ={ks_med_b3_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_med_b3_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_med_b3_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/b3_simples_kde_e_morf_zhao.png')
	plt.close()

	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b3 - comparação cD vs E - Nossa vs Zhao')
	axs.plot(slope_b3_linspace,slope_b3_kde_e_zhao(slope_b3_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(slope_b3_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(slope_b3_e_zhao):.3e}$')
	axs.plot(slope_b3_linspace,slope_b3_kde_cd_zhao(slope_b3_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(slope_b3_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(slope_b3_cd_zhao):.3e}$')

	axs.plot(slope_b3_linspace,slope_b3_kde_e(slope_b3_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(slope_b3_e),color='red',ls='--',label=fr'$\mu_E = {np.average(slope_b3_e):.3e}$')
	axs.plot(slope_b3_linspace,slope_b3_kde_cd(slope_b3_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(slope_b3_cd),color='red',label=fr'$\mu_cD = {np.average(slope_b3_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_slope_b3_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_slope_b3_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b3 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_b3_cD):
		axs.plot(slope_b3_linspace,dist(slope_b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_b3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_slope_b3_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_b3_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_slope_b3_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_b3_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b3 - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_slope_b3_e):
		axs.plot(slope_b3_linspace,dist(slope_b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_slope_b3_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b3_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(E/cD,E) ={ks_slope_b3_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_slope_b3_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_slope_b3_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b3/slope_b3_kde_e_morf_zhao.png')
	plt.close()

#################################################################################
#B4
b4_med_cd,b4_med_cd_small,b4_med_cd_big,b4_med_e=med_b4[cd_lim_photutils],med_b4[cd_lim_photutils_small],med_b4[cd_lim_photutils_big],med_b4[elip_lim_photutils]
##slope
slope_b4_cd,slope_b4_cd_small,slope_b4_cd_big,slope_b4_e=slope_b4[cd_lim_photutils],slope_b4[cd_lim_photutils_small],slope_b4[cd_lim_photutils_big],slope_b4[elip_lim_photutils]
slope_b4_cd_err,slope_b4_cd_small_err,slope_b4_cd_big_err,slope_b4_e_err=slope_b4_err[cd_lim_photutils],slope_b4_err[cd_lim_photutils_small],slope_b4_err[cd_lim_photutils_big],slope_b4_err[elip_lim_photutils]

ks_b4=ks_calc([b4_med_cd,b4_med_cd_small,b4_med_cd_big,b4_med_e])
ks_slope_b4=ks_calc([slope_b4_cd,slope_b4_cd_small,slope_b4_cd_big,slope_b4_e])

vec_ks_b4=ks_b4[0][1][2],ks_b4[0][1][3],ks_b4[0][2][3]
vec_ks_slope_b4=ks_slope_b4[0][1][2],ks_slope_b4[0][1][3],ks_slope_b4[0][2][3]

vec_ks_b4_2c=ks_b4[0][0][1],ks_b4[0][0][2],ks_b4[0][0][3]
vec_ks_slope_b4_2c=ks_slope_b4[0][0][1],ks_slope_b4[0][0][2],ks_slope_b4[0][0][3]

##média
kde_b4=kde(med_b4)
b4_factor = kde_b4.factor
if sample == 'WHL':
	b4_linspace=np.linspace(-0.03,0.03,3000)
elif sample == 'L07':
	b4_linspace=np.linspace(-0.03,0.03,3000)
##slope
kde_slope_b4=kde(slope_b4,weights=slope_b4_err)
slope_b4_factor = kde_slope_b4.factor
if sample == 'WHL':
	slope_b4_linspace=np.linspace(-0.022,max(slope_b4),3000)
elif sample == 'L07':
	slope_b4_linspace=np.linspace(-0.01,0.01,3000)

##média
b4_med_kde_cd,b4_med_kde_cd_small,b4_med_kde_cd_big,b4_med_kde_e=kde(b4_med_cd,bw_method=b4_factor),kde(b4_med_cd_small,bw_method=b4_factor),kde(b4_med_cd_big,bw_method=b4_factor),kde(b4_med_e,bw_method=b4_factor)
##slope
slope_b4_kde_cd,slope_b4_kde_cd_small,slope_b4_kde_cd_big,slope_b4_kde_e=kde(slope_b4_cd,bw_method=slope_b4_factor,weights=slope_b4_cd_err),kde(slope_b4_cd_small,bw_method=slope_b4_factor,weights=slope_b4_cd_small_err),kde(slope_b4_cd_big,bw_method=slope_b4_factor,weights=slope_b4_cd_big_err),kde(slope_b4_e,bw_method=slope_b4_factor,weights=slope_b4_e_err)

vec_med_b4=med_b4_cd,med_b4_cd_small,med_b4_cd_big,med_b4_e=np.average(b4_med_cd),np.average(b4_med_cd_small),np.average(b4_med_cd_big),np.average(b4_med_e)
vec_med_slope_b4=med_slope_b4_cd,med_slope_b4_cd_small,med_slope_b4_cd_big,med_slope_b4_e=np.average(slope_b4_cd,weights=slope_b4_cd_err),np.average(slope_b4_cd_small,weights=slope_b4_cd_small_err),np.average(slope_b4_cd_big,weights=slope_b4_cd_big_err),np.average(slope_b4_e,weights=slope_b4_e_err)

vec_kde_b4=b4_med_kde_cd,b4_med_kde_cd_small,b4_med_kde_cd_big,b4_med_kde_e
vec_kde_slope_b4=slope_b4_kde_cd,slope_b4_kde_cd_small,slope_b4_kde_cd_big,slope_b4_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/b4',exist_ok=True)
##
plt.figure()
plt.title('b4 - p_value')
sns.heatmap(ks_b4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_pvalue.png')
plt.close()

plt.figure()
plt.title('b4 - D_value')
sns.heatmap(ks_b4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope b4 - p_value')
sns.heatmap(ks_slope_b4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_pvalue.png')
plt.close()

plt.figure()
plt.title('slope b4 - D_value')
sns.heatmap(ks_slope_b4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_dvalue.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('b4 médio')
for i,dist in enumerate(vec_kde_b4):
	axs.plot(b4_linspace,dist(b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_b4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b4[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$b_4$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_b4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_b4[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_b4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_b4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_b4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_b4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope b4')
for i,dist in enumerate(vec_kde_slope_b4):
	axs.plot(slope_b4_linspace,dist(slope_b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_b4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b4[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\alpha b_4$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_slope_b4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_b4[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_slope_b4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_b4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_b4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_b4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_simples_kde.png')
plt.close()

if sample=='L07':

	#media - E,cD,E/cD DO ZHAO
	b4_med_cd_zhao,b4_med_e_zhao,b4_med_misc_zhao=med_b4[cd_cut_photutils],med_b4[e_cut_photutils],med_b4[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - E,cD,E/cD DO ZHAO
	slope_b4_cd_zhao,slope_b4_e_zhao,slope_b4_misc_zhao=slope_b4[cd_cut_photutils],slope_b4[e_cut_photutils],slope_b4[(ecd_cut_photutils | cde_cut_photutils)]
	#media - SUBGRUPO cD DO ZHAO
	b4_med_cd_cD,b4_med_cd_small_cD,b4_med_cd_big_cD,b4_med_e_cD=med_b4[cd_lim_photutils & cd_cut_photutils],med_b4[cd_lim_photutils_small & cd_cut_photutils],med_b4[cd_lim_photutils_big & cd_cut_photutils],med_b4[elip_lim_photutils & cd_cut_photutils]
	b4_med_e_E,b4_med_e_misc=med_b4[elip_lim_photutils & e_cut_photutils],med_b4[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	slope_b4_cd_cD,slope_b4_cd_small_cD,slope_b4_cd_big_cD,slope_b4_e_cD=slope_b4[cd_lim_photutils & cd_cut_photutils],slope_b4[cd_lim_photutils_small & cd_cut_photutils],slope_b4[cd_lim_photutils_big & cd_cut_photutils],slope_b4[elip_lim_photutils & cd_cut_photutils]
	slope_b4_e_E,slope_b4_e_misc=slope_b4[elip_lim_photutils & e_cut_photutils],slope_b4[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##média - KDE
	b4_med_kde_cd_zhao,b4_med_kde_e_zhao=kde(b4_med_cd_zhao,bw_method=b4_factor),kde(b4_med_e_zhao,bw_method=b4_factor)
	vec_kde_b4_med_cD=b4_med_kde_cd_cD,b4_med_kde_cd_small_cD,b4_med_kde_cd_big_cD,b4_med_kde_e_cD=kde(b4_med_cd_cD,bw_method=b4_factor),kde(b4_med_cd_small_cD,bw_method=b4_factor),kde(b4_med_cd_big_cD,bw_method=b4_factor),kde(b4_med_e_cD,bw_method=b4_factor)
	vec_kde_b4_med_e=b4_med_kde_e_cD,b4_med_kde_e_misc,b4_med_kde_e_E=kde(b4_med_e_cD,bw_method=b4_factor),kde(b4_med_e_misc,bw_method=b4_factor),kde(b4_med_e_E,bw_method=b4_factor)
	##slope - KDE
	slope_b4_kde_cd_zhao,slope_b4_kde_e_zhao=kde(slope_b4_cd_zhao,bw_method=slope_b4_factor),kde(slope_b4_e_zhao,bw_method=slope_b4_factor)
	vec_kde_slope_b4_cD=slope_b4_kde_cd_cD,slope_b4_kde_cd_small_cD,slope_b4_kde_cd_big_cD,slope_b4_kde_e_cD=kde(slope_b4_cd_cD,bw_method=slope_b4_factor),kde(slope_b4_cd_small_cD,bw_method=slope_b4_factor),kde(slope_b4_cd_big_cD,bw_method=slope_b4_factor),kde(slope_b4_e_cD,bw_method=slope_b4_factor)
	vec_kde_slope_b4_e=slope_b4_kde_e_cD,slope_b4_kde_e_misc,slope_b4_kde_e_E=kde(slope_b4_e_cD,bw_method=slope_b4_factor),kde(slope_b4_e_misc,bw_method=slope_b4_factor),kde(slope_b4_e_E,bw_method=slope_b4_factor)

	#MÉDIAS
	vec_med_b4_cD=med_b4_cd_cD,med_b4_cd_small_cD,med_b4_cd_big_cD,med_b4_e_cD=np.average(b4_med_cd_cD),np.average(b4_med_cd_small_cD),np.average(b4_med_cd_big_cD),np.average(b4_med_e_cD)
	vec_med_b4_E=med_b4_e_cD,med_b4_e_misc,med_b4_e_E=np.average(b4_med_e_cD),np.average(b4_med_e_misc),np.average(b4_med_e_E)
	#
	vec_med_slope_b4_cD=med_slope_b4_cd,med_slope_b4_cd_small,med_slope_b4_cd_big,med_slope_b4_e=np.average(slope_b4_cd_cD),np.average(slope_b4_cd_small_cD),np.average(slope_b4_cd_big_cD),np.average(slope_b4_e_cD)
	vec_med_slope_b4_E=med_slope_b4_e_cD,med_slope_b4_e_misc,med_slope_b4_e_E=np.average(slope_b4_e_cD),np.average(slope_b4_e_misc),np.average(slope_b4_e_E)
	#

	ks_med_b4_zhao=ks_2samp(b4_med_cd,b4_med_e)[1],ks_2samp(b4_med_cd_zhao,b4_med_e_zhao)[1]
	ks_med_b4_cD_zhao=ks_2samp(b4_med_cd_cD,b4_med_e_cD)[1],ks_2samp(b4_med_cd_big_cD,b4_med_e_cD)[1],ks_2samp(b4_med_cd_big_cD,b4_med_cd_small_cD)[1],ks_2samp(b4_med_e_cD,b4_med_cd_small_cD)[1]
	ks_med_b4_e_zhao=ks_2samp(b4_med_e_E,b4_med_e_misc)[1],ks_2samp(b4_med_e_cD,b4_med_e_E)[1],ks_2samp(b4_med_e_cD,b4_med_e_misc)[1]

	ks_slope_b4_zhao=ks_2samp(slope_b4_cd,slope_b4_e)[1],ks_2samp(slope_b4_cd_zhao,slope_b4_e_zhao)[1]
	ks_slope_b4_cD_zhao=ks_2samp(slope_b4_cd_cD,slope_b4_e_cD)[1],ks_2samp(slope_b4_cd_big_cD,slope_b4_e_cD)[1],ks_2samp(slope_b4_cd_big_cD,slope_b4_cd_small_cD)[1],ks_2samp(slope_b4_e_cD,slope_b4_cd_small_cD)[1]
	ks_slope_b4_e_zhao=ks_2samp(slope_b4_e_E,slope_b4_e_misc)[1],ks_2samp(slope_b4_e_cD,slope_b4_e_E)[1],ks_2samp(slope_b4_e_cD,slope_b4_e_misc)[1]

	#############
	#VALORES MÉDIOS DE b4
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b4 médio - comparação cD vs E - Nossa vs Zhao')
	axs.plot(b4_linspace,b4_med_kde_e_zhao(b4_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(b4_med_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(b4_med_e_zhao):.3e}$')
	axs.plot(b4_linspace,b4_med_kde_cd_zhao(b4_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(b4_med_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(b4_med_cd_zhao):.3e}$')

	axs.plot(b4_linspace,b4_med_kde_e(b4_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(b4_med_e),color='red',ls='--',label=fr'$\mu_E = {np.average(b4_med_e):.3e}$')
	axs.plot(b4_linspace,b4_med_kde_cd(b4_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(b4_med_cd),color='red',label=fr'$\mu_cD = {np.average(b4_med_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_med_b4_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_med_b4_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_med_kde_comp_zhao.png')
	plt.close()
	######
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b4 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_b4_med_cD):
		axs.plot(b4_linspace,dist(b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_b4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_med_b4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_b4_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_med_b4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_b4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_med_kde_cD_zhao.png')
	plt.close()
	########
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b4 médio - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_b4_med_e):
		axs.plot(b4_linspace,dist(b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_b4_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b4_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(E/cD,E) ={ks_med_b4_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_med_b4_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_med_b4_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/b4_simples_kde_e_morf_zhao.png')
	plt.close()
	########################
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b4 - comparação cD vs E - Nossa vs Zhao')
	axs.plot(slope_b4_linspace,slope_b4_kde_e_zhao(slope_b4_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(slope_b4_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(slope_b4_e_zhao):.3e}$')
	axs.plot(slope_b4_linspace,slope_b4_kde_cd_zhao(slope_b4_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(slope_b4_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(slope_b4_cd_zhao):.3e}$')

	axs.plot(slope_b4_linspace,slope_b4_kde_e(slope_b4_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(slope_b4_e),color='red',ls='--',label=fr'$\mu_E = {np.average(slope_b4_e):.3f}$')
	axs.plot(slope_b4_linspace,slope_b4_kde_cd(slope_b4_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(slope_b4_cd),color='red',label=fr'$\mu_cD = {np.average(slope_b4_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_slope_b4_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_slope_b4_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b4 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_b4_cD):
		axs.plot(slope_b4_linspace,dist(slope_b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_b4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_slope_b4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_slope_b4_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_slope_b4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_slope_b4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b4 - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_slope_b4_e):
		axs.plot(slope_b4_linspace,dist(slope_b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_slope_b4_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b4_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(E/cD,E) ={ks_slope_b4_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_slope_b4_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_slope_b4_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/b4/slope_b4_kde_e_morf_zhao.png')
	plt.close()
######################################################################
#G-R
##SÉRSIC -- cD,E(EL),cD(std),E
slope_gr_cd,slope_gr_cd_small,slope_gr_cd_big,slope_gr_e=slope_gr[cd_lim_photutils],slope_gr[cd_lim_photutils_small],slope_gr[cd_lim_photutils_big],slope_gr[elip_lim_photutils]
slope_gr_cd_err,slope_gr_cd_small_err,slope_gr_cd_big_err,slope_gr_e_err=slope_gr_err[cd_lim_photutils],slope_gr_err[cd_lim_photutils_small],slope_gr_err[cd_lim_photutils_big],slope_gr_err[elip_lim_photutils]

ks_slope_gr=ks_calc([slope_gr_cd,slope_gr_cd_small,slope_gr_cd_big,slope_gr_e])

vec_ks_slope_gr=ks_slope_gr[0][1][2],ks_slope_gr[0][1][3],ks_slope_gr[0][2][3]
vec_ks_slope_gr_2c=ks_slope_gr[0][0][1],ks_slope_gr[0][0][2],ks_slope_gr[0][0][3]

kde_slope_gr=kde(slope_gr,weights=slope_gr_err)
slope_gr_factor = kde_slope_gr.factor
slope_gr_linspace=np.linspace(-1.0,1.0,3000)
slope_gr_kde_cd,slope_gr_kde_cd_small,slope_gr_kde_cd_big,slope_gr_kde_e=kde(slope_gr_cd,bw_method=slope_gr_factor,weights=slope_gr_cd_err),kde(slope_gr_cd_small,bw_method=slope_gr_factor,weights=slope_gr_cd_small_err),kde(slope_gr_cd_big,bw_method=slope_gr_factor,weights=slope_gr_cd_big_err),kde(slope_gr_e,bw_method=slope_gr_factor,weights=slope_gr_e_err)

vec_med_slope_gr=med_slope_gr_cd,med_slope_gr_cd_small,med_slope_gr_cd_big,med_slope_gr_e=np.average(slope_gr_cd,weights=slope_gr_cd_err),np.average(slope_gr_cd_small,weights=slope_gr_cd_small_err),np.average(slope_gr_cd_big,weights=slope_gr_cd_big_err),np.average(slope_gr_e,weights=slope_gr_e_err)
vec_kde_slope_gr=slope_gr_kde_cd,slope_gr_kde_cd_small,slope_gr_kde_cd_big,slope_gr_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/test_gr',exist_ok=True)

plt.figure()
plt.title('slope gr - Modelos Simples - p_value')
sns.heatmap(ks_slope_gr[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_pvalue.png')
plt.close()

plt.figure()
plt.title('slope gr - Modelos Simples - D_value')
sns.heatmap(ks_slope_gr[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_dvalue.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope g-r')
for i,dist in enumerate(vec_kde_slope_gr):
	axs.plot(slope_gr_linspace,dist(slope_gr_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_gr[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_gr[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\alpha g-r$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_slope_gr[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_gr[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_slope_gr[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_gr_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_gr_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_gr_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_simples_kde.png')
plt.close()

# 
if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	slope_gr_cd_zhao,slope_gr_e_zhao,slope_gr_misc_zhao=slope_gr[cd_cut_photutils],slope_gr[e_cut_photutils],slope_gr[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	slope_gr_cd_cD,slope_gr_cd_small_cD,slope_gr_cd_big_cD,slope_gr_e_cD=slope_gr[cd_lim_photutils & cd_cut_photutils],slope_gr[cd_lim_photutils_small & cd_cut_photutils],slope_gr[cd_lim_photutils_big & cd_cut_photutils],slope_gr[elip_lim_photutils & cd_cut_photutils]
	slope_gr_e_E,slope_gr_e_misc=slope_gr[elip_lim_photutils & e_cut_photutils],slope_gr[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##slope - KDE
	slope_gr_kde_cd_zhao,slope_gr_kde_e_zhao=kde(slope_gr_cd_zhao,bw_method=slope_gr_factor),kde(slope_gr_e_zhao,bw_method=slope_gr_factor)
	vec_kde_slope_gr_cD=slope_gr_kde_cd_cD,slope_gr_kde_cd_small_cD,slope_gr_kde_cd_big_cD,slope_gr_kde_e_cD=kde(slope_gr_cd_cD,bw_method=slope_gr_factor),kde(slope_gr_cd_small_cD,bw_method=slope_gr_factor),kde(slope_gr_cd_big_cD,bw_method=slope_gr_factor),kde(slope_gr_e_cD,bw_method=slope_gr_factor)
	vec_kde_slope_gr_e=slope_gr_kde_e_cD,slope_gr_kde_e_misc,slope_gr_kde_e_E=kde(slope_gr_e_cD,bw_method=slope_gr_factor),kde(slope_gr_e_misc,bw_method=slope_gr_factor),kde(slope_gr_e_E,bw_method=slope_gr_factor)

	#MÉDIAS
	vec_med_slope_gr_cD=med_slope_gr_cd,med_slope_gr_cd_small,med_slope_gr_cd_big,med_slope_gr_e=np.average(slope_gr_cd_cD),np.average(slope_gr_cd_small_cD),np.average(slope_gr_cd_big_cD),np.average(slope_gr_e_cD)
	vec_med_slope_gr_E=med_slope_gr_e_cD,med_slope_gr_e_misc,med_slope_gr_e_E=np.average(slope_gr_e_cD),np.average(slope_gr_e_misc),np.average(slope_gr_e_E)
	#

	ks_slope_gr_zhao=ks_2samp(slope_gr_cd,slope_gr_e)[1],ks_2samp(slope_gr_cd_zhao,slope_gr_e_zhao)[1]
	ks_slope_gr_cD_zhao=ks_2samp(slope_gr_cd_cD,slope_gr_e_cD)[1],ks_2samp(slope_gr_cd_big_cD,slope_gr_e_cD)[1],ks_2samp(slope_gr_cd_big_cD,slope_gr_cd_small_cD)[1],ks_2samp(slope_gr_e_cD,slope_gr_cd_small_cD)[1]
	ks_slope_gr_e_zhao=ks_2samp(slope_gr_e_E,slope_gr_e_misc)[1],ks_2samp(slope_gr_e_cD,slope_gr_e_E)[1],ks_2samp(slope_gr_e_cD,slope_gr_e_misc)[1]

	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope gr - comparação cD vs E - Nossa vs Zhao')
	axs.plot(slope_gr_linspace,slope_gr_kde_e_zhao(slope_gr_linspace),alpha=0.5,lw=2.5,color='green',label='E[Zhao]')
	axs.axvline(np.average(slope_gr_e_zhao),color='green',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_E = {np.average(slope_gr_e_zhao):.3e}$')
	axs.plot(slope_gr_linspace,slope_gr_kde_cd_zhao(slope_gr_linspace),color='red',alpha=0.5,lw=2.5,label='cD[Zhao]')
	axs.axvline(np.average(slope_gr_cd_zhao),color='red',alpha=0.5,lw=2.5,ls='--',label=fr'$\mu_cD = {np.average(slope_gr_cd_zhao):.3e}$')

	axs.plot(slope_gr_linspace,slope_gr_kde_e(slope_gr_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(slope_gr_e),color='red',ls='--',label=fr'$\mu_E = {np.average(slope_gr_e):.3e}$')
	axs.plot(slope_gr_linspace,slope_gr_kde_cd(slope_gr_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(slope_gr_cd),color='red',label=fr'$\mu_cD = {np.average(slope_gr_cd):.3e}$')
	axs.legend()
	axs.set_xlabel(r'$\alpha g-r$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_slope_gr_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_slope_gr_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope gr - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_gr_cD):
		axs.plot(slope_gr_linspace,dist(slope_gr_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_gr_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_gr_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$\alpha g-r$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_slope_gr_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_slope_gr_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_slope_gr_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_slope_gr_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope gr - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_slope_gr_e):
		axs.plot(slope_gr_linspace,dist(slope_gr_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_slope_gr_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_gr_E[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$\alpha g-r$')
	info_labels = (f'K-S(E/cD,E) ={ks_slope_gr_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_slope_gr_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_slope_gr_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/test_gr/slope_gr_kde_e_morf_zhao.png')
	plt.close()
######################################################################
#GRADIENTE E
##SÉRSIC -- cD,E(EL),cD(std),E
grad_e_cd,grad_e_cd_small,grad_e_cd_big,grad_e_e=grad_e[cd_lim_photutils],grad_e[cd_lim_photutils_small],grad_e[cd_lim_photutils_big],grad_e[elip_lim_photutils]
grad_e_cd_err,grad_e_cd_small_err,grad_e_cd_big_err,grad_e_e_err=grad_e_err[cd_lim_photutils],grad_e_err[cd_lim_photutils_small],grad_e_err[cd_lim_photutils_big],grad_e_err[elip_lim_photutils]

ks_grad_e=ks_calc([grad_e_cd,grad_e_cd_small,grad_e_cd_big,grad_e_e])

vec_ks_grad_e=ks_grad_e[0][1][2],ks_grad_e[0][1][3],ks_grad_e[0][2][3]
vec_ks_grad_e_2c=ks_grad_e[0][0][1],ks_grad_e[0][0][1],ks_grad_e[0][0][3]

kde_grad_e=kde(grad_e,weights=grad_e_err)
grad_e_factor = kde_grad_e.factor
grad_e_linspace=np.linspace(min(grad_e),max(grad_e),3000)
grad_e_kde_cd,grad_e_kde_cd_small,grad_e_kde_cd_big,grad_e_kde_e=kde(grad_e_cd,bw_method=grad_e_factor,weights=grad_e_cd_err),kde(grad_e_cd_small,bw_method=grad_e_factor,weights=grad_e_cd_small_err),kde(grad_e_cd_big,bw_method=grad_e_factor,weights=grad_e_cd_big_err),kde(grad_e_e,bw_method=grad_e_factor,weights=grad_e_e_err)

vec_med_grad_e=med_grad_e_cd,med_grad_e_cd_small,med_grad_e_cd_big,med_grad_e_e=np.average(grad_e_cd,weights=grad_e_cd_err),np.average(grad_e_cd_small,weights=grad_e_cd_small_err),np.average(grad_e_cd_big,weights=grad_e_cd_big_err),np.average(grad_e_e,weights=grad_e_e_err)
vec_kde_grad_e=grad_e_kde_cd,grad_e_kde_cd_small,grad_e_kde_cd_big,grad_e_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/grad_e',exist_ok=True)

plt.figure()
plt.title('Gradiente de Elipticidade - p_value')
sns.heatmap(ks_grad_e[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_pvalue.png')
plt.close()

plt.figure()
plt.title('Gradiente de Elipticidade - D_value')
sns.heatmap(ks_grad_e[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_dvalue.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Gradiente de Elipticidade')
for i,dist in enumerate(vec_kde_grad_e):
	axs.plot(grad_e_linspace,dist(grad_e_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_grad_e[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_e[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\nabla e$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_grad_e[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_grad_e[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_grad_e[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_grad_e_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_grad_e_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_grad_e_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	grad_e_cd_zhao,grad_e_e_zhao,grad_e_misc_zhao=grad_e[cd_cut_photutils],grad_e[e_cut_photutils],grad_e[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	grad_e_cd_cD,grad_e_cd_small_cD,grad_e_cd_big_cD,grad_e_e_cD=grad_e[cd_lim_photutils & cd_cut_photutils],grad_e[cd_lim_photutils_small & cd_cut_photutils],grad_e[cd_lim_photutils_big & cd_cut_photutils],grad_e[elip_lim_photutils & cd_cut_photutils]
	grad_e_e_E,grad_e_e_misc=grad_e[elip_lim_photutils & e_cut_photutils],grad_e[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##slope - KDE
	grad_e_kde_cd_zhao,grad_e_kde_e_zhao=kde(grad_e_cd_zhao,bw_method=grad_e_factor),kde(grad_e_e_zhao,bw_method=grad_e_factor)
	vec_kde_grad_e_cD=grad_e_kde_cd_cD,grad_e_kde_cd_small_cD,grad_e_kde_cd_big_cD,grad_e_kde_e_cD=kde(grad_e_cd_cD,bw_method=grad_e_factor),kde(grad_e_cd_small_cD,bw_method=grad_e_factor),kde(grad_e_cd_big_cD,bw_method=grad_e_factor),kde(grad_e_e_cD,bw_method=grad_e_factor)
	vec_kde_grad_e_e=grad_e_kde_e_cD,grad_e_kde_e_misc,grad_e_kde_e_E=kde(grad_e_e_cD,bw_method=grad_e_factor),kde(grad_e_e_misc,bw_method=grad_e_factor),kde(grad_e_e_E,bw_method=grad_e_factor)

	#MÉDIAS
	vec_med_grad_e_cD=med_grad_e_cd,med_grad_e_cd_small,med_grad_e_cd_big,med_grad_e_e=np.average(grad_e_cd_cD),np.average(grad_e_cd_small_cD),np.average(grad_e_cd_big_cD),np.average(grad_e_e_cD)
	vec_med_grad_e_E=med_grad_e_e_cD,med_grad_e_e_misc,med_grad_e_e_E=np.average(grad_e_e_cD),np.average(grad_e_e_misc),np.average(grad_e_e_E)
	#

	ks_grad_e_zhao=ks_2samp(grad_e_cd,grad_e_e)[1],ks_2samp(grad_e_cd_zhao,grad_e_e_zhao)[1]
	ks_grad_e_cD_zhao=ks_2samp(grad_e_cd_cD,grad_e_e_cD)[1],ks_2samp(grad_e_cd_big_cD,grad_e_e_cD)[1],ks_2samp(grad_e_cd_big_cD,grad_e_cd_small_cD)[1],ks_2samp(grad_e_e_cD,grad_e_cd_small_cD)[1]
	ks_grad_e_e_zhao=ks_2samp(grad_e_e_E,grad_e_e_misc)[1],ks_2samp(grad_e_e_cD,grad_e_e_E)[1],ks_2samp(grad_e_e_cD,grad_e_e_misc)[1]

	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Elipticidade - comparação cD vs E - Nossa vs Zhao')
	axs.plot(grad_e_linspace,grad_e_kde_e_zhao(grad_e_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(grad_e_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(grad_e_e_zhao):.3f}$')
	axs.plot(grad_e_linspace,grad_e_kde_cd_zhao(grad_e_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(grad_e_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(grad_e_cd_zhao):.3f}$')

	axs.plot(grad_e_linspace,grad_e_kde_e(grad_e_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(grad_e_e),color='red',ls='--',label=fr'$\mu_E = {np.average(grad_e_e):.3f}$')
	axs.plot(grad_e_linspace,grad_e_kde_cd(grad_e_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(grad_e_cd),color='red',label=fr'$\mu_cD = {np.average(grad_e_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla e$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_grad_e_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_grad_e_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Elipticidade - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_grad_e_cD):
		axs.plot(grad_e_linspace,dist(grad_e_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_grad_e_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_e_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla e$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_grad_e_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_grad_e_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_grad_e_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_grad_e_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Elipticidade - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_grad_e_e):
		axs.plot(grad_e_linspace,dist(grad_e_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_grad_e_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_e_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla e$')
	info_labels = (f'K-S(E/cD,E) ={ks_grad_e_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_grad_e_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_grad_e_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_e/grad_e_kde_e_morf_zhao.png')
	plt.close()

#####################################################################
#GRADIENTE PA
grad_pa_cd,grad_pa_cd_small,grad_pa_cd_big,grad_pa_e=grad_pa[cd_lim_photutils],grad_pa[cd_lim_photutils_small],grad_pa[cd_lim_photutils_big],grad_pa[elip_lim_photutils]
grad_pa_cd_err,grad_pa_cd_small_err,grad_pa_cd_big_err,grad_pa_e_err=grad_pa_err[cd_lim_photutils],grad_pa_err[cd_lim_photutils_small],grad_pa_err[cd_lim_photutils_big],grad_pa_err[elip_lim_photutils]

ks_grad_pa=ks_calc([grad_pa_cd,grad_pa_cd_small,grad_pa_cd_big,grad_pa_e])

vec_ks_grad_pa=ks_grad_pa[0][1][2],ks_grad_pa[0][1][3],ks_grad_pa[0][2][3]
vec_ks_grad_pa_2c=ks_grad_pa[0][0][1],ks_grad_pa[0][0][2],ks_grad_pa[0][0][3]

kde_grad_pa=kde(grad_pa,weights=grad_pa_err)
grad_pa_factor = kde_grad_pa.factor
grad_pa_linspace=np.linspace(min(grad_pa),5,3000)
grad_pa_kde_cd,grad_pa_kde_cd_small,grad_pa_kde_cd_big,grad_pa_kde_e=kde(grad_pa_cd,bw_method=grad_pa_factor,weights=grad_pa_cd_err),kde(grad_pa_cd_small,bw_method=grad_pa_factor,weights=grad_pa_cd_small_err),kde(grad_pa_cd_big,bw_method=grad_pa_factor,weights=grad_pa_cd_big_err),kde(grad_pa_e,bw_method=grad_pa_factor,weights=grad_pa_e_err)

vec_med_grad_pa=med_grad_pa_cd,med_grad_pa_cd_small,med_grad_pa_cd_big,med_grad_pa_e=np.average(grad_pa_cd,weights=grad_pa_cd_err),np.average(grad_pa_cd_small,weights=grad_pa_cd_small_err),np.average(grad_pa_cd_big,weights=grad_pa_cd_big_err),np.average(grad_pa_e,weights=grad_pa_e_err)
vec_kde_grad_pa=grad_pa_kde_cd,grad_pa_kde_cd_small,grad_pa_kde_cd_big,grad_pa_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/grad_pa',exist_ok=True)

plt.figure()
plt.title('Gradiente de PA - p_value')
sns.heatmap(ks_grad_e[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_pvalue.png')
plt.close()

plt.figure()
plt.title('Gradiente de PA - D_value')
sns.heatmap(ks_grad_e[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_dvalue.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Gradiente de PA')
for i,dist in enumerate(vec_kde_grad_pa):
	axs.plot(grad_pa_linspace,dist(grad_pa_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_grad_pa[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_pa[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\nabla PA$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_grad_pa[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_grad_pa[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_grad_pa[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_grad_pa_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_grad_pa_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_grad_pa_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	grad_pa_cd_zhao,grad_pa_e_zhao,grad_pa_misc_zhao=grad_pa[cd_cut_photutils],grad_pa[e_cut_photutils],grad_pa[(ecd_cut_photutils | cde_cut_photutils)]
	##slope - SUBGRUPO cD DO ZHAO
	grad_pa_cd_cD,grad_pa_cd_small_cD,grad_pa_cd_big_cD,grad_pa_e_cD=grad_pa[cd_lim_photutils & cd_cut_photutils],grad_pa[cd_lim_photutils_small & cd_cut_photutils],grad_pa[cd_lim_photutils_big & cd_cut_photutils],grad_pa[elip_lim_photutils & cd_cut_photutils]
	grad_pa_e_E,grad_pa_e_misc=grad_pa[elip_lim_photutils & e_cut_photutils],grad_pa[elip_lim_photutils & (ecd_cut_photutils | cde_cut_photutils)]
	
	##slope - KDE
	grad_pa_kde_cd_zhao,grad_pa_kde_e_zhao=kde(grad_pa_cd_zhao,bw_method=grad_pa_factor),kde(grad_pa_e_zhao,bw_method=grad_pa_factor)
	vec_kde_grad_pa_cD=grad_pa_kde_cd_cD,grad_pa_kde_cd_small_cD,grad_pa_kde_cd_big_cD,grad_pa_kde_e_cD=kde(grad_pa_cd_cD,bw_method=grad_pa_factor),kde(grad_pa_cd_small_cD,bw_method=grad_pa_factor),kde(grad_pa_cd_big_cD,bw_method=grad_pa_factor),kde(grad_pa_e_cD,bw_method=grad_pa_factor)
	vec_kde_grad_pa_e=grad_pa_kde_e_cD,grad_pa_kde_e_misc,grad_pa_kde_e_E=kde(grad_pa_e_cD,bw_method=grad_pa_factor),kde(grad_pa_e_misc,bw_method=grad_pa_factor),kde(grad_pa_e_E,bw_method=grad_pa_factor)

	#MÉDIAS
	vec_med_grad_pa_cD=med_grad_pa_cd,med_grad_pa_cd_small,med_grad_pa_cd_big,med_grad_pa_e=np.average(grad_pa_cd_cD),np.average(grad_pa_cd_small_cD),np.average(grad_pa_cd_big_cD),np.average(grad_pa_e_cD)
	vec_med_grad_pa_E=med_grad_pa_e_cD,med_grad_pa_e_misc,med_grad_pa_e_E=np.average(grad_pa_e_cD),np.average(grad_pa_e_misc),np.average(grad_pa_e_E)
	#

	ks_grad_pa_zhao=ks_2samp(grad_pa_cd,grad_pa_e)[1],ks_2samp(grad_pa_cd_zhao,grad_pa_e_zhao)[1]
	ks_grad_pa_cD_zhao=ks_2samp(grad_pa_cd_cD,grad_pa_e_cD)[1],ks_2samp(grad_pa_cd_big_cD,grad_pa_e_cD)[1],ks_2samp(grad_pa_cd_big_cD,grad_pa_cd_small_cD)[1],ks_2samp(grad_pa_e_cD,grad_pa_cd_small_cD)[1]
	ks_grad_pa_e_zhao=ks_2samp(grad_pa_e_E,grad_pa_e_misc)[1],ks_2samp(grad_pa_e_cD,grad_pa_e_E)[1],ks_2samp(grad_pa_e_cD,grad_pa_e_misc)[1]

	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Posição Angular - comparação cD vs E - Nossa vs Zhao')
	axs.plot(grad_pa_linspace,grad_pa_kde_e_zhao(grad_pa_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(grad_pa_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(grad_pa_e_zhao):.3f}$')
	axs.plot(grad_pa_linspace,grad_pa_kde_cd_zhao(grad_pa_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(grad_pa_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(grad_pa_cd_zhao):.3f}$')

	axs.plot(grad_pa_linspace,grad_pa_kde_e(grad_pa_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(grad_pa_e),color='red',ls='--',label=fr'$\mu_E = {np.average(grad_pa_e):.3f}$')
	axs.plot(grad_pa_linspace,grad_pa_kde_cd(grad_pa_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(grad_pa_cd),color='red',label=fr'$\mu_cD = {np.average(grad_pa_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla PA$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_grad_pa_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_grad_pa_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Posição Angular - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_grad_pa_cD):
		axs.plot(grad_pa_linspace,dist(grad_pa_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_grad_pa_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_pa_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla PA$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_grad_pa_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_grad_pa_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_grad_pa_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_grad_pa_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Posição Angular - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_grad_pa_e):
		axs.plot(grad_pa_linspace,dist(grad_pa_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_grad_pa_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_pa_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla PA$')
	info_labels = (f'K-S(E/cD,E) ={ks_grad_pa_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_grad_pa_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_grad_pa_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/grad_pa/grad_pa_kde_e_morf_zhao.png')
	plt.close()
#MAG ABS
magabs_cd,magabs_cd_small,magabs_cd_big,magabs_e=magabs[cd_lim_casjobs],magabs[cd_lim_casjobs_small],magabs[cd_lim_casjobs_big],magabs[elip_lim_casjobs]
ks_magabs=ks_calc([magabs_cd,magabs_cd_small,magabs_cd_big,magabs_e])

vec_ks_magabs=ks_magabs[0][1][2],ks_magabs[0][1][3],ks_magabs[0][2][3]

kde_magabs=kde(magabs)
magabs_factor = kde_magabs.factor
magabs_linspace=np.linspace(min(magabs),max(magabs),3000)
magabs_kde_cd,magabs_kde_cd_small,magabs_kde_cd_big,magabs_kde_e=kde(magabs_cd,bw_method=magabs_factor),kde(magabs_cd_small,bw_method=magabs_factor),kde(magabs_cd_big,bw_method=magabs_factor),kde(magabs_e,bw_method=magabs_factor)

vec_med_magabs=med_magabs_cd,med_magabs_cd_small,med_magabs_cd_big,med_magabs_e=np.average(magabs_cd),np.average(magabs_cd_small),np.average(magabs_cd_big),np.average(magabs_e)
vec_kde_magabs_simples=magabs_kde_cd,magabs_kde_cd_small,magabs_kde_cd_big,magabs_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/magabs',exist_ok=True)

plt.figure()
plt.title('Magnitude Absoluta - p_value')
sns.heatmap(ks_magabs[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_pvalue.png')
plt.close()

plt.figure()
plt.title('Magnitude Absoluta - D_value')
sns.heatmap(ks_magabs[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Magnitude Absoluta')
for i,dist in enumerate(vec_kde_magabs_simples[1:]):
	axs.plot(magabs_linspace,dist(magabs_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_magabs[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_magabs[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$Mag_\odot$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_magabs[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_magabs[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_magabs[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	magabs_cd_zhao,magabs_e_zhao,magabs_misc_zhao=magabs[cd_cut_casjobs],magabs[e_cut_casjobs],magabs[(ecd_cut_casjobs | cde_cut_casjobs)]
	##slope - SUBGRUPO cD DO ZHAO
	magabs_cd_cD,magabs_cd_small_cD,magabs_cd_big_cD,magabs_e_cD=magabs[cd_lim_casjobs & cd_cut_casjobs],magabs[cd_lim_casjobs_small & cd_cut_casjobs],magabs[cd_lim_casjobs_big & cd_cut_casjobs],magabs[elip_lim_casjobs & cd_cut_casjobs]
	magabs_e_E,magabs_e_misc=magabs[elip_lim_casjobs & e_cut_casjobs],magabs[elip_lim_casjobs & (ecd_cut_casjobs | cde_cut_casjobs)]
	
	##slope - KDE
	magabs_kde_cd_zhao,magabs_kde_e_zhao=kde(magabs_cd_zhao,bw_method=magabs_factor),kde(magabs_e_zhao,bw_method=magabs_factor)
	vec_kde_magabs_cD=magabs_kde_cd_cD,magabs_kde_cd_small_cD,magabs_kde_cd_big_cD,magabs_kde_e_cD=kde(magabs_cd_cD,bw_method=magabs_factor),kde(magabs_cd_small_cD,bw_method=magabs_factor),kde(magabs_cd_big_cD,bw_method=magabs_factor),kde(magabs_e_cD,bw_method=magabs_factor)
	vec_kde_magabs_e=magabs_kde_e_cD,magabs_kde_e_misc,magabs_kde_e_E=kde(magabs_e_cD,bw_method=magabs_factor),kde(magabs_e_misc,bw_method=magabs_factor),kde(magabs_e_E,bw_method=magabs_factor)

	#MÉDIAS
	vec_med_magabs_cD=med_magabs_cd,med_magabs_cd_small,med_magabs_cd_big,med_magabs_e=np.average(magabs_cd_cD),np.average(magabs_cd_small_cD),np.average(magabs_cd_big_cD),np.average(magabs_e_cD)
	vec_med_magabs_E=med_magabs_e_cD,med_magabs_e_misc,med_magabs_e_E=np.average(magabs_e_cD),np.average(magabs_e_misc),np.average(magabs_e_E)
	#
	ks_magabs_zhao=ks_2samp(magabs_cd,magabs_e)[1],ks_2samp(magabs_cd_zhao,magabs_e_zhao)[1]
	ks_magabs_cD_zhao=ks_2samp(magabs_cd_cD,magabs_e_cD)[1],ks_2samp(magabs_cd_big_cD,magabs_e_cD)[1],ks_2samp(magabs_cd_big_cD,magabs_cd_small_cD)[1],ks_2samp(magabs_e_cD,magabs_cd_small_cD)[1]
	ks_magabs_e_zhao=ks_2samp(magabs_e_E,magabs_e_misc)[1],ks_2samp(magabs_e_cD,magabs_e_E)[1],ks_2samp(magabs_e_cD,magabs_e_misc)[1]
	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Magnitude Absoluta - comparação cD vs E - Nossa vs Zhao')
	axs.plot(magabs_linspace,magabs_kde_e_zhao(magabs_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(magabs_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(magabs_e_zhao):.3f}$')
	axs.plot(magabs_linspace,magabs_kde_cd_zhao(magabs_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(magabs_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(magabs_cd_zhao):.3f}$')

	axs.plot(magabs_linspace,magabs_kde_e(magabs_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(magabs_e),color='red',ls='--',label=fr'$\mu_E = {np.average(magabs_e):.3f}$')
	axs.plot(magabs_linspace,magabs_kde_cd(magabs_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(magabs_cd),color='red',label=fr'$\mu_cD = {np.average(magabs_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$Mag_\odot$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_magabs_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_magabs_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Magnitude Absoluta - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_magabs_cD):
		axs.plot(magabs_linspace,dist(magabs_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_magabs_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_magabs_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$Mag_\odot$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_magabs_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_magabs_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_magabs_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_magabs_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Magnitude Absoluta - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_magabs_e):
		axs.plot(magabs_linspace,dist(magabs_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_magabs_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_magabs_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$Mag_\odot$')
	info_labels = (f'K-S(E/cD,E) ={ks_magabs_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_magabs_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_magabs_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/magabs/magabs_kde_e_morf_zhao.png')
	plt.close()
############################################################
#MASS ESTELAR
starmass_cd,starmass_cd_small,starmass_cd_big,starmass_e=starmass[cd_lim_casjobs],starmass[cd_lim_casjobs_small],starmass[cd_lim_casjobs_big],starmass[elip_lim_casjobs]
ks_starmass=ks_calc([starmass_cd,starmass_cd_small,starmass_cd_big,starmass_e])

vec_ks_starmass=ks_starmass[0][1][2],ks_starmass[0][1][3],ks_starmass[0][2][3]

kde_starmass=kde(starmass)
starmass_factor = kde_starmass.factor
starmass_linspace=np.linspace(10.8,max(starmass),3000)
starmass_kde_cd,starmass_kde_cd_small,starmass_kde_cd_big,starmass_kde_e=kde(starmass_cd,bw_method=starmass_factor),kde(starmass_cd_small,bw_method=starmass_factor),kde(starmass_cd_big,bw_method=starmass_factor),kde(starmass_e,bw_method=starmass_factor)

vec_med_starmass=med_starmass_cd,med_starmass_cd_small,med_starmass_cd_big,med_starmass_e=np.average(starmass_cd),np.average(starmass_cd_small),np.average(starmass_cd_big),np.average(starmass_e)
vec_kde_starmass=starmass_kde_cd,starmass_kde_cd_small,starmass_kde_cd_big,starmass_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/starmass',exist_ok=True)

plt.figure()
plt.title('Massa estelar - p_value')
sns.heatmap(ks_starmass[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_pvalue.png')
plt.close()

plt.figure()
plt.title('Massa estelar - D_value')
sns.heatmap(ks_starmass[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Massa estelar')
for i,dist in enumerate(vec_kde_starmass[1:]):
	axs.plot(starmass_linspace,dist(starmass_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_starmass[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\log M_\odot$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_starmass[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_starmass[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_starmass[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	starmass_cd_zhao,starmass_e_zhao,starmass_misc_zhao=starmass[cd_cut_casjobs],starmass[e_cut_casjobs],starmass[(ecd_cut_casjobs | cde_cut_casjobs)]
	##slope - SUBGRUPO cD DO ZHAO
	starmass_cd_cD,starmass_cd_small_cD,starmass_cd_big_cD,starmass_e_cD=starmass[cd_lim_casjobs & cd_cut_casjobs],starmass[cd_lim_casjobs_small & cd_cut_casjobs],starmass[cd_lim_casjobs_big & cd_cut_casjobs],starmass[elip_lim_casjobs & cd_cut_casjobs]
	starmass_e_E,starmass_e_misc=starmass[elip_lim_casjobs & e_cut_casjobs],starmass[elip_lim_casjobs & (ecd_cut_casjobs | cde_cut_casjobs)]
	
	##slope - KDE
	vec_kde_starmass_zhao=starmass_kde_cd_zhao,starmass_kde_e_zhao=kde(starmass_cd_zhao,bw_method=starmass_factor),kde(starmass_e_zhao,bw_method=starmass_factor)
	vec_kde_starmass_cD=starmass_kde_cd_cD,starmass_kde_cd_small_cD,starmass_kde_cd_big_cD,starmass_kde_e_cD=kde(starmass_cd_cD,bw_method=starmass_factor),kde(starmass_cd_small_cD,bw_method=starmass_factor),kde(starmass_cd_big_cD,bw_method=starmass_factor),kde(starmass_e_cD,bw_method=starmass_factor)
	vec_kde_starmass_e=starmass_kde_e_cD,starmass_kde_e_misc,starmass_kde_e_E=kde(starmass_e_cD,bw_method=starmass_factor),kde(starmass_e_misc,bw_method=starmass_factor),kde(starmass_e_E,bw_method=starmass_factor)

	#MÉDIAS
	vec_med_starmass_zhao=med_starmass_cd_zhao,med_starmass_e_zhao=np.average(starmass_cd_zhao),np.average(starmass_e_zhao)
	vec_med_starmass_cD=med_starmass_cd,med_starmass_cd_small,med_starmass_cd_big,med_starmass_e=np.average(starmass_cd_cD),np.average(starmass_cd_small_cD),np.average(starmass_cd_big_cD),np.average(starmass_e_cD)
	vec_med_starmass_E=med_starmass_e_cD,med_starmass_e_misc,med_starmass_e_E=np.average(starmass_e_cD),np.average(starmass_e_misc),np.average(starmass_e_E)
	#
	ks_starmass_zhao=ks_2samp(starmass_cd,starmass_e)[1],ks_2samp(starmass_cd_zhao,starmass_e_zhao)[1]
	ks_starmass_cD_zhao=ks_2samp(starmass_cd_cD,starmass_e_cD)[1],ks_2samp(starmass_cd_big_cD,starmass_e_cD)[1],ks_2samp(starmass_cd_big_cD,starmass_cd_small_cD)[1],ks_2samp(starmass_e_cD,starmass_cd_small_cD)[1]
	ks_starmass_e_zhao=ks_2samp(starmass_e_E,starmass_e_misc)[1],ks_2samp(starmass_e_cD,starmass_e_E)[1],ks_2samp(starmass_e_cD,starmass_e_misc)[1]
	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Massa estelar - comparação cD vs E - Nossa vs Zhao')
	axs.plot(starmass_linspace,starmass_kde_e_zhao(starmass_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(starmass_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(starmass_e_zhao):.3f}$')
	axs.plot(starmass_linspace,starmass_kde_cd_zhao(starmass_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(starmass_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(starmass_cd_zhao):.3f}$')

	axs.plot(starmass_linspace,starmass_kde_e(starmass_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(starmass_e),color='red',ls='--',label=fr'$\mu_E = {np.average(starmass_e):.3f}$')
	axs.plot(starmass_linspace,starmass_kde_cd(starmass_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(starmass_cd),color='red',label=fr'$\mu_cD = {np.average(starmass_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_starmass_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_starmass_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Massa estelar - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_starmass_cD):
		axs.plot(starmass_linspace,dist(starmass_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_starmass_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_starmass_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_starmass_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_starmass_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_starmass_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Massa estelar - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_starmass_e):
		axs.plot(starmass_linspace,dist(starmass_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_starmass_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(E/cD,E) ={ks_starmass_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_starmass_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_starmass_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass/starmass_kde_e_morf_zhao.png')
	plt.close()

#########################################################
#IDADE
age_cd,age_cd_small,age_cd_big,age_e=age[cd_lim_casjobs],age[cd_lim_casjobs_small],age[cd_lim_casjobs_big],age[elip_lim_casjobs]
ks_age=ks_calc([age_cd,age_cd_small,age_cd_big,age_e])

vec_ks_age=ks_age[0][1][2],ks_age[0][1][3],ks_age[0][2][3]

kde_age=kde(age)
age_factor = kde_age.factor
age_linspace=np.linspace(min(age),max(age),3000)
age_kde_cd,age_kde_cd_small,age_kde_cd_big,age_kde_e=kde(age_cd,bw_method=age_factor),kde(age_cd_small,bw_method=age_factor),kde(age_cd_big,bw_method=age_factor),kde(age_e,bw_method=age_factor)

vec_med_age=med_age_cd,med_age_cd_small,med_age_cd_big,med_age_e=np.average(age_cd),np.average(age_cd_small),np.average(age_cd_big),np.average(age_e)
vec_kde_age=age_kde_cd,age_kde_cd_small,age_kde_cd_big,age_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/age',exist_ok=True)

plt.figure()
plt.title('Idade Estelar - p_value')
sns.heatmap(ks_age[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_pvalue.png')
plt.close()

plt.figure()
plt.title('Idade Estelar - D_value')
sns.heatmap(ks_age[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Idade Estelar')
for i,dist in enumerate(vec_kde_age[1:]):
	axs.plot(age_linspace,dist(age_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_age[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_age[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\tau (Gyr)$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_age[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_age[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_age[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	age_cd_zhao,age_e_zhao,age_misc_zhao=age[cd_cut_casjobs],age[e_cut_casjobs],age[(ecd_cut_casjobs | cde_cut_casjobs)]
	##slope - SUBGRUPO cD DO ZHAO
	age_cd_cD,age_cd_small_cD,age_cd_big_cD,age_e_cD=age[cd_lim_casjobs & cd_cut_casjobs],age[cd_lim_casjobs_small & cd_cut_casjobs],age[cd_lim_casjobs_big & cd_cut_casjobs],age[elip_lim_casjobs & cd_cut_casjobs]
	age_e_E,age_e_misc=age[elip_lim_casjobs & e_cut_casjobs],age[elip_lim_casjobs & (ecd_cut_casjobs | cde_cut_casjobs)]
	
	##slope - KDE
	vec_kde_age_zhao=age_kde_cd_zhao,age_kde_e_zhao=kde(age_cd_zhao,bw_method=age_factor),kde(age_e_zhao,bw_method=age_factor)
	vec_kde_age_cD=age_kde_cd_cD,age_kde_cd_small_cD,age_kde_cd_big_cD,age_kde_e_cD=kde(age_cd_cD,bw_method=age_factor),kde(age_cd_small_cD,bw_method=age_factor),kde(age_cd_big_cD,bw_method=age_factor),kde(age_e_cD,bw_method=age_factor)
	vec_kde_age_e=age_kde_e_cD,age_kde_e_misc,age_kde_e_E=kde(age_e_cD,bw_method=age_factor),kde(age_e_misc,bw_method=age_factor),kde(age_e_E,bw_method=age_factor)

	#MÉDIAS
	vec_med_age_zhao=med_age_cd_zhao,med_age_e_zhao=np.average(age_cd_zhao),np.average(age_e_zhao)
	vec_med_age_cD=med_age_cd_cD,med_age_cd_small_cD,med_age_cd_big_cD,med_age_e_cD=np.average(age_cd_cD),np.average(age_cd_small_cD),np.average(age_cd_big_cD),np.average(age_e_cD)
	vec_med_age_E=med_age_e_cD,med_age_e_misc,med_age_e_E=np.average(age_e_cD),np.average(age_e_misc),np.average(age_e_E)
	#
	ks_age_zhao=ks_2samp(age_cd,age_e)[1],ks_2samp(age_cd_zhao,age_e_zhao)[1]
	ks_age_cD_zhao=ks_2samp(age_cd_cD,age_e_cD)[1],ks_2samp(age_cd_big_cD,age_e_cD)[1],ks_2samp(age_cd_big_cD,age_cd_small_cD)[1],ks_2samp(age_e_cD,age_cd_small_cD)[1]
	ks_age_e_zhao=ks_2samp(age_e_E,age_e_misc)[1],ks_2samp(age_e_cD,age_e_E)[1],ks_2samp(age_e_cD,age_e_misc)[1]
	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Idade estelar - comparação cD vs E - Nossa vs Zhao')
	axs.plot(age_linspace,age_kde_e_zhao(age_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(age_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(age_e_zhao):.3f}$')
	axs.plot(age_linspace,age_kde_cd_zhao(age_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(age_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(age_cd_zhao):.3f}$')

	axs.plot(age_linspace,age_kde_e(age_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(age_e),color='red',ls='--',label=fr'$\mu_E = {np.average(age_e):.3f}$')
	axs.plot(age_linspace,age_kde_cd(age_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(age_cd),color='red',label=fr'$\mu_cD = {np.average(age_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_age_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_age_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Idade estelar - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_age_cD):
		axs.plot(age_linspace,dist(age_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_age_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_age_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_age_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_age_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_age_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_age_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Idade estelar - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_age_e):
		axs.plot(age_linspace,dist(age_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_age_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_age_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(E/cD,E) ={ks_age_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_age_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_age_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/age/age_kde_e_morf_zhao.png')
	plt.close()
#########################################################
#M200

m200=m200[lim_casjobs]
m200_cd,m200_cd_small,m200_cd_big,m200_e=m200[cd_lim_casjobs],m200[cd_lim_casjobs_small],m200[cd_lim_casjobs_big],m200[elip_lim_casjobs]
ks_m200=ks_calc([m200_cd,m200_cd_small,m200_cd_big,m200_e])

vec_ks_m200=ks_m200[0][1][2],ks_m200[0][1][3],ks_m200[0][2][3]

kde_m200=kde(m200)
m200_factor = kde_m200.factor
m200_linspace=np.linspace(min(m200),max(m200),3000)
m200_kde_cd,m200_kde_cd_small,m200_kde_cd_big,m200_kde_e=kde(m200_cd,bw_method=m200_factor),kde(m200_cd_small,bw_method=m200_factor),kde(m200_cd_big,bw_method=m200_factor),kde(m200_e,bw_method=m200_factor)

vec_med_m200=med_m200_cd,med_m200_cd_small,med_m200_cd_big,med_m200_e=np.average(m200_cd),np.average(m200_cd_small),np.average(m200_cd_big),np.average(m200_e)
vec_kde_m200=m200_kde_cd,m200_kde_cd_small,m200_kde_cd_big,m200_kde_e
###############################################
##

#JOINTPLOT DA MASSA x M200

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ajust_e,cov_e=np.polyfit(m200_e,starmass_e,1,cov=True)
ajust_small,cov_small=np.polyfit(m200_cd_small,starmass_cd_small,1,cov=True)
ajust_big,cov_big=np.polyfit(m200_cd_big,starmass_cd_big,1,cov=True)

ax_center.scatter(m200_e,starmass_e,marker='o',edgecolor='black',label=label_elip,color='green')
ax_center.scatter(m200_cd_small,starmass_cd_small,marker='o',edgecolor='black',label=label_elip_el,color='blue')
ax_center.scatter(m200_cd_big,starmass_cd_big,marker='o',edgecolor='black',label='True cD',color='red')
ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_e),color='green',label=fr'$\alpha$={ajust_e[0]:.3f}$\pm${np.sqrt(cov_e[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e[1]:.3f}$\pm${np.sqrt(cov_e[1,1]):.3f}')
ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_small),color='blue',label=fr'$\alpha$={ajust_small[0]:.3f}$\pm${np.sqrt(cov_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_small[1]:.3f}$\pm${np.sqrt(cov_small[1,1]):.3f}')
ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_big),color='red',label=fr'$\alpha$={ajust_big[0]:.3f}$\pm${np.sqrt(cov_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_big[1]:.3f}$\pm${np.sqrt(cov_big[1,1]):.3f}')
ax_center.legend(fontsize='x-small')
ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
ax_center.set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
ax_center.set_ylabel(r'$\log M_{\bigstar}$')

for i,dist in enumerate(vec_kde_m200[1:]):
	ax_topx.plot(m200_linspace,dist(m200_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	ax_topx.axvline(vec_med_m200[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_m200[1:][i]:.3f}$')
ax_topx.legend(fontsize='x-small')
ax_topx.tick_params(labelbottom=False)

for i,dist in enumerate(vec_kde_starmass[1:]):
	ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cores[1:][i],label=f'{names_simples[1:][i]}')
	ax_righty.axhline(vec_med_starmass[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass[1:][i]:.3f}$')
ax_righty.legend(fontsize='x-small')
ax_righty.tick_params(labelleft=False)


plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_m200.png')
plt.close()

#JOINTPLOT DA MASSA x IDADE ESTELAR

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ajust_e=np.polyfit(starmass_e,age_e,1)
ajust_small=np.polyfit(starmass_cd_small,age_cd_small,1)
ajust_big=np.polyfit(starmass_cd_big,age_cd_big,1)

ax_center.scatter(starmass_e,age_e,marker='o',edgecolor='black',label=label_elip,color='green')
ax_center.scatter(starmass_cd_small,age_cd_small,marker='o',edgecolor='black',label=label_elip_el,color='blue')
ax_center.scatter(starmass_cd_big,age_cd_big,marker='o',edgecolor='black',label='True cD',color='red')
ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_e),color='green',label=fr'$\alpha$={ajust_e[0]:.3f}')
ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_small),color='blue',label=fr'$\alpha$={ajust_small[0]:.3f}')
ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_big),color='red',label=fr'$\alpha$={ajust_big[0]:.3f}')
ax_center.legend()
ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
ax_center.set_ylabel(r'$\tau$ (Gyr)')
ax_center.set_xlabel(r'$\log M_{\bigstar}$')
ax_center.set_ylabel(r'$\tau$ (Gyr)')


for i,dist in enumerate(vec_kde_starmass[1:]):
	ax_topx.plot(starmass_linspace,dist(starmass_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	ax_topx.axvline(vec_med_starmass[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass[1:][i]:.3f}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

for i,dist in enumerate(vec_kde_age[1:]):
	ax_righty.plot(dist(age_linspace),age_linspace,color=cores[1:][i],label=f'{names_simples[1:][i]}')
	ax_righty.axhline(vec_med_age[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_age[1:][i]:.3f}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)
plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_age.png')
plt.close()
#MASSAS DO COMPONENTE INTERNO/EXTERNO x M200 -- BT SEM CORREÇÃO

bt_mass=bt_vec_12[lim_casjobs]

bt_cd,bt_cd_small,bt_cd_big,bt_e=bt_mass[cd_lim_casjobs],bt_mass[cd_lim_casjobs_small],bt_mass[cd_lim_casjobs_big],bt_mass[elip_lim_casjobs]

mass_c1_e=np.log10(np.multiply(bt_e,np.power(10,starmass_e)))
mass_c2_e=np.log10(np.multiply((1-bt_e),np.power(10,starmass_e)))

mass_c1_cd=np.log10(np.multiply(bt_cd,np.power(10,starmass_cd)))
mass_c2_cd=np.log10(np.multiply((1-bt_cd),np.power(10,starmass_cd)))

mass_c1_cd_small=np.log10(np.multiply(bt_cd_small,np.power(10,starmass_cd_small)))
mass_c2_cd_small=np.log10(np.multiply((1-bt_cd_small),np.power(10,starmass_cd_small)))

mass_c1_cd_big=np.log10(np.multiply(bt_cd_big,np.power(10,starmass_cd_big)))
mass_c2_cd_big=np.log10(np.multiply((1-bt_cd_big),np.power(10,starmass_cd_big)))

#
fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO')

ajust_c1_small,cov_c1_small=np.polyfit(m200_cd_small,mass_c1_cd_small,1,cov=True)
ajust_c1_big,cov_c1_big=np.polyfit(m200_cd_big,mass_c1_cd_big,1,cov=True)

ajust_c2_small,cov_c2_small=np.polyfit(m200_cd_small,mass_c2_cd_small,1,cov=True)
ajust_c2_big,cov_c2_big=np.polyfit(m200_cd_big,mass_c2_cd_big,1,cov=True)

axs[0].scatter(m200_cd_small,mass_c1_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small[0]:.3f}$\pm${np.sqrt(cov_c1_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small[1]:.3f}$\pm${np.sqrt(cov_c1_small[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c1_cd_big,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big[0]:.3f}$\pm${np.sqrt(cov_c1_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big[1]:.3f}$\pm${np.sqrt(cov_c1_big[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c1.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('ELIPTICAS X COMPONENTE EXTERNO EL')

axs[0].scatter(m200_e,starmass_e,marker='o',edgecolor='black',label=label_elip,alpha=0.6,color='green')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_e),color='black',ls='--',label=fr'$\alpha$={ajust_e[0]:.3f}$\pm${np.sqrt(cov_e[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e[1]:.3f}$\pm${np.sqrt(cov_e[1,1]):.3f}')

axs[1].scatter(m200_cd_small,mass_c2_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small[0]:.3f}$\pm${np.sqrt(cov_c2_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small[1]:.3f}$\pm${np.sqrt(cov_c2_small[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_elip_elc2.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE EXTERNO')

axs[0].scatter(m200_cd_small,mass_c2_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small[0]:.3f}$\pm${np.sqrt(cov_c2_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small[1]:.3f}$\pm${np.sqrt(cov_c2_small[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c2_cd_big,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big[0]:.3f}$\pm${np.sqrt(cov_c2_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big[1]:.3f}$\pm${np.sqrt(cov_c2_big[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c2.png')
plt.close()

fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES)')

axs[0,0].scatter(m200_cd_small,mass_c1_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small[0]:.3f}$\pm${np.sqrt(cov_c1_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small[1]:.3f}$\pm${np.sqrt(cov_c1_small[1,1]):.3f}')
axs[0,0].set_ylim(9.2,12.2)
axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

axs[0,1].scatter(m200_cd_big,mass_c1_cd_big,marker='o',edgecolor='black',label='True cD',color='red')
axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big[0]:.3f}$\pm${np.sqrt(cov_c1_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big[1]:.3f}$\pm${np.sqrt(cov_c1_big[1,1]):.3f}')
axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,0].scatter(m200_cd_small,mass_c2_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small[0]:.3f}$\pm${np.sqrt(cov_c2_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small[1]:.3f}$\pm${np.sqrt(cov_c2_small[1,1]):.3f}')
axs[1,0].set_ylabel(r'$\log M_{\odot}$')
axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,1].scatter(m200_cd_big,mass_c2_cd_big,marker='o',edgecolor='black',label='True cD',color='red')
axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big[0]:.3f}$\pm${np.sqrt(cov_c2_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big[1]:.3f}$\pm${np.sqrt(cov_c2_big[1,1]):.3f}')
axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps.png')
plt.close()


#MASSAS DO COMPONENTE INTERNO/EXTERNO x M200 -- BT CORRIGIda

bt_mass_corr=bt_vec_corr[lim_casjobs]

bt_cd_corr,bt_cd_small_corr,bt_cd_big_corr,bt_e_corr=bt_mass_corr[cd_lim_casjobs],bt_mass_corr[cd_lim_casjobs_small],bt_mass_corr[cd_lim_casjobs_big],bt_mass_corr[elip_lim_casjobs]

mass_c1_e_corr=np.log10(np.multiply(bt_e_corr,np.power(10,starmass_e)))
mass_c2_e_corr=np.log10(np.multiply((1-bt_e_corr),np.power(10,starmass_e)))

mass_c1_cd_corr=np.log10(np.multiply(bt_cd_corr,np.power(10,starmass_cd)))
mass_c2_cd_corr=np.log10(np.multiply((1-bt_cd_corr),np.power(10,starmass_cd)))

mass_c1_cd_small_corr=np.log10(np.multiply(bt_cd_small_corr,np.power(10,starmass_cd_small)))
mass_c2_cd_small_corr=np.log10(np.multiply((1-bt_cd_small_corr),np.power(10,starmass_cd_small)))

mass_c1_cd_big_corr=np.log10(np.multiply(bt_cd_big_corr,np.power(10,starmass_cd_big)))
mass_c2_cd_big_corr=np.log10(np.multiply((1-bt_cd_big_corr),np.power(10,starmass_cd_big)))

#
fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO -- corrigida')

ajust_c1_small_corr,cov_c1_small_corr=np.polyfit(m200_cd_small,mass_c1_cd_small_corr,1,cov=True)
ajust_c1_big_corr,cov_c1_big_corr=np.polyfit(m200_cd_big,mass_c1_cd_big_corr,1,cov=True)

ajust_c2_small_corr,cov_c2_small_corr=np.polyfit(m200_cd_small,mass_c2_cd_small_corr,1,cov=True)
ajust_c2_big_corr,cov_c2_big_corr=np.polyfit(m200_cd_big,mass_c2_cd_big_corr,1,cov=True)

axs[0].scatter(m200_cd_small,mass_c1_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_corr[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c1_cd_big_corr,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_corr[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c1_corr.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('ELIPTICAS X COMPONENTE EXTERNO EL - corrigida')

axs[0].scatter(m200_e,starmass_e,marker='o',edgecolor='black',label=label_elip,alpha=0.6,color='green')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_e),color='black',ls='--',label=fr'$\alpha$={ajust_e[0]:.3f}$\pm${np.sqrt(cov_e[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e[1]:.3f}$\pm${np.sqrt(cov_e[1,1]):.3f}')

axs[1].scatter(m200_cd_small,mass_c2_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_corr[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_elip_elc2_corr.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE EXTERNO - corrigida')

axs[0].scatter(m200_cd_small,mass_c2_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_corr[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c2_cd_big_corr,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_corr[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c2_corr.png')
plt.close()

fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - corrigida')

axs[0,0].scatter(m200_cd_small,mass_c1_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_corr[1,1]):.3f}')
axs[0,0].set_ylim(9.2,12.2)
axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

axs[0,1].scatter(m200_cd_big,mass_c1_cd_big_corr,marker='o',edgecolor='black',label='True cD',color='red')
axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_corr[1,1]):.3f}')
axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,0].scatter(m200_cd_small,mass_c2_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_corr[1,1]):.3f}')
axs[1,0].set_ylabel(r'$\log M_{\odot}$')
axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,1].scatter(m200_cd_big,mass_c2_cd_big_corr,marker='o',edgecolor='black',label='True cD',color='red')
axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_corr[1,1]):.3f}')
axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

fig.legend()
plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps_corr.png')
plt.close()

if sample=='L07':

	m200_cd_zhao,m200_e_zhao=m200[cd_cut_casjobs],m200[e_cut_casjobs]
	m200_cd_cD,m200_cd_small_cD,m200_cd_big_cD,m200_e_cD=m200[cd_lim_casjobs & cd_cut_casjobs],m200[cd_lim_casjobs_small & cd_cut_casjobs],m200[cd_lim_casjobs_big & cd_cut_casjobs],m200[elip_lim_casjobs & cd_cut_casjobs]
	m200_e_misc,m200_e_E=m200[elip_lim_casjobs & (ecd_cut_casjobs | cde_cut_casjobs)],m200[elip_lim_casjobs & e_cut_casjobs]

	m200_kde_cd_zhao,m200_kde_e_zhao=kde(m200_cd_zhao,bw_method=m200_factor),kde(m200_e_zhao,bw_method=m200_factor)
	m200_kde_cd_cD,m200_kde_cd_small_cD,m200_kde_cd_big_cD,m200_kde_e_cD=kde(m200_cd_cD,bw_method=m200_factor),kde(m200_cd_small_cD,bw_method=m200_factor),kde(m200_cd_big_cD,bw_method=m200_factor),kde(m200_e_cD,bw_method=m200_factor)
	m200_kde_e_cD,m200_kde_e_misc,m200_kde_e_E=kde(m200_e_cD,bw_method=m200_factor),kde(m200_e_misc,bw_method=m200_factor),kde(m200_e_E,bw_method=m200_factor)
	
	vec_kde_m200_zhao=m200_kde_cd_zhao,m200_kde_e_zhao
	vec_kde_m200_cD_zhao=m200_kde_cd_cD,m200_kde_cd_small_cD,m200_kde_cd_big_cD,m200_kde_e_cD
	vec_kde_m200_e_zhao=m200_kde_e_cD,m200_kde_e_misc,m200_kde_e_E
	
	vec_med_m200_zhao=med_m200_cd_zhao,med_m200_e_zhao=np.average(m200_cd_zhao),np.average(m200_e_zhao)
	vec_med_m200_cD_zhao=med_m200_cd_cD,med_m200_cd_small_cD,med_m200_cd_big_cD,med_m200_e_cD=np.average(m200_cd_cD),np.average(m200_cd_small_cD),np.average(m200_cd_big_cD),np.average(m200_e_cD)
	vec_med_m200_e_zhao=med_m200_e_cD,med_m200_e_misc,med_m200_e_E=np.average(m200_e_cD),np.average(m200_e_misc),np.average(m200_e_E)

	#JOINTPLOT DA MASSA x M200 -- CLASSIFICAÇÃO DO ZHAO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e_zhao,cov_e_zhao=np.polyfit(m200_e_zhao,starmass_e_zhao,1,cov=True)
	ajust_cd_zhao,cov_cd_zhao=np.polyfit(m200_cd_zhao,starmass_cd_zhao,1,cov=True)

	ax_center.scatter(m200_e_zhao,starmass_e_zhao,marker='o',color='green',edgecolor='black',label=f'{label_elip}[ZHAO]')
	ax_center.scatter(m200_cd_zhao,starmass_cd_zhao,marker='o',color='red',edgecolor='black',label=f'{label_cd}[ZHAO]')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_e_zhao),ls='--',color='green',label=fr'$\alpha$={ajust_e_zhao[0]:.3f}$\pm${np.sqrt(cov_e_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_zhao[1]:.3f}$\pm${np.sqrt(cov_e_zhao[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_cd_zhao),ls='--',color='red',label=fr'$\alpha$={ajust_cd_zhao[0]:.3f}$\pm${np.sqrt(cov_cd_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_cd_zhao[1]:.3f}$\pm${np.sqrt(cov_cd_zhao[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	ax_center.set_ylabel(r'$\log M_{\bigstar}$')
	label_2=['cD[ZHAO]','E[ZHAO]']
	cor_2=['green','red']
	for i,dist in enumerate(vec_kde_m200_zhao):
		ax_topx.plot(m200_linspace,dist(m200_linspace),color=cor_2[i],label=f'{label_2[i]}')
		ax_topx.axvline(vec_med_m200_zhao[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_m200_zhao[i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_starmass_zhao):
		ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cor_2[i],label=f'{label_2[i]}')
		ax_righty.axhline(vec_med_starmass_zhao[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_starmass_zhao[i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_m200_zhao.png')
	plt.close()

	#JOINTPLOT DA MASSA x M200 -- NOSSA CLASSIFICAÇÃO -- cDS[cD & E(EL)], E

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e,cov_e=np.polyfit(m200_e,starmass_e,1,cov=True)
	ajust_cd,cov_cd=np.polyfit(m200_cd,starmass_cd,1,cov=True)

	ax_center.scatter(m200_e,starmass_e,marker='o',color='green',edgecolor='black',label=label_elip)
	ax_center.scatter(m200_cd,starmass_cd,marker='o',edgecolor='black',label=label_cd,color='red')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_e),color='green',label=fr'$\alpha$={ajust_e[0]:.3f}$\pm${np.sqrt(cov_e[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e[1]:.3f}$\pm${np.sqrt(cov_e[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_cd),color='red',label=fr'$\alpha$={ajust_cd[0]:.3f}$\pm${np.sqrt(cov_cd[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_cd[1]:.3f}$\pm${np.sqrt(cov_cd[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	ax_center.set_ylabel(r'$\log M_{\bigstar}$')
	vec_kde_m200_dual=m200_kde_cd,m200_kde_e
	vec_med_m200_dual=med_m200_cd,med_m200_e
	label_2=['cD','E']
	for i,dist in enumerate(vec_kde_m200_dual):
		ax_topx.plot(m200_linspace,dist(m200_linspace),color=cor_2[i],label=f'{label_2[i]}')
		ax_topx.axvline(vec_med_m200_dual[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_m200_dual[i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	vec_kde_starmass_dual=starmass_kde_cd,starmass_kde_e
	vec_med_starmass_dual=med_starmass_cd,med_starmass_e

	for i,dist in enumerate(vec_kde_starmass_dual):
		ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cor_2[i],label=f'{label_2[i]}')
		ax_righty.axhline(vec_med_starmass_dual[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_starmass_dual[i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_m200_dual_nossa.png')
	plt.close()
	
	#JOINTPLOT DA MASSA x M200 -- cDS DO ZHAO PELA NOSSA CLASSIFICAÇÃO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e_cD,cov_e_cD=np.polyfit(m200_e_cD,starmass_e_cD,1,cov=True)
	ajust_small_cD,cov_small_cD=np.polyfit(m200_cd_small_cD,starmass_cd_small_cD,1,cov=True)
	ajust_big_cD,cov_big_cD=np.polyfit(m200_cd_big_cD,starmass_cd_big_cD,1,cov=True)

	ax_center.scatter(m200_e_cD,starmass_e_cD,marker='o',edgecolor='black',label=f'{label_elip}[cD]',color='green')
	ax_center.scatter(m200_cd_small_cD,starmass_cd_small_cD,marker='o',edgecolor='black',label=f'{label_elip_el}[cD]',color='blue')
	ax_center.scatter(m200_cd_big_cD,starmass_cd_big_cD,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_e_cD),color='green',label=fr'$\alpha$={ajust_e_cD[0]:.3f}$\pm${np.sqrt(cov_e_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_cD[1]:.3f}$\pm${np.sqrt(cov_e_cD[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_small_cD),color='blue',label=fr'$\alpha$={ajust_small_cD[0]:.3f}$\pm${np.sqrt(cov_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_small_cD[1]:.3f}$\pm${np.sqrt(cov_small_cD[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_big_cD),color='red',label=fr'$\alpha$={ajust_big_cD[0]:.3f}$\pm${np.sqrt(cov_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_big_cD[1]:.3f}$\pm${np.sqrt(cov_big_cD[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	ax_center.set_ylabel(r'$\log M_{\bigstar}$')

	for i,dist in enumerate(vec_kde_m200_cD_zhao[1:]):
		ax_topx.plot(m200_linspace,dist(m200_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}[cD]')
		ax_topx.axvline(vec_med_m200_cD_zhao[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_m200_cD_zhao[1:][i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_starmass_cD[1:]):
		ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cores[1:][i],label=f'{names_simples[1:][i]}[cD]')
		ax_righty.axhline(vec_med_starmass_cD[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[1:][i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_m200_cDs_zhao.png')
	plt.close()

	#####################################

	#JOINTPLOT DA MASSA x AGE -- CLASSIFICAÇÃO DO ZHAO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e_zhao,cov_e_zhao=np.polyfit(starmass_e_zhao,age_e_zhao,1,cov=True)
	ajust_cd_zhao,cov_cd_zhao=np.polyfit(starmass_cd_zhao,age_cd_zhao,1,cov=True)

	ax_center.scatter(starmass_e_zhao,age_e_zhao,marker='o',color='green',edgecolor='black',label=f'{label_elip}[ZHAO]')
	ax_center.scatter(starmass_cd_zhao,age_cd_zhao,marker='o',color='red',edgecolor='black',label=f'{label_cd}[ZHAO]')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_e_zhao),ls='--',color='green',label=fr'$\alpha$={ajust_e_zhao[0]:.3f}$\pm${np.sqrt(cov_e_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_zhao[1]:.3f}$\pm${np.sqrt(cov_e_zhao[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_cd_zhao),ls='--',color='red',label=fr'$\alpha$={ajust_cd_zhao[0]:.3f}$\pm${np.sqrt(cov_cd_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_cd_zhao[1]:.3f}$\pm${np.sqrt(cov_cd_zhao[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{\bigstar}$')
	ax_center.set_ylabel(r'$\tau$ (Gyr)')

	label_2=['cD[ZHAO]','E[ZHAO]']
	cor_2=['green','red']
	for i,dist in enumerate(vec_kde_starmass_zhao):
		ax_topx.plot(dist(starmass_linspace),starmass_linspace,color=cor_2[i],label=f'{label_2[i]}')
		ax_topx.axhline(vec_med_starmass_zhao[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_starmass_zhao[i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_age_zhao):
		ax_righty.plot(dist(age_linspace),age_linspace,color=cor_2[i],label=f'{label_2[i]}')
		ax_righty.axhline(vec_med_age_zhao[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_age_zhao[i]:.3f}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}_stats_observation_desi/test_ks/age_starmass_zhao.png')
	plt.close()

	#JOINTPLOT DA MASSA x AGE -- NOSSA CLASSIFICAÇÃO -- cDS[cD & E(EL)], E

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e,cov_e=np.polyfit(starmass_e,age_e,1,cov=True)
	ajust_cd,cov_cd=np.polyfit(starmass_cd,age_cd,1,cov=True)

	ax_center.scatter(starmass_e,age_e,marker='o',color='green',edgecolor='black',label=label_elip)
	ax_center.scatter(starmass_cd,age_cd,marker='o',edgecolor='black',label=label_cd,color='red')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_e),color='green',label=fr'$\alpha$={ajust_e[0]:.3f}$\pm${np.sqrt(cov_e[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e[1]:.3f}$\pm${np.sqrt(cov_e[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_cd),color='red',label=fr'$\alpha$={ajust_cd[0]:.3f}$\pm${np.sqrt(cov_cd[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_cd[1]:.3f}$\pm${np.sqrt(cov_cd[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{\bigstar}$')
	ax_center.set_ylabel(r'$\tau$ (Gyr)')

	vec_kde_starmass_dual=starmass_kde_cd,starmass_kde_e
	vec_med_starmass_dual=med_starmass_cd,med_starmass_e

	for i,dist in enumerate(vec_kde_starmass_dual):
		ax_topx.plot(dist(starmass_linspace),starmass_linspace,color=cor_2[i],label=f'{label_2[i]}')
		ax_topx.axhline(vec_med_starmass_dual[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_starmass_dual[i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	vec_kde_age_dual=age_kde_cd,age_kde_e
	vec_med_age_dual=med_age_cd,med_age_e
	label_2=['cD','E']
	for i,dist in enumerate(vec_kde_age_dual):
		ax_righty.plot(age_linspace,dist(age_linspace),color=cor_2[i],label=f'{label_2[i]}')
		ax_righty.axvline(vec_med_age_dual[i],color=cor_2[i],ls='--',label=fr'$\mu = {vec_med_age_dual[i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_age_dual_nossa.png')
	plt.close()
	
	#JOINTPLOT DA MASSA x AGE -- cDS DO ZHAO PELA NOSSA CLASSIFICAÇÃO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_e_cD,cov_e_cD=np.polyfit(starmass_e_cD,age_e_cD,1,cov=True)
	ajust_small_cD,cov_small_cD=np.polyfit(starmass_cd_small_cD,age_cd_small_cD,1,cov=True)
	ajust_big_cD,cov_big_cD=np.polyfit(starmass_cd_big_cD,age_cd_big_cD,1,cov=True)

	ax_center.scatter(starmass_e_cD,age_e_cD,marker='o',edgecolor='black',label=f'{label_elip}[cD]',color='green')
	ax_center.scatter(starmass_cd_small_cD,age_cd_small_cD,marker='o',edgecolor='black',label=f'{label_elip_el}[cD]',color='blue')
	ax_center.scatter(starmass_cd_big_cD,age_cd_big_cD,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_e_cD),color='green',label=fr'$\alpha$={ajust_e_cD[0]:.3f}$\pm${np.sqrt(cov_e_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_cD[1]:.3f}$\pm${np.sqrt(cov_e_cD[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_small_cD),color='blue',label=fr'$\alpha$={ajust_small_cD[0]:.3f}$\pm${np.sqrt(cov_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_small_cD[1]:.3f}$\pm${np.sqrt(cov_small_cD[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_big_cD),color='red',label=fr'$\alpha$={ajust_big_cD[0]:.3f}$\pm${np.sqrt(cov_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_big_cD[1]:.3f}$\pm${np.sqrt(cov_big_cD[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{\bigstar}$')
	ax_center.set_ylabel(r'$\tau$ (Gyr)')

	for i,dist in enumerate(vec_kde_starmass_cD[1:]):
		ax_topx.plot(dist(starmass_linspace),starmass_linspace,color=cores[1:][i],label=f'{names_simples[1:][i]}[cD]')
		ax_topx.axhline(vec_med_starmass_cD[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[1:][i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_age_cD[1:]):
		ax_righty.plot(age_linspace,dist(age_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}[cD]')
		ax_righty.axvline(vec_med_age_cD[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_age_cD[1:][i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/starmass_age_cDs_zhao.png')
	plt.close()

	####################################
	bt_cd_zhao,bt_e_zhao=bt_mass[cd_cut_casjobs],bt_mass[e_cut_casjobs]
	bt_cd_cD,bt_cd_small_cD,bt_cd_big_cD,bt_e_cD=bt_mass[cd_lim_casjobs & cd_cut_casjobs],bt_mass[cd_lim_casjobs_small & cd_cut_casjobs],bt_mass[cd_lim_casjobs_big & cd_cut_casjobs],bt_mass[elip_lim_casjobs & cd_cut_casjobs]
	##
	mass_c1_e_zhao=np.log10(np.multiply(bt_e_zhao,np.power(10,starmass_e_zhao)))
	mass_c2_e_zhao=np.log10(np.multiply((1-bt_e_zhao),np.power(10,starmass_e_zhao)))

	mass_c1_e_cD=np.log10(np.multiply(bt_e_cD,np.power(10,starmass_e_cD)))
	mass_c2_e_cD=np.log10(np.multiply((1-bt_e_cD),np.power(10,starmass_e_cD)))
	##
	mass_c1_cd_small_cD=np.log10(np.multiply(bt_cd_small_cD,np.power(10,starmass_cd_small_cD)))
	mass_c2_cd_small_cD=np.log10(np.multiply((1-bt_cd_small_cD),np.power(10,starmass_cd_small_cD)))

	mass_c1_cd_big_cD=np.log10(np.multiply(bt_cd_big_cD,np.power(10,starmass_cd_big_cD)))
	mass_c2_cd_big_cD=np.log10(np.multiply((1-bt_cd_big_cD),np.power(10,starmass_cd_big_cD)))
	##	
	mass_c1_cd_zhao=np.log10(np.multiply(bt_cd_zhao,np.power(10,starmass_cd_zhao)))
	mass_c2_cd_zhao=np.log10(np.multiply((1-bt_cd_zhao),np.power(10,starmass_cd_zhao)))

	mass_c1_cd_cD=np.log10(np.multiply(bt_cd_cD,np.power(10,starmass_cd_cD)))
	mass_c2_cd_cD=np.log10(np.multiply((1-bt_cd_cD),np.power(10,starmass_cd_cD)))
	##
	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO -- SEM CORREÇÃO-cD Zhao - nossa cD vs E(EL)')

	ajust_c1_small_cD,cov_c1_small_cD=np.polyfit(m200_cd_small_cD,mass_c1_cd_small_cD,1,cov=True)
	ajust_c1_big_cD,cov_c1_big_cD=np.polyfit(m200_cd_big_cD,mass_c1_cd_big_cD,1,cov=True)

	ajust_c2_small_cD,cov_c2_small_cD=np.polyfit(m200_cd_small_cD,mass_c2_cd_small_cD,1,cov=True)
	ajust_c2_big_cD,cov_c2_big_cD=np.polyfit(m200_cd_big_cD,mass_c2_cd_big_cD,1,cov=True)

	axs[0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD,marker='o',edgecolor='black',alpha=0.6,label='True cD[cD]',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c1_cD.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE EXTERNO -- SEM CORREÇÃO-cD Zhao- nossa cD vs E(EL)')

	axs[0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c2_cD.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('ELIPTICAS X COMPONENTE EXTERNO EL - cDs do ZHAO')

	axs[0].scatter(m200_e_cD,starmass_e_cD,marker='o',edgecolor='black',label=f'{label_elip}[cD]',alpha=0.6,color='green')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_e_cD),color='black',ls='--',label=fr'$\alpha$={ajust_e_cD[0]:.3f}$\pm${np.sqrt(cov_e_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_cD[1]:.3f}$\pm${np.sqrt(cov_e_cD[1,1]):.3f}')

	axs[1].scatter(m200_cd_small_cD,mass_c2_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_elip_elc2_cD.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - cDs do ZHAO')

	axs[0,0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps_cD.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (SUPERIORES) -- EXTERNO (INFERIORES) - comparação - nossa cD vs cD zhao')

	ajust_c1_cd,cov_c1_cd=np.polyfit(m200_cd,mass_c1_cd,1,cov=True)
	ajust_c2_cd,cov_c2_cd=np.polyfit(m200_cd,mass_c2_cd,1,cov=True)

	ajust_c1_cd_zhao,cov_c1_cd_zhao=np.polyfit(m200_cd_zhao,mass_c1_cd_zhao,1,cov=True)
	ajust_c2_cd_zhao,cov_c2_cd_zhao=np.polyfit(m200_cd_zhao,mass_c2_cd_zhao,1,cov=True)

	axs[0,0].scatter(m200_cd,mass_c1_cd,marker='o',edgecolor='black',alpha=0.6,label='cD[Kaipper]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_cd),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_cd[0]:.3f}$\pm${np.sqrt(cov_c1_cd[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_cd[1]:.3f}$\pm${np.sqrt(cov_c1_cd[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_zhao,mass_c1_cd_zhao,marker='o',edgecolor='black',label='cD[Zhao]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_cd_zhao),color='black',ls='-',label=fr'$\alpha$={ajust_c1_cd_zhao[0]:.3f}$\pm${np.sqrt(cov_c1_cd_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_cd_zhao[1]:.3f}$\pm${np.sqrt(cov_c1_cd_zhao[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd,mass_c2_cd,marker='o',edgecolor='black',alpha=0.6,label='cD[Kaipper]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_cd),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_cd[0]:.3f}$\pm${np.sqrt(cov_c2_cd[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_cd[1]:.3f}$\pm${np.sqrt(cov_c2_cd[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_zhao,mass_c2_cd_zhao,marker='o',edgecolor='black',label='cD[Zhao]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_cd_zhao),color='black',ls='-',label=fr'$\alpha$={ajust_c2_cd_zhao[0]:.3f}$\pm${np.sqrt(cov_c2_cd_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_cd_zhao[1]:.3f}$\pm${np.sqrt(cov_c2_cd_zhao[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps_nos_zhao.png')
	plt.close()
	##################################################
	# BT -- CORRGIDA

	bt_cd_zhao_corr,bt_e_zhao_corr=bt_mass_corr[cd_cut_casjobs],bt_mass_corr[e_cut_casjobs]
	bt_cd_cD_corr,bt_cd_small_cD_corr,bt_cd_big_cD_corr,bt_e_cD_corr=bt_mass_corr[cd_lim_casjobs & cd_cut_casjobs],bt_mass_corr[cd_lim_casjobs_small & cd_cut_casjobs],bt_mass_corr[cd_lim_casjobs_big & cd_cut_casjobs],bt_mass_corr[elip_lim_casjobs & cd_cut_casjobs]
	##
	mass_c1_e_zhao_corr=np.log10(np.multiply(bt_e_zhao_corr,np.power(10,starmass_e_zhao)))
	mass_c2_e_zhao_corr=np.log10(np.multiply((1-bt_e_zhao_corr),np.power(10,starmass_e_zhao)))

	mass_c1_e_cD_corr=np.log10(np.multiply(bt_e_cD_corr,np.power(10,starmass_e_cD)))
	mass_c2_e_cD_corr=np.log10(np.multiply((1-bt_e_cD_corr),np.power(10,starmass_e_cD)))
	##
	mass_c1_cd_small_cD_corr=np.log10(np.multiply(bt_cd_small_cD_corr,np.power(10,starmass_cd_small_cD)))
	mass_c2_cd_small_cD_corr=np.log10(np.multiply((1-bt_cd_small_cD_corr),np.power(10,starmass_cd_small_cD)))

	mass_c1_cd_big_cD_corr=np.log10(np.multiply(bt_cd_big_cD_corr,np.power(10,starmass_cd_big_cD)))
	mass_c2_cd_big_cD_corr=np.log10(np.multiply((1-bt_cd_big_cD_corr),np.power(10,starmass_cd_big_cD)))
	##	
	mass_c1_cd_zhao_corr=np.log10(np.multiply(bt_cd_zhao_corr,np.power(10,starmass_cd_zhao)))
	mass_c2_cd_zhao_corr=np.log10(np.multiply((1-bt_cd_zhao_corr),np.power(10,starmass_cd_zhao)))

	mass_c1_cd_cD_corr=np.log10(np.multiply(bt_cd_cD_corr,np.power(10,starmass_cd_cD)))
	mass_c2_cd_cD_corr=np.log10(np.multiply((1-bt_cd_cD_corr),np.power(10,starmass_cd_cD)))
	##
	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO -- CORRIGIDA-cD Zhao - nossa cD vs E(EL)')

	ajust_c1_small_cD_corr,cov_c1_small_cD_corr=np.polyfit(m200_cd_small_cD,mass_c1_cd_small_cD_corr,1,cov=True)
	ajust_c1_big_cD_corr,cov_c1_big_cD_corr=np.polyfit(m200_cd_big_cD,mass_c1_cd_big_cD_corr,1,cov=True)

	ajust_c2_small_cD_corr,cov_c2_small_cD_corr=np.polyfit(m200_cd_small_cD,mass_c2_cd_small_cD_corr,1,cov=True)
	ajust_c2_big_cD_corr,cov_c2_big_cD_corr=np.polyfit(m200_cd_big_cD,mass_c2_cd_big_cD_corr,1,cov=True)

	axs[0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD_corr,marker='o',edgecolor='black',alpha=0.6,label='True cD[cD]',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c1_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE EXTERNO -- CORRIGIDA-cD Zhao- nossa cD vs E(EL)')

	axs[0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD_corr,marker='o',edgecolor='black',alpha=0.6,label='True cD',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_c2_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('ELIPTICAS X COMPONENTE EXTERNO EL - cDs do ZHAO - CORRIGIDA')

	axs[0].scatter(m200_e_cD,starmass_e_cD,marker='o',edgecolor='black',label=f'{label_elip}[cD]',alpha=0.6,color='green')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_e_cD),color='black',ls='--',label=fr'$\alpha$={ajust_e_cD[0]:.3f}$\pm${np.sqrt(cov_e_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e_cD[1]:.3f}$\pm${np.sqrt(cov_e_cD[1,1]):.3f}')

	axs[1].scatter(m200_cd_small_cD,mass_c2_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_elip_elc2_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - cDs do ZHAO - CORRIGIDA')

	axs[0,0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD_corr,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD_corr,marker='o',edgecolor='black',label='True cD[cD]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (SUPERIORES) -- EXTERNO (INFERIORES) - comparação - nossa cD vs cD zhao')

	ajust_c1_cd_corr,cov_c1_cd_corr=np.polyfit(m200_cd,mass_c1_cd_corr,1,cov=True)
	ajust_c2_cd_corr,cov_c2_cd_corr=np.polyfit(m200_cd,mass_c2_cd_corr,1,cov=True)

	ajust_c1_cd_zhao_corr,cov_c1_cd_zhao_corr=np.polyfit(m200_cd_zhao,mass_c1_cd_zhao_corr,1,cov=True)
	ajust_c2_cd_zhao_corr,cov_c2_cd_zhao_corr=np.polyfit(m200_cd_zhao,mass_c2_cd_zhao_corr,1,cov=True)

	axs[0,0].scatter(m200_cd,mass_c1_cd_corr,marker='o',edgecolor='black',alpha=0.6,label='cD[Kaipper]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_cd_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_cd_corr[0]:.3f}$\pm${np.sqrt(cov_c1_cd_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_cd_corr[1]:.3f}$\pm${np.sqrt(cov_c1_cd_corr[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_zhao,mass_c1_cd_zhao_corr,marker='o',edgecolor='black',label='cD[Zhao]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_cd_zhao_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_cd_zhao_corr[0]:.3f}$\pm${np.sqrt(cov_c1_cd_zhao_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_cd_zhao_corr[1]:.3f}$\pm${np.sqrt(cov_c1_cd_zhao_corr[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd,mass_c2_cd_corr,marker='o',edgecolor='black',alpha=0.6,label='cD[Kaipper]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_cd_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_cd_corr[0]:.3f}$\pm${np.sqrt(cov_c2_cd_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_cd_corr[1]:.3f}$\pm${np.sqrt(cov_c2_cd_corr[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_zhao,mass_c2_cd_zhao_corr,marker='o',edgecolor='black',label='cD[Zhao]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_cd_zhao_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_cd_zhao_corr[0]:.3f}$\pm${np.sqrt(cov_c2_cd_zhao_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_cd_zhao_corr[1]:.3f}$\pm${np.sqrt(cov_c2_cd_zhao_corr[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/m200_mass_comps_nos_zhao_corr.png')
	plt.close()

#
###########################################################
#LINHA DO H ALPHA
# h_line_cd,h_line_cd_small,h_line_cd_big,h_line_e=h_line[cd_lim_halpha],h_line[cd_lim_halpha_small],h_line[cd_lim_halpha_big],h_line[elip_lim_halpha]
# ks_h_line=ks_calc([h_line_cd,h_line_cd_small,h_line_cd_big,h_line_e])

# vec_ks_h_line=ks_h_line[0][1][2],ks_h_line[0][1][3],ks_h_line[0][2][3]
# vec_ks_h_line_2c=ks_h_line[0][0][1],ks_h_line[0][0][2],ks_h_line[0][0][3]

# kde_h_line=kde(h_line)
# h_line_factor = kde_h_line.factor
# h_line_linspace=np.linspace(min(h_line),4.5,3000)
# h_line_kde_cd,h_line_kde_cd_small,h_line_kde_cd_big,h_line_kde_e=kde(h_line_cd,bw_method=h_line_factor),kde(h_line_cd_small,bw_method=h_line_factor),kde(h_line_cd_big,bw_method=h_line_factor),kde(h_line_e,bw_method=h_line_factor)

# vec_med_h_line=med_h_line_cd,med_h_line_cd_small,med_h_line_cd_big,med_h_line_e=np.average(h_line_cd),np.average(h_line_cd_small),np.average(h_line_cd_big),np.average(h_line_e)
# vec_kde_h_line=h_line_kde_cd,h_line_kde_cd_small,h_line_kde_cd_big,h_line_kde_e

# os.makedirs(f'{sample}_stats_observation_desi/test_ks/h_line',exist_ok=True)

# fig,axs=plt.subplots(1,1,figsize=(10,5))
# plt.title('Linha do h_alpha')
# for i,dist in enumerate(vec_kde_h_line):
# 	axs.plot(h_line_linspace,dist(h_line_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
# 	axs.axvline(vec_med_h_line[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_h_line[i]:.3f}$')
# axs.legend()
# axs.set_xlabel(r'$H_\alpha$')
# info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_h_line[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_h_line[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_h_line[2]:.3e}')
# info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_h_line_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_h_line_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_h_line_2c[2]:.3e}')
# fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
# fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
# plt.savefig(f'{sample}_stats_observation_desi/test_ks/h_line/hline_kde.png')
# plt.close()

##############################################################
#RAZÃO BT
bt_cd,bt_cd_small,bt_cd_big,bt_e=bt_vec_12[cd_lim],bt_vec_12[lim_cd_small],bt_vec_12[lim_cd_big],bt_vec_12[elip_lim]
#
kde_bt=kde(bt_vec_12)
bt_factor = kde_bt.factor
bt_linspace=np.linspace(min(bt_vec_12),max(bt_vec_12),3000)

bt_kde_cd,bt_kde_cd_small,bt_kde_cd_big,bt_kde_e=kde(bt_cd,bw_method=bt_factor),kde(bt_cd_small,bw_method=bt_factor),kde(bt_cd_big,bw_method=bt_factor),kde(bt_e,bw_method=bt_factor)
vec_med_bt=med_bt_cd,med_bt_cd_small,med_bt_cd_big,med_bt_e=np.average(bt_cd),np.average(bt_cd_small),np.average(bt_cd_big),np.average(bt_e)

vec_kde_bt=bt_kde_cd,bt_kde_cd_small,bt_kde_cd_big

os.makedirs(f'{sample}_stats_observation_desi/test_ks/bt',exist_ok=True)

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('RAZÃO BT -- SEM CORREÇÃO')
for i,dist in enumerate(vec_kde_bt):
	axs.plot(bt_linspace,dist(bt_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_bt[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_bt[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$B/T$')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/bt/bt_kde.png')
plt.close()

#BT CORRIGIDA

bt_cd_corr,bt_cd_small_corr,bt_cd_big_corr,bt_e_corr=bt_vec_corr[cd_lim],bt_vec_corr[lim_cd_small],bt_vec_corr[lim_cd_big],bt_vec_corr[elip_lim]
#
kde_bt_corr=kde(bt_vec_corr)
bt_corr_factor = kde_bt_corr.factor
bt_corr_linspace=np.linspace(min(bt_vec_corr),max(bt_vec_corr),3000)
bt_kde_cd_corr,bt_kde_cd_small_corr,bt_kde_cd_big_corr,bt_kde_e_corr=kde(bt_cd_corr,bw_method=bt_corr_factor),kde(bt_cd_small_corr,bw_method=bt_corr_factor),kde(bt_cd_big_corr,bw_method=bt_corr_factor),kde(bt_e_corr,bw_method=bt_corr_factor)
vec_med_bt_corr=med_bt_cd_corr,med_bt_cd_small_corr,med_bt_cd_big_corr,med_bt_e_corr=np.average(bt_cd_corr),np.average(bt_cd_small_corr),np.average(bt_cd_big_corr),np.average(bt_e_corr)

vec_kde_bt_corr=bt_kde_cd_corr,bt_kde_cd_small_corr,bt_kde_cd_big_corr

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('RAZÃO BT -- corrigida')
for i,dist in enumerate(vec_kde_bt_corr):
	axs.plot(bt_linspace,dist(bt_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_bt_corr[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_bt_corr[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$B/T$')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/bt/bt_corr_kde.png')
plt.close()
#############################################
#CONCENTRÃÇO

conc_cd,conc_cd_small,conc_cd_big,conc_e=conc[cd_lim_casjobs],conc[cd_lim_casjobs_small],conc[cd_lim_casjobs_big],conc[elip_lim_casjobs]
ks_conc=ks_calc([conc_cd,conc_cd_small,conc_cd_big,conc_e])

vec_ks_conc=ks_conc[0][1][2],ks_conc[0][1][3],ks_conc[0][2][3]
vec_ks_conc_2c=ks_conc[0][0][1],ks_conc[0][0][2],ks_conc[0][0][3]

kde_conc=kde(conc)
conc_factor = kde_conc.factor
conc_linspace=np.linspace(min(conc),4,3000)
conc_kde_cd,conc_kde_cd_small,conc_kde_cd_big,conc_kde_e=kde(conc_cd,bw_method=conc_factor),kde(conc_cd_small,bw_method=conc_factor),kde(conc_cd_big,bw_method=conc_factor),kde(conc_e,bw_method=conc_factor)

vec_med_conc=med_conc_cd,med_conc_cd_small,med_conc_cd_big,med_conc_e=np.average(conc_cd),np.average(conc_cd_small),np.average(conc_cd_big),np.average(conc_e)
vec_kde_conc=conc_kde_cd,conc_kde_cd_small,conc_kde_cd_big,conc_kde_e

os.makedirs(f'{sample}_stats_observation_desi/test_ks/conc',exist_ok=True)

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Concentração')
for i,dist in enumerate(vec_kde_conc):
	axs.plot(conc_linspace,dist(conc_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_conc[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_conc[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$R_{90}/R_{50}$')
info_labels = (f'K-S(True cD,E(EL)) ={vec_ks_conc[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_conc[1]:.3e}\n' f'K-S(E,True cD)={vec_ks_conc[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_conc_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_conc_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_conc_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}_stats_observation_desi/test_ks/conc/conc_kde.png')
plt.close()

if sample=='L07':

	##slope - E,cD,E/cD DO ZHAO
	conc_cd_zhao,conc_e_zhao,conc_misc_zhao=conc[cd_cut_casjobs],conc[e_cut_casjobs],conc[(ecd_cut_casjobs | cde_cut_casjobs)]
	##slope - SUBGRUPO cD DO ZHAO
	conc_cd_cD,conc_cd_small_cD,conc_cd_big_cD,conc_e_cD=conc[cd_lim_casjobs & cd_cut_casjobs],conc[cd_lim_casjobs_small & cd_cut_casjobs],conc[cd_lim_casjobs_big & cd_cut_casjobs],conc[elip_lim_casjobs & cd_cut_casjobs]
	conc_e_E,conc_e_misc=conc[elip_lim_casjobs & e_cut_casjobs],conc[elip_lim_casjobs & (ecd_cut_casjobs | cde_cut_casjobs)]
	
	##slope - KDE
	vec_kde_conc_zhao=conc_kde_cd_zhao,conc_kde_e_zhao=kde(conc_cd_zhao,bw_method=conc_factor),kde(conc_e_zhao,bw_method=conc_factor)
	vec_kde_conc_cD=conc_kde_cd_cD,conc_kde_cd_small_cD,conc_kde_cd_big_cD,conc_kde_e_cD=kde(conc_cd_cD,bw_method=conc_factor),kde(conc_cd_small_cD,bw_method=conc_factor),kde(conc_cd_big_cD,bw_method=conc_factor),kde(conc_e_cD,bw_method=conc_factor)
	vec_kde_conc_e=conc_kde_e_cD,conc_kde_e_misc,conc_kde_e_E=kde(conc_e_cD,bw_method=conc_factor),kde(conc_e_misc,bw_method=conc_factor),kde(conc_e_E,bw_method=conc_factor)

	#MÉDIAS
	vec_med_conc_zhao=med_conc_cd_zhao,med_conc_e_zhao=np.average(conc_cd_zhao),np.average(conc_e_zhao)
	vec_med_conc_cD=med_conc_cd,med_conc_cd_small,med_conc_cd_big,med_conc_e=np.average(conc_cd_cD),np.average(conc_cd_small_cD),np.average(conc_cd_big_cD),np.average(conc_e_cD)
	vec_med_conc_E=med_conc_e_cD,med_conc_e_misc,med_conc_e_E=np.average(conc_e_cD),np.average(conc_e_misc),np.average(conc_e_E)
	#
	ks_conc_zhao=ks_2samp(conc_cd,conc_e)[1],ks_2samp(conc_cd_zhao,conc_e_zhao)[1]
	ks_conc_cD_zhao=ks_2samp(conc_cd_cD,conc_e_cD)[1],ks_2samp(conc_cd_big_cD,conc_e_cD)[1],ks_2samp(conc_cd_big_cD,conc_cd_small_cD)[1],ks_2samp(conc_e_cD,conc_cd_small_cD)[1]
	ks_conc_e_zhao=ks_2samp(conc_e_E,conc_e_misc)[1],ks_2samp(conc_e_cD,conc_e_E)[1],ks_2samp(conc_e_cD,conc_e_misc)[1]
	##
	# #MODELO SIMPLES - SLOPE

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Concentração - comparação cD vs E - Nossa vs Zhao')
	axs.plot(conc_linspace,conc_kde_e_zhao(conc_linspace),color='black',label='E[Zhao]')
	axs.axvline(np.average(conc_e_zhao),color='black',ls='--',label=fr'$\mu_E = {np.average(conc_e_zhao):.3f}$')
	axs.plot(conc_linspace,conc_kde_cd_zhao(conc_linspace),color='black',label='cD[Zhao]')
	axs.axvline(np.average(conc_cd_zhao),color='black',label=fr'$\mu_cD = {np.average(conc_cd_zhao):.3f}$')

	axs.plot(conc_linspace,conc_kde_e(conc_linspace),color='red',label='E[Kaipper]')
	axs.axvline(np.average(conc_e),color='red',ls='--',label=fr'$\mu_E = {np.average(conc_e):.3f}$')
	axs.plot(conc_linspace,conc_kde_cd(conc_linspace),color='red',label='cD[Kaipper]')
	axs.axvline(np.average(conc_cd),color='red',label=fr'$\mu_cD = {np.average(conc_cd):.3f}$')
	axs.legend()
	axs.set_xlabel(r'$R_{90}/R_{50}$')
	info_labels = (f'K-S(cD,E)[Zhao] ={ks_conc_zhao[0]:.3e}\n' f'K-S(cD,E)[Kaipper]={ks_conc_zhao[1]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/conc/conc_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Concentração - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_conc_cD):
		axs.plot(conc_linspace,dist(conc_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_conc_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_conc_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$R_{90}/R_{50}$')
	info_labels = (f'K-S(True cD,E(EL)) ={ks_conc_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_conc_cD_zhao[3]:.3e}\n' f'K-S(E,True cD)={ks_conc_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_conc_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/conc/conc_kde_cD_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Concentração - Elipticas - morfolofia do zhao')
	for i,dist in enumerate(vec_kde_conc_e):
		axs.plot(conc_linspace,dist(conc_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_conc_E[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_conc_E[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$R_{90}/R_{50}$')
	info_labels = (f'K-S(E/cD,E) ={ks_conc_e_zhao[0]:.3e}\n' f'K-S(E,cD)={ks_conc_e_zhao[1]:.3e}\n' f'K-S(E/cD,cD)={ks_conc_e_zhao[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}_stats_observation_desi/test_ks/conc/conc_kde_e_morf_zhao.png')
	plt.close()


#GRÁFICO JOINTPLOT DA RELAÇÃO DE KORMENDY

#SÉRSIC SIMPLES DAS CDS (SMALL E BIG) CONTRA ELIPTICAS
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_small,mue_sersic_cd_small,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_small, color='red', linestyle='-', label=linha_cd_label+' nuc\n'+fr'$\alpha={alpha_cd_label_small}$'+'\n'+fr'$\beta={beta_cd_label_small}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)
plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic_small.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_big,mue_sersic_cd_big,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_big, color='red', linestyle='-', label=linha_cd_label+' std\n'+fr'$\alpha={alpha_cd_label_big}$'+'\n'+fr'$\beta={beta_cd_label_big}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_big(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic_big.png')
plt.close()
###
#POR COMPONENTES EM RELAÇÃO A SÉRSIC

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
ax_center.plot(linspace_re, linha_interna, color='red', linestyle='--', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
ax_center.scatter(re_extern,mue_extern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='red')
ax_center.plot(linspace_re, linha_externa, color='red', linestyle='-.', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern),'.3')}$')
ax_topx.axvline(np.average(re_extern),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern),'.3')}$')
ax_righty.axhline(np.average(mue_extern),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample.png')
plt.close()

####
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo+' nuc',color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='red', linestyle='--', label=linha_interna_label+' nuc\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env+' nuc',color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_label+' nuc\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_small.png')
plt.close()
####
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo+' std',color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_label+' std\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env+' std',color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_label+' std\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(-0.28322921653227484,2.716867381942037)#x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_big.png')
plt.close()
###################################################################################
###################################################################################
###################################################################################
#INVESTIGAÇÃO DO P_VALUE

os.makedirs(f'{sample}_stats_observation_desi/p_value_delta_bic',exist_ok=True)

fig,axs=plt.subplots(1,3,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],color='green',edgecolor='black',alpha=0.9,label='E')
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],color='blue',edgecolor='black',alpha=0.9,label='E(EL)')
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],color='red',edgecolor='black',alpha=0.9,label='True cD')
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(loc='outside upper right',fontsize=9)

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_lines.png')
plt.close(fig)

#INDICE DE SÉRSIC - MODELO SIMPLES
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=n_sersic_e,edgecolor='black',vmax=8,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=n_sersic_cd_small,edgecolor='black',vmax=8,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=n_sersic_cd_big,edgecolor='black',vmax=8,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_n_simples.png')
plt.close(fig)

#RAZÃO AXIAL SÉRSIC - MODELO SIMPLES
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=axrat_sersic_e,edgecolor='black',alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=axrat_sersic_cd_small,edgecolor='black',alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=axrat_sersic_cd_big,edgecolor='black',alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$q$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_axrat.png')
plt.close()

#BOXINESS
plt.close(fig)
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=box_sersic_e,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=box_sersic_cd_small,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=box_sersic_cd_big,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$a_4/a$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_box.png')
plt.close(fig)

#B/T
fig,axs=plt.subplots(1,3,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=bt_vec_12[elip_lim],edgecolor='black',alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=bt_vec_12[lim_cd_small],edgecolor='black',alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=bt_vec_12[lim_cd_big],edgecolor='black',alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$B/T$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_bt.png')
plt.close(fig)

#RAZÃO DE RAIOS
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=re_ratio_12[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap_r)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=re_ratio_12[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap_r)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=re_ratio_12[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap_r)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$Re_1/Re_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_re_ratio.png')
plt.close(fig)

#RAZÃO DE INDICES DE SÉRSIC
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=n_ratio_12[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n_1/n_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_n_ratio.png')
plt.close(fig)

#RAZÃO DE RAZÃO AXIAL
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=(e1/e2)[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=(e1/e2)[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=(e1/e2)[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$q_1/q_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_q_ratio.png')
plt.close(fig)

r'''
fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
sc=axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,vmin=0.0,vmax=1,c=bt_vec_12[cd_lim],cmap=cmap)
axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
# axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
# axs.plot(linspace_re, linha_externa, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label('B/T')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
plt.close()

fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(7,7))
axs.scatter(re_ratio_12,n_ratio_12,color='white',edgecolor='black',alpha=0.2,label='WHL')
axs.scatter(re_ratio_12[high_lim_cd],n_ratio_12[high_lim_cd],color='red',edgecolor='black',alpha=0.4,label='n1>2.5')
axs.scatter(re_ratio_12[low_lim_cd],n_ratio_12[low_lim_cd],color='blue',edgecolor='black',alpha=0.6,label='n1<2.5')
axs.scatter(re_ratio_12[higher_lim_cd & (re_1_kpc<0.5)],n_ratio_12[higher_lim_cd & (re_1_kpc<0.5)],color='orange',edgecolor='black',alpha=1,label='n1>9.5')
axs.legend()
axs.set_xlabel(r'$Re_1/Re_2$')
axs.set_ylabel(r'$n_1/n_2$')
axs.set_xlim(-0.1,1.1)


# plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic_lines.png')
plt.close(fig)

xp1,xp2=min(p_value),max(p_value)
xz_p=np.linspace(0,1,len(delta_bic_obs))

fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(7,7))
axs.scatter(delta_bic_obs,p_value,color='white',edgecolor='black',alpha=0.9)
axs.axhline(0.32,label=r'$1 \sigma$')
axs.plot(log_model(xz,*pov),xz_p,color='black',label=r'$\Delta BIC$ $MAD_{line}$')
axs.legend()
axs.set_xlabel(r'$\Delta BIC$')
axs.set_ylabel(r'$P_{value}$')
axs.set_xlim(2000,-8000)
axs.set_ylim(-0.01,1.01)

# plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic_lines.png')
plt.close(fig)
'''

###################################################################################
###################################################################################
###################################################################################
#SVC TESTER
r'''
var_name_vec=['ax_ratio','ax_ratio_1','ax_ratio_2','box','bt','bt_corr','chi2_ratio','n1','n_sersic','q1q2','n1n2','rff_ratio','re1re2','n2','box1','box2','ass','rff','eta']
par_vec=[e_s,e1,e2,box_s,bt_vec_12,bt_vec_corr,chi2_ratio,n1,n_s,axrat_ratio_12,n_ratio_12,rff_ratio,np.log10(re_ratio_12),n2,box1,box2,ass,np.log10(rff_s),eta]

vec_espec_photutils=['slope_a3','slope_a4','slope_b3','slope_b4','test_gr','grad_e','grad_pa']
par_vec_photutils=[slope_a3,slope_a4,slope_b3,slope_b4,slope_gr,grad_e,grad_pa]

vec_espec_casjobs=['conc','magabs','starmass','age']
par_vec_casjobs=[conc_temp,magabs_temp,starmass_temp,age_temp]

# var_name_vec.extend(vec_espec_photutils)
# par_vec.extend(par_vec_photutils)
# svc_lim=cd_lim & lim_photutils & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}_stats_observation_desi/svc_score_el_cd_dupla_{sample}_photutils.dat'

svc_lim=cd_lim & np.isfinite(np.log10(rff_s))
save_file=f'{sample}_stats_observation_desi/svc_score_el_cd_dupla_{sample}.dat'

# var_name_vec.extend(vec_espec_casjobs)
# par_vec.extend(par_vec_casjobs)
# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}_stats_observation_desi/svc_score_el_cd_dupla_{sample}_casjobs.dat'

out=open(save_file,'w')
vec_comb=combinations(var_name_vec,2)
for i,dupla in enumerate(vec_comb):
	name1,name2=dupla
	ind1,ind2=var_name_vec.index(name1),var_name_vec.index(name2)
	par_temp=[par_vec[ind1][svc_lim],par_vec[ind2][svc_lim]]

	vec_x_el_cd=np.column_stack(par_temp)
	split_cd_el=svc_calc_dupla(vec_x_el_cd,lim_cd_small[svc_lim].astype(int).T,dupla,save_file)
########################################################
########################################################

# var_name_vec.extend(vec_espec_photutils)
# par_vec.extend(par_vec_photutils)
# svc_lim=cd_lim & lim_photutils & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}_stats_observation_desi/svc_score_el_cd_{sample}_photutils.dat'

# var_name_vec.extend(vec_espec_casjobs)
# par_vec.extend(par_vec_casjobs)
# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}_stats_observation_desi/svc_score_el_cd_{sample}_casjobs.dat'
#bt_corr re1re2 magabs 0.8805970149253731

# out=open(save_file,'w')
# vec_comb=combinations(var_name_vec,3)
# for i,trio in enumerate(vec_comb):
# 	name1,name2,name3=trio
# 	ind1,ind2,ind3=var_name_vec.index(name1),var_name_vec.index(name2),var_name_vec.index(name3)
# 	par_temp=[par_vec[ind1][svc_lim],par_vec[ind2][svc_lim],par_vec[ind3][svc_lim]]

# 	vec_x_el_cd=np.column_stack(par_temp)
# 	split_cd_el=svc_calc_trio(vec_x_el_cd,lim_cd_small[svc_lim].astype(int).T,trio,save_file)
###############################################
'''
#HISTOGRAMA DE RFF COM A LINHA DE CORTE

x0=np.linspace(0,0.1,10000)
if sample == 'WHL':
	bins = np.arange(0,0.1,0.0005)
	dlpov=[1.54622724e-02,4.61969720e-03,1.23642039e+02,-3.93661732e+00,-4.81476825e-01,7.57926648e+01]
	converter=(7057*0.0005)
elif sample == 'L07':
	bins = np.arange(0,0.1,0.0017)
	dlpov=[0.017835,0.005812,42.214771,-3.522754,0.328877,14.599428]
	converter=1
ynorm=[]
idx_split_rff = np.argwhere(np.diff(np.sign(diff(x0,dlpov)))).flatten()
fig0=plt.figure()#(figsize=(9,7))
y,x,_ = plt.hist(rff_s,bins=bins,density=True,histtype='step',color='white',edgecolor='black')
x = (x[1:]+x[:-1])/2
ll=y!=0.0
y=y[ll]
x=x[ll]
sigma_y = np.sqrt(y/converter)
ynorm=np.power(y-(dlognorm(x,*dlpov)/converter),2)/(sigma_y**2)
chi2_norm=np.sum(ynorm)/(len(y)-7)

dlog_norm_conv=dlognorm(x0,*dlpov)/converter
gauss_conv=gauss(x0,*dlpov[:3])/converter
log_norm_conv=lognormal(x0,*dlpov[3:6])/converter
plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
plt.plot(x0,gauss_conv,linewidth=1,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]/converter),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
plt.plot(x0,log_norm_conv,linewidth=1,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]/converter),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlabel('RFF')
plt.ylabel('Objetos')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines.png')
plt.close(fig0)

#HISTOGRAMA DO RFF PARA ELIPTICAS
fig0=plt.figure(figsize=(9,7))
plt.hist(rff_s[elip_lim],bins=bins,color='green',edgecolor='black',alpha=0.4,label='E')
plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlabel('RFF')
plt.ylabel('Objetos')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip.png')
plt.close(fig0)

#HISTOGRAMA DO RFF PARA ELIPTICAS - EXTRA LIGHT

fig0=plt.figure(figsize=(9,7))
plt.hist(rff_s[lim_cd_small],bins=bins,color='blue',edgecolor='black',alpha=0.4,label='E(EL)')
plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlabel('RFF')
plt.ylabel('Objetos')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_el.png')
plt.close(fig0)

#HISTOGRAMA DO RFF PARA TRUE cD
fig0=plt.figure(figsize=(9,7))
plt.hist(rff_s[lim_cd_big],bins=bins,color='red',alpha=0.4,edgecolor='black',label='True cD')
plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlabel('RFF')
plt.ylabel('Objetos')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_true_cd.png')
plt.close(fig0)

if sample == 'L07':

	#HISTOGRAMA DO RFF PARA cDs do Zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[cd_cut],bins=bins,color='red',edgecolor='black',alpha=0.4,label='cD[Zhao]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_cDs_zhao.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA ELIPTICAS do Zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[e_cut],bins=bins,color='green',edgecolor='black',alpha=0.4,label='E[Zhao]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_zhao.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA ELIPTICAS -- subgrupo cD do zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[elip_lim & cd_cut],bins=bins,color='green',edgecolor='red',alpha=0.4,label='E[cD]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_cd.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA ELIPTICAS -- subgrupo E do Zhao 

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[elip_lim & e_cut],bins=bins,color='green',edgecolor='green',alpha=0.4,label='E[E]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_e.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA ELIPTICAS -- subgrupo E/cD ou cD/E do Zhao 

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[elip_lim & (ecd_cut | cde_cut)],bins=bins,color='green',edgecolor='black',alpha=0.4,label='E[E/cD]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_misc.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA TRUE cD -- subgrupo cd do Zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[lim_cd_big & cd_cut],bins=bins,color='red',alpha=0.4,edgecolor='black',label='True cD[cD]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_true_cd_cd.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA E(EL) -- subgrupo cd do Zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[lim_cd_small & cd_cut],bins=bins,color='blue',alpha=0.4,edgecolor='black',label='E(EL)[cD]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_el_cd.png')
	plt.close(fig0)

#########################################
#PLANO DE RFF - ETA

x=np.linspace(0.001,0.3,1000)
y=np.linspace(0.001,0.3,1000)
limx=[-2.5,-0.5]
limy=[-0.02,0.1]
#RFF - ETA / PONTOS COLORIDOS PELOS GRUPOS E,E(EL),cD
fig1=plt.figure(figsize=(9,7))
vec=[0.1,0.5,0.9]
for item in vec:
	plt.plot(np.log10(x),x-item*x,label=str(item))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c='green',edgecolors='black',label='E')
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',label='True cD')
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c='blue',edgecolors='black',label='E(EL)')
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_color_coded.png')
plt.close()
#SUBPLOTS

fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,10))
vec=[0.1,0.5,0.9]
for item in vec:
	axs[0].plot(np.log10(x),x-item*x,label=str(item))
	axs[1].plot(np.log10(x),x-item*x,label=str(item))
	axs[2].plot(np.log10(x),x-item*x,label=str(item))

axs[0].scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(rff_s)[elip_lim],y=eta[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs[0])
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_ylabel(r'$\eta$')
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].legend()
axs[0].axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')

axs[1].scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_big],y=eta[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(r'$\log\,RFF$')
axs[1].axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs[1].legend()

axs[2].scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_small],y=eta[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs[2].set_xlabel(r'$\log\,RFF$')
axs[2].legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_sub_grupos.png')
plt.close()

#UNITÁRIOS

fig,axs=plt.subplots(1,1,figsize=(9,7))
axs.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(rff_s)[elip_lim],y=eta[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_xlim(limx)
axs.set_ylim(limy)
axs.set_ylabel(r'$\eta$')
axs.set_xlabel(r'$\log\,RFF$')
axs.legend()
axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7))
axs.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_big],y=eta[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs.set_xlim(limx)
axs.set_ylim(limy)
axs.set_ylabel(r'$\eta$')
axs.set_xlabel(r'$\log\,RFF$')
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_true_cds.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7))
axs.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_small],y=eta[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs.legend()
axs.set_xlim(limx)
axs.set_ylim(limy)
axs.set_ylabel(r'$\eta$')
axs.set_xlabel(r'$\log\,RFF$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_extra_light.png')
plt.close()

if sample == 'L07':
	#ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[e_cut],eta[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
	sns.kdeplot(x=np.log10(rff_s)[e_cut],y=eta[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_elipticas_zhao.png')
	plt.close()

	#cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[cd_cut],eta[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
	sns.kdeplot(x=np.log10(rff_s)[cd_cut],y=eta[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_cds_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[elip_lim & e_cut],eta[elip_lim & e_cut],c='green',edgecolors='black',alpha=0.2,label='E[E]')
	sns.kdeplot(x=np.log10(rff_s)[elip_lim & e_cut],y=eta[elip_lim & e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_elipticas_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[elip_lim & cd_cut],eta[elip_lim & cd_cut],c='green',edgecolors='red',alpha=0.2,label='E[cD]')
	sns.kdeplot(x=np.log10(rff_s)[elip_lim & cd_cut],y=eta[elip_lim & cd_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_elipticas_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[elip_lim & (ecd_cut | cde_cut)],eta[elip_lim & (ecd_cut | cde_cut)],c='green',edgecolors='black',alpha=0.2,label='E[E/cD]')
	sns.kdeplot(x=np.log10(rff_s)[elip_lim & (ecd_cut | cde_cut)],y=eta[elip_lim & (ecd_cut | cde_cut)],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_elipticas_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[lim_cd_big & cd_cut],eta[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='True cD[cD]')
	sns.kdeplot(x=np.log10(rff_s)[lim_cd_big & cd_cut],y=eta[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_true_cds_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[lim_cd_small & cd_cut],eta[lim_cd_small & cd_cut],c='blue',edgecolors='black',alpha=0.2,label='E(EL)[cD]')
	sns.kdeplot(x=np.log10(rff_s)[lim_cd_small & cd_cut],y=eta[lim_cd_small & cd_cut],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	axs.legend()
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rffxeta_extra_light_cd.png')
	plt.close()

###############################################
#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
vec=[0.1,0.5,0.9]
for item in vec:
	plt.plot(np.log10(x),x-item*x,label=str(item))
plt.scatter(np.log10(rff_s),eta,c=bt_vec_12,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt.png')
plt.close()

#MAPA DE COR BT - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=bt_e,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=bt_cd_big,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,bt_vec_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$B/T$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_bt_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=bt_vec_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=bt_vec_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=bt_vec_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_el_cd.png')
	plt.close()

##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')

plt.scatter(np.log10(rff_s),eta,c=bt_vec_corr,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_true_cd_corr.png')
plt.close()

fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=bt_e_corr,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_elip_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,bt_vec_corr
vec_label=r'$\log\,RFF$',r'$\eta$',r'$B/T$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_bt_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=bt_vec_corr[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=bt_vec_corr[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=bt_vec_corr[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_corr[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_el_cd.png')
	plt.close()

###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=rff_ratio,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio.png')
plt.close()

#MAPA DE RAZÃO DE RFF - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=rff_ratio_e,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=rff_ratio_cd_big,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=rff_ratio_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],rff_ratio_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],rff_ratio_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],rff_ratio_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,rff_ratio
vec_label=r'$\log\,RFF$',r'$\eta$',r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_rff_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_rff_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=rff_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=rff_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=rff_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=rff_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=chi2_ratio,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=chi2_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.9,1.5)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,chi2_ratio
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=limx,limy,[0.9,1.5]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_chi2_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_chi2_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=chi2_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=chi2_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=chi2_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=chi2_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=delta_bic_obs,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic.png')
plt.close()

#MAPA DE COR DELTA BIC - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=delta_bic_obs[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(1000,-3000)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,delta_bic_obs
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\Delta BIC$'
xlim,ylim,zlim=limx,limy,[1000,-3000]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_delta_bic_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_delta_bic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=delta_bic_obs[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=delta_bic_obs[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=delta_bic_obs[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=delta_bic_obs[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=delta_bic_obs[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=e_s,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=axrat_sersic_e,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,1.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,e_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q$'
xlim,ylim,zlim=limx,limy,[0.5,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_ax_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_ax_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=e_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=e_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=e_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=e_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=e_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=axrat_ratio_12,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=axrat_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,axrat_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_1/q_2$'
xlim,ylim,zlim=limx,limy,[0.1,2.8]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_q_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_q_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=axrat_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=axrat_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=axrat_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=axrat_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=axrat_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s),eta,c=box_s,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=box_sersic_e,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-0.5,0.5)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,box_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$a_4/a$'
xlim,ylim,zlim=limx,limy,[-0.5,0.5]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_box_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_box_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=box_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=box_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=box_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=box_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=box_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO - re1/re2
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s),eta,c=np.log10(re_ratio_12),edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=np.log10(re_ratio_12[elip_lim]),edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=np.log10(re_ratio_12[lim_cd_big]),edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=np.log10(re_ratio_12[lim_cd_small]),edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_el.png')
plt.close()

#ANALISE 3D - RE1/RE2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-2,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[elip_lim]),np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-2,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-2,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - RE1/RE2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-2,1)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-2,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-2,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,re_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\log_{10}({R_e}_{1}/{R_e}_{2})$'
xlim,ylim,zlim=limx,limy,[-2,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_re_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_re_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)


if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=re_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=re_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=re_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=re_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=re_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=re_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=re_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_el_cd.png')
	plt.close()

################################
#MAPA DE COR RAZÃO - n1/n2
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=n_ratio_12,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=n_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,5.)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_1/{n}_{2})$'
xlim,ylim,zlim=limx,limy,[0,5]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_n_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_n_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=n_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=n_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=n_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=n_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=n_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_el_cd.png')
	plt.close()

################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=n_s,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=n_sersic_e,edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,12)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,12])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,12])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,n_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n$'
xlim,ylim,zlim=limx,limy,[0.5,12]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_n_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_n_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=n_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=n_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=n_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=n_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=n_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/n1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=n1,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=n1[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,15.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,15.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,15.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,15.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,15.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,15.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n1
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_1$'
xlim,ylim,zlim=limx,limy,[0.5,15]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_n1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_n1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=n1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=n1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=n1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=n1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=n1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - INDICE DE SÉRSIC 2 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=n2,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=n2[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n2[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n2[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n2[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim(0.5,10.)
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n2
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_2$'
xlim,ylim,zlim=limx,limy,[0.5,10]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_n2_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_n2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=n2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=n2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=n2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=n2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=n2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=e1,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=e1[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,e1
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_1$'
xlim,ylim,zlim=limx,limy,[0.,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_q1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_q1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=e1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=e1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=e1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=e1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=e1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s),eta,c=e2,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=e2[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,e2
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_2$'
xlim,ylim,zlim=limx,limy,[0.,1]
save_place=f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_q2_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_q2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=e2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=e2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=e2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=e2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=e2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}_stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_el_cd.png')
	plt.close()

###################################################################################
###################################################################################
###################################################################################
#PLANO DE RAZÃO DE RAIOS (RE1/RE2) X N_2

label_x=r'$R_{1}/R_{2}$'
label_y=r'$n_{2}$'

fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
axs[0].scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolor='black',label='E')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
axs[1].scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolor='black',label='True cD')
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/scatter_re_ratio_n2_geral.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolor='black',label='E')
axs.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolor='black',label='True cD')
axs.set_xlabel(label_x)
axs.set_ylabel(label_y)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/scatter_re_ratio_n2_geral.png')
plt.close()

#SUBPLOTS

fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,10),constrained_layout=True)
axs[0].scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim],y=n2[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs[0])
axs[0].set_ylabel(label_y)
axs[0].set_xlabel(label_x)
axs[0].legend()

axs[1].scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big],y=n2[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(label_x)
axs[1].legend()

axs[2].scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small],y=n2[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_sub_grupos.png')
plt.close()

#UNITÁRIOS

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim],y=n2[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big],y=n2[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_true_cds.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small],y=n2[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.legend()
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_extra_light.png')
plt.close()

if sample == 'L07':
	#ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[e_cut],n2[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
	sns.kdeplot(x=np.log10(re_ratio_12)[e_cut],y=n2[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_elipticas_zhao.png')
	plt.close()

	#cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[cd_cut],n2[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
	sns.kdeplot(x=np.log10(re_ratio_12)[cd_cut],y=n2[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_cds_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & e_cut],n2[elip_lim & e_cut],c='green',edgecolors='black',alpha=0.2,label='E[E]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & e_cut],y=n2[elip_lim & e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_elipticas_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & cd_cut],n2[elip_lim & cd_cut],c='green',edgecolors='red',alpha=0.2,label='E[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & cd_cut],y=n2[elip_lim & cd_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_elipticas_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],n2[elip_lim & (ecd_cut | cde_cut)],c='green',edgecolors='black',alpha=0.2,label='E[E/cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],y=n2[elip_lim & (ecd_cut | cde_cut)],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_elipticas_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='True cD[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big & cd_cut],y=n2[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_true_cds_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c='blue',edgecolors='black',alpha=0.2,label='E(EL)[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small & cd_cut],y=n2[lim_cd_small & cd_cut],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.legend()
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rexn2_extra_light_cd.png')
	plt.close()

################################################
#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(np.log10(re_ratio_12),n2,c=bt_vec_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt.png')
plt.close()

#MAPA DE COR BT - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=bt_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=bt_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,bt_vec_12
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)


if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=bt_vec_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=bt_vec_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=bt_vec_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_el_cd.png')
	plt.close()
##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12),n2,c=bt_vec_corr,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_true_cd_corr.png')
plt.close()

fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=bt_e_corr,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_elip_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,bt_vec_corr
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)


if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=bt_vec_corr[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=bt_vec_corr[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=bt_vec_corr[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_corr[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_el_cd.png')
	plt.close()
###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=rff_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio.png')
plt.close()

#MAPA DE RAZÃO DE RFF - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=rff_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=rff_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=rff_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],rff_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],rff_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],rff_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,rff_ratio
vec_label=label_x,label_y,r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/re_n2_rff_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/re_n2_rff_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=rff_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=rff_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=rff_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=rff_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=chi2_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=chi2_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.9,1.5)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,chi2_ratio
vec_label=label_x,label_y,r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=None,None,[0.9,1.5]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/re_n2_chi2_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/re_n2_chi2_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=chi2_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=chi2_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=chi2_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=chi2_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=delta_bic_obs,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic.png')
plt.close()

#MAPA DE COR DELTA BIC - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=delta_bic_obs[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(1000,-3000)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,delta_bic_obs
vec_label=label_x,label_y,r'$\Delta BIC$'
xlim,ylim,zlim=None,None,[1000,-3000]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/re_n2_delta_bic_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/re_n2_delta_bic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=delta_bic_obs[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=delta_bic_obs[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=delta_bic_obs[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=delta_bic_obs[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=delta_bic_obs[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_el_cd.png')
	plt.close()
##############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=e_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=axrat_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,1.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,e_s
vec_label=label_x,label_y,r'$q$'
xlim,ylim,zlim=None,None,[0.5,1.]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/re_n2_ax_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/re_n2_ax_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/q1q2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=axrat_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=axrat_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_elip.png')
plt.close()


#MAPA DE COR RAZÃO Q1/Q2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,axrat_ratio_12
vec_label=label_x,label_y,r'$q_1/q_2$'
xlim,ylim,zlim=None,None,[0.1,2.8]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/q1q2/re_n2_q_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/q1q2/re_n2_q_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=axrat_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=axrat_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=axrat_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=axrat_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=axrat_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_el_cd.png')
	plt.close()
##############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=box_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=box_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(-0.5,0.5)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,box_s
vec_label=label_x,label_y,r'$a_4/a$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/box/re_n2_box_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/box/re_n2_box_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=box_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=box_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=box_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=box_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=box_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR RAZÃO - n1/n2
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/n1n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=n_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,5.])
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,5])
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,5.])
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,5])
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,n_ratio_12
vec_label=label_x,label_y,r'$n_1/n_2$'
xlim,ylim,zlim=None,None,[0.,5]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/n1n2/re_n2_n_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/n1n2/re_n2_n_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=n_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,12)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,12])
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,12])
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,n_s
vec_label=label_x,label_y,r'$n$'
xlim,ylim,zlim=None,None,[0.5,12]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/re_n2_n_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/re_n2_n_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/n1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=n1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,n1
vec_label=label_x,label_y,r'$n_1$'
xlim,ylim,zlim=None,None,[0.5,10]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/n1/re_n2_n1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/n1/re_n2_n1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_colotio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_colorio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_o_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=e1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=e1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,e1
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/re_n2_q1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/re_n2_q1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=e2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=e2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexeta_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,e2
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/re_n2_q2_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/re_n2_q2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_el_cd.png')
	plt.close()

###################################################################################
###################################################################################
###################################################################################
#PLANO DE N1 X NS
#PARA DIVIDIR AS cDS DAS E(EL)

os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns',exist_ok=True)

par_temp = np.column_stack([n1[cd_lim], n_s[cd_lim]])
coefs, inter, acc = svc_calc(par_temp, lim_cd_small[cd_lim].astype(int))

a, c = coefs
b = inter
# Plot
plt.figure()
plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='True cD')
xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
yplot = -(a*xplot + b) / c
lim_y=yplot<np.max(n_s[cd_lim])
m = -(a/c)
b_plot = -(b/c)
label_line = rf'$F_s={acc:.3f}$'+'\n'+rf'$\alpha={m:.3f}$'+'\n'+rf'$\beta={b_plot:.3f}$'
plt.plot(xplot[lim_y], yplot[lim_y], color='black', lw=3, label=label_line)
plt.legend()
plt.xlabel(r'$n_1$')
plt.ylabel(r'$n_s$')
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n1_ns_split.png')
plt.close()

plt.figure(figsize=(8, 7))
plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.1, c='blue', edgecolor='black', label='E(EL)')
plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='True cD')
sns.kdeplot(x=n1[cd_lim],y=n_s[cd_lim],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,bw_adjust=0.7,thresh=0.05)#thresh=0)
plt.plot([],[],color='black',label=r'$\rho$')

# GRADE DE X MAIS SUAVE
xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
yplot = -(a*xplot + b) / c
lim_y=yplot<np.max(n_s[cd_lim])
m = -(a/c)
b_plot = -(b/c)
label_line = rf'$F_s={acc:.3f}$'+'\n'+rf'$\alpha={m:.3f}$'+'\n'+rf'$\beta={b_plot:.3f}$'
plt.plot(xplot[lim_y], yplot[lim_y], color='black', lw=3, label=label_line)
plt.legend()
plt.xlabel(r'$n_1$')
plt.ylabel(r'$n_s$')
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n1_ns_contorno.png')
plt.close()

xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
yplot = -(a*xplot + b) / c
lim_y=yplot<np.max(n_s[cd_lim])
fig,axs=plt.subplots(1,2)
axs[0].scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.1, c='blue', edgecolor='black', label='E(EL)')
sns.kdeplot(x=n1[lim_cd_small],y=n_s[lim_cd_small],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,ax=axs[0])#thresh=0)
axs[0].plot([],[],color='black',label=r'$\rho$')
axs[0].plot(xplot[lim_y], yplot[lim_y], color='black', lw=3)
axs[0].legend()
axs[0].set_xlabel(r'$n_1$')
axs[0].set_ylabel(r'$n_s$')

axs[1].scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='True cD')
sns.kdeplot(x=n1[lim_cd_big],y=n_s[lim_cd_big],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,ax=axs[1])#thresh=0)
axs[1].plot([],[],color='black',label=r'$\rho$')
axs[1].plot(xplot[lim_y], yplot[lim_y], color='black', lw=3)
axs[1].legend()
axs[1].set_xlabel(r'$n_1$')
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n1_ns_contorno.png')
plt.close()
##################################################
label_x=r'$n_1$'
label_y=r'$n_s$'

#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=bt_vec_12[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_2C.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=bt_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_e_el_cd.png')
	plt.close()
#########################################################
#MAPA DE COR - RAZÃO BT - COM CORREÇÃO
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=bt_vec_corr[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_2C.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_e_el_cd.png')
	plt.close()
###################################################################################
#MAPA DE COR - RFF RATIO
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=rff_ratio[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_2C.png')
plt.close()

#MAPA DE COR DA RFF RATIO - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=rff_ratio_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE COR DA RFF RATIO - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=rff_ratio_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_e_el_cd.png')
	plt.close()
############
#MAPA DE COR - N2

os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=n2[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_2C.png')
plt.close()

#MAPA DE COR N2 - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=n2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_true_cd.png')
plt.close()

#MAPA DE COR N2 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=n2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=n2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=n2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=n2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_e_el_cd.png')
	plt.close()
#############################
#MAPA DE COR - CHI2 RATIO
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=chi2_ratio[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_2C.png')
plt.close()

#MAPA DE COR CHI2 RATIO - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR CHI2 RATIO - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_e_el_cd.png')
	plt.close()

#MAPA DE COR - BOX 1
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/box1',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=box1[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_2C.png')
plt.close()

#MAPA DE COR DA BOX 1 - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=box1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 1 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=box1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=box1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=box1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=box1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_e_el_cd.png')
	plt.close()

#MAPA DE COR - BOX 2
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/box2',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=box2[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_2C.png')
plt.close()

#MAPA DE COR DA BOX 2 - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=box2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 2 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=box2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=box2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=box2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=box2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_e_el_cd.png')
	plt.close()

#MAPA DE COR - ETA
os.makedirs(f'{sample}_stats_observation_desi/plano_n1_ns/eta',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=eta[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_2C.png')
plt.close()

#MAPA DE COR DA BOX 1 - TRUE cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=eta[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 1 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=eta[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=eta[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_cd_zhao.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=eta[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=eta[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_e_el_cd.png')
	plt.close()

###################################################################################
###################################################################################
###################################################################################
#PLANO DE RAZÃO DE RAIOS (N1/N2) x N2


label_x=r'$n_{1}/n_{2}$'
label_y=r'$log_{10} n_{2}$'

fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolor='black',label='E')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolor='black',label='True cD')
axs[2].set_xlabel(label_x)
axs[2].legend()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/scatter_n_ratio_n2_geral.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
axs.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolor='black',label='E')
axs.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolor='black',label='True cD')
axs.set_xlabel(label_x)
axs.set_ylabel(label_y)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/scatter_n_ratio_n2_geral.png')
plt.close()
#SUBPLOTS

fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,10),constrained_layout=True)
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=n_ratio_12[elip_lim],y=np.log10(n2)[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs[0])
axs[0].set_ylabel(label_y)
axs[0].set_xlabel(label_x)
axs[0].legend()

axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=n_ratio_12[lim_cd_big],y=np.log10(n2)[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(label_x)
axs[1].legend()

axs[2].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=n_ratio_12[lim_cd_small],y=np.log10(n2)[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_sub_grupos.png')
plt.close()

#UNITÁRIOS

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=n_ratio_12[elip_lim],y=np.log10(n2)[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='True cD')
sns.kdeplot(x=n_ratio_12[lim_cd_big],y=np.log10(n2)[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_true_cds.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=n_ratio_12[lim_cd_small],y=np.log10(n2)[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.legend()
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_extra_light.png')
plt.close()

if sample == 'L07':
	#ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
	sns.kdeplot(x=n_ratio_12[e_cut],y=np.log10(n2)[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_zhao.png')
	plt.close()

	#cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
	sns.kdeplot(x=n_ratio_12[cd_cut],y=np.log10(n2)[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_cds_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c='green',edgecolors='black',alpha=0.2,label='E[E]')
	sns.kdeplot(x=n_ratio_12[elip_lim & e_cut],y=np.log10(n2)[elip_lim & e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c='green',edgecolors='red',alpha=0.2,label='E[cD]')
	sns.kdeplot(x=n_ratio_12[elip_lim & cd_cut],y=np.log10(n2)[elip_lim & cd_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c='green',edgecolors='black',alpha=0.2,label='E[E/cD]')
	sns.kdeplot(x=n_ratio_12[elip_lim & (ecd_cut | cde_cut)],y=np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='True cD[cD]')
	sns.kdeplot(x=n_ratio_12[lim_cd_big & cd_cut],y=np.log10(n2)[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_true_cds_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c='blue',edgecolors='black',alpha=0.2,label='E(EL)[cD]')
	sns.kdeplot(x=n_ratio_12[lim_cd_small & cd_cut],y=np.log10(n2)[lim_cd_small & cd_cut],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.legend()
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/nxn2_extra_light_cd.png')
	plt.close()
################################################

#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n_ratio_12,np.log10(n2),c=bt_vec_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt.png')
plt.close()

#MAPA DE COR BT - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=bt_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=bt_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),bt_vec_12
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/n_ratio_n2_bt_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/n_ratio_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=bt_vec_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=bt_vec_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=bt_vec_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_el_cd.png')
	plt.close()

##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12,np.log10(n2),c=bt_vec_corr,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_true_cd_corr.png')
plt.close()

fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=bt_e_corr,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_elip_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_only2c.png')
plt.close()


fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),bt_vec_corr
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/n_ratio_n2_bt_corr_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/n_ratio_n2_bt_corr_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=bt_vec_corr[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=bt_vec_corr[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=bt_vec_corr[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_corr[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_el_cd.png')
	plt.close()

###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=rff_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio.png')
plt.close()

#MAPA DE RAZÃO DE RFF - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=rff_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=rff_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=rff_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],rff_ratio_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],rff_ratio_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],rff_ratio_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),rff_ratio
vec_label=label_x,label_y,r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/n_ratio_n2_rff_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/n_ratio_n2_rff_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=rff_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=rff_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=rff_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=rff_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=chi2_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=chi2_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.9,1.5)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),chi2_ratio
vec_label=label_x,label_y,r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=None,None,[0.9,1.5]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/n_ratio_n2_chi2_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/n_ratio_n2_chi2_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=chi2_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=chi2_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=chi2_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=chi2_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=delta_bic_obs,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic.png')
plt.close()

#MAPA DE COR DELTA BIC - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=delta_bic_obs[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(1000,-3000)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),delta_bic_obs
vec_label=label_x,label_y,r'$\Delta BIC$'
xlim,ylim,zlim=None,None,[1000,-3000]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/n_ratio_n2_delta_bic_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/n_ratio_n2_delta_bic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=delta_bic_obs[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=delta_bic_obs[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=delta_bic_obs[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=delta_bic_obs[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=delta_bic_obs[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=e_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=axrat_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,1.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e_s
vec_label=label_x,label_y,r'$q$'
xlim,ylim,zlim=None,None,[0.5,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/n_ratio_n2_ax_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/n_ratio_n2_ax_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=axrat_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=axrat_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_elip.png')
plt.close()


#MAPA DE COR RAZÃO Q1/Q2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),axrat_ratio_12
vec_label=label_x,label_y,r'$q_1/q_2$'
xlim,ylim,zlim=None,None,[0.1,2.8]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/n_ratio_n2_q_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/n_ratio_n2_q_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=axrat_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=axrat_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=axrat_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=axrat_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=axrat_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=box_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=box_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(-0.5,0.5)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),box_s
vec_label=label_x,label_y,r'$a_4/a$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/n_ratio_n2_box_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/n_ratio_n2_box_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=box_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=box_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=box_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=box_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=box_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR RAZÃO - re1/re2
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=np.log10(re_ratio_12),edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=np.log10(re_ratio_12[elip_lim]),edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=np.log10(re_ratio_12[lim_cd_big]),edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=np.log10(re_ratio_12[lim_cd_small]),edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_el.png')
plt.close()

#ANALISE 3D - re1/re2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'${R_e}_1/{R_e}_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[elip_lim]),n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-2,2.])
axs[1].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[elip_lim]),np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-2,2])
axs[2].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - re1/re2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],re_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],re_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'${R_e}_1/{R_e}_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([-2,2.])
axs[1].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([-2,2])
axs[2].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),re_ratio_12
vec_label=label_x,label_y,r'${R_e}_1/{R_e}_2$'
xlim,ylim,zlim=None,None,[-2,2]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/n_ratio_n2_re_ratio_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/n_ratio_n2_re_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=np.log10(re_ratio_12)[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=np.log10(re_ratio_12)[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=np.log10(re_ratio_12)[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=np.log10(re_ratio_12)[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=np.log10(re_ratio_12)[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=np.log10(re_ratio_12)[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=n_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=n_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - TRUE cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,12)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,12])
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,12])
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),n_s
vec_label=label_x,label_y,r'$n$'
xlim,ylim,zlim=None,None,[0.5,12]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/n_ratio_n2_n_sersic_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/n_ratio_n2_n_sersic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=n_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=n_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=n_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=n_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=n_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=n1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=n1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),n1
vec_label=label_x,label_y,r'$n_1$'
xlim,ylim,zlim=None,None,[0.5,10]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/n_ratio_n2_n1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/n_ratio_n2_n1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=n1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=n1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=n1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=n1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=n1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12,np.log10(n2),c=e1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=e1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e1
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/n_ratio_n2_q1_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/n_ratio_n2_q1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=e2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=e2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - TRUE cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_true_cd.png')
plt.close()
#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='True cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='True cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e2
vec_label=label_x,label_y,r'$q_2$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/n_ratio_n2_q2_3d_spin_only2c.gif',f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/n_ratio_n2_q2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_misc.png')
	plt.close()

	#True cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}_stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_el_cd.png')
	plt.close()

#########################################

#BASE DE PLOT PARA 3D
r'''

#bt_corr re1re2 magabs 0.8805970149253731
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],bt_vec_corr[elip_lim],edgecolors='black',color='green')
# ax.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],np.log10(bt_vec_corr[lim_cd_big]),edgecolors='black',color='red')
# ax.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],np.log10(bt_vec_corr[lim_cd_small]),edgecolors='black',color='blue')
# ax.scatter(re_1_kpc[lim_cd_big],mue_1[lim_cd_big],(bt_vec_corr[lim_cd_big]),edgecolors='black',color='red')
# ax.scatter(re_1_kpc[lim_cd_small],mue_1[lim_cd_small],(bt_vec_corr[lim_cd_small]),edgecolors='black',color='blue')
ax.scatter(np.log10(re_ratio_12)[lim_cd_small & lim_casjobs],magabs[cd_lim_casjobs_small],np.log10(bt_vec_corr[lim_cd_small & lim_casjobs]),edgecolors='black',color='blue')
ax.scatter(np.log10(re_ratio_12)[lim_cd_big & lim_casjobs],magabs[cd_lim_casjobs_big],np.log10(bt_vec_corr[lim_cd_big & lim_casjobs]),edgecolors='black',color='red')

ax.set_xlabel("re_rat")
ax.set_ylabel("Mag")
ax.set_zlabel("bt")

plt.tight_layout()
plt.show()
plt.close()
'''
#########################################
###
#GRÁFICO JOINTPLOT DA RELAÇÃO DE KORMENDY

r'''#SÉRSIC SIMPLES DAS CDS (SMALL E BIG) CONTRA ELIPTICAS
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_small,mue_sersic_cd_small,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_small, color='red', linestyle='-', label=linha_cd_label+' nuc\n'+fr'$\alpha={alpha_cd_label_small}$'+'\n'+fr'$\beta={beta_cd_label_small}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)
plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic_small.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_big,mue_sersic_cd_big,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_big, color='red', linestyle='-', label=linha_cd_label+' std\n'+fr'$\alpha={alpha_cd_label_big}$'+'\n'+fr'$\beta={beta_cd_label_big}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_big(re_linspace),color='blue')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample_sersic_big.png')
plt.close()
###
#POR COMPONENTES EM RELAÇÃO A SÉRSIC

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
ax_center.plot(linspace_re, linha_interna, color='red', linestyle='--', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
ax_center.scatter(re_extern,mue_extern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='red')
ax_center.plot(linspace_re, linha_externa, color='red', linestyle='-.', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern),'.3')}$')
ax_topx.axvline(np.average(re_extern),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern),'.3')}$')
ax_righty.axhline(np.average(mue_extern),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_sample.png')
plt.close()

####
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo+' nuc',color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='red', linestyle='--', label=linha_interna_label+' nuc\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env+' nuc',color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_label+' nuc\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_small.png')
plt.close()
####
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo+' std',color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_label+' std\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env+' std',color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_label+' std\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend()
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(-0.28322921653227484,2.716867381942037)#x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}_stats_observation_desi/rel_kormendy_big.png')
plt.close()
'''
r'''
###################################################################################
###################################################################################
###################################################################################
#INVESTIGAÇÃO DO P_VALUE

fig,axs=plt.subplots(1,3,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],color='green',edgecolor='black',alpha=0.9,label='E')
axs[0].axhline(0.32)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],color='blue',edgecolor='black',alpha=0.9,label='E(EL)')
axs[1].axhline(0.32)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],color='red',edgecolor='black',alpha=0.9,label='True cD')
axs[2].axhline(0.32,label=r'$1 \sigma$')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(loc='outside upper right',fontsize=9)

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_lines.png')
plt.close(fig)

#INDICE DE SÉRSIC - MODELO SIMPLES
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=n_sersic_e,edgecolor='black',vmax=8,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=n_sersic_cd_small,edgecolor='black',vmax=8,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=n_sersic_cd_big,edgecolor='black',vmax=8,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_n_simples.png')
plt.close(fig)

#RAZÃO AXIAL SÉRSIC - MODELO SIMPLES
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=axrat_sersic_e,edgecolor='black',alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=axrat_sersic_cd_small,edgecolor='black',alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=axrat_sersic_cd_big,edgecolor='black',alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$q$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_axrat.png')
plt.close()

#BOXINESS
plt.close(fig)
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=box_sersic_e,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=box_sersic_cd_small,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=box_sersic_cd_big,edgecolor='black',vmin=-0.3,vmax=0.3,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$a_4/a$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_box.png')
plt.close(fig)

#B/T
fig,axs=plt.subplots(1,3,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=bt_vec_12[elip_lim],edgecolor='black',alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=bt_vec_12[lim_cd_small],edgecolor='black',alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=bt_vec_12[lim_cd_big],edgecolor='black',alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$B/T$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_bt.png')
plt.close(fig)

#RAZÃO DE RAIOS
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=re_ratio_12[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap_r)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=re_ratio_12[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap_r)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=re_ratio_12[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap_r)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$Re_1/Re_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_re_ratio.png')
plt.close(fig)

#RAZÃO DE INDICES DE SÉRSIC
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=n_ratio_12[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n_1/n_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_n_ratio.png')
plt.close(fig)

#RAZÃO DE RAZÃO AXIAL
fig,axs=plt.subplots(1,3,sharey=True,tight_layout=True,figsize=(15,5))
axs[0].scatter(delta_bic_obs[elip_lim],p_value[elip_lim],c=(e1/e2)[elip_lim],edgecolor='black',vmax=1.5,alpha=0.9,label='E',cmap=cmap)
axs[0].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[0].axvline(corte,c=line_colors[i],lw=0.5)
axs[0].set_xlabel(r'$\Delta BIC$')
axs[0].set_ylabel(r'$P_{value}$')
axs[0].set_xlim(2000,-8000)
axs[0].set_ylim(-0.01,1.01)

axs[1].scatter(delta_bic_obs[lim_cd_small],p_value[lim_cd_small],c=(e1/e2)[lim_cd_small],edgecolor='black',vmax=1.5,alpha=0.9,label='E(EL)',cmap=cmap)
axs[1].axhline(0.32)
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[1].axvline(corte,c=line_colors[i],lw=0.5)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$P_{value}$')
axs[1].set_xlim(2000,-8000)
axs[1].set_ylim(-0.01,1.01)

sc=axs[2].scatter(delta_bic_obs[lim_cd_big],p_value[lim_cd_big],c=(e1/e2)[lim_cd_big],edgecolor='black',vmax=1.5,alpha=0.9,label='True cD',cmap=cmap)
axs[2].axhline(0.32,label=r'$1 \sigma$')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
for i,corte in enumerate(vec_cuts):
	axs[2].axvline(corte,c=line_colors[i],lw=0.5,label=f'{corte:.2f}')
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$P_{value}$')
axs[2].set_xlim(2000,-8000)
axs[2].set_ylim(-0.01,1.01)
fig.legend(bbox_to_anchor=(0.3,0.95),fontsize=9)
cbar=plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$q_1/q_2$')

plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic/p_value_delta_bic_q_ratio.png')
plt.close(fig)
'''
r'''
fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
sc=axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,vmin=0.0,vmax=1,c=bt_vec_12[cd_lim],cmap=cmap)
axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
# axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
# axs.plot(linspace_re, linha_externa, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label('B/T')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
plt.close()

fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(7,7))
axs.scatter(re_ratio_12,n_ratio_12,color='white',edgecolor='black',alpha=0.2,label='WHL')
axs.scatter(re_ratio_12[high_lim_cd],n_ratio_12[high_lim_cd],color='red',edgecolor='black',alpha=0.4,label='n1>2.5')
axs.scatter(re_ratio_12[low_lim_cd],n_ratio_12[low_lim_cd],color='blue',edgecolor='black',alpha=0.6,label='n1<2.5')
axs.scatter(re_ratio_12[higher_lim_cd & (re_1_kpc<0.5)],n_ratio_12[higher_lim_cd & (re_1_kpc<0.5)],color='orange',edgecolor='black',alpha=1,label='n1>9.5')
axs.legend()
axs.set_xlabel(r'$Re_1/Re_2$')
axs.set_ylabel(r'$n_1/n_2$')
axs.set_xlim(-0.1,1.1)


# plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic_lines.png')
plt.close(fig)

xp1,xp2=min(p_value),max(p_value)
xz_p=np.linspace(0,1,len(delta_bic_obs))

fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(7,7))
axs.scatter(delta_bic_obs,p_value,color='white',edgecolor='black',alpha=0.9)
axs.axhline(0.32,label=r'$1 \sigma$')
axs.plot(log_model(xz,*pov),xz_p,color='black',label=r'$\Delta BIC$ $MAD_{line}$')
axs.legend()
axs.set_xlabel(r'$\Delta BIC$')
axs.set_ylabel(r'$P_{value}$')
axs.set_xlim(2000,-8000)
axs.set_ylim(-0.01,1.01)

# plt.savefig(f'{sample}_stats_observation_desi/p_value_delta_bic_lines.png')
plt.close(fig)
'''
##########################################################
#ANALISE POR PARAMETROS DE INTERESSE POR FAIXAS DE N1 E RE
r'''
param='re_08',r'$n_1$'
# param='n',r'$n$'
# param='bt',r'$B/T$'
# param='age',r'$\tau$'
n1_intern=n1[cd_lim]

faixas_n1=np.percentile(n1_intern,np.linspace(0,90,10))
idx_faixas_n1=np.digitize(n1_intern,faixas_n1,right=False)
bins_med_n1=[]
for n in range(len(faixas_n1)-1):
	bins_med_n1.append((faixas_n1[n],faixas_n1[n+1]))
bins_med_n1.append(faixas_n1[-1])
num_n1=np.unique(idx_faixas_n1)
for item in num_n1:
	sub_sample_n1=(idx_faixas_n1==item) & (re_intern<0.8)
	sub_sample_n1_resto=(idx_faixas_n1==item) & (re_intern>0.8)

	med_n1_sub_sample=np.average(n1_intern[sub_sample_n1])
	###
	re_intern_n1,re_extern_n1,re_extern_n1_res=re_intern[sub_sample_n1],re_extern[sub_sample_n1],re_extern[sub_sample_n1_resto]
	mue_intern_n1,mue_extern_n1,mue_extern_n1_res=mue_intern[sub_sample_n1],mue_extern[sub_sample_n1],mue_extern[sub_sample_n1_resto]

	[alpha_med_1_n1,beta_med_1_n1],cov_1_n1 = np.polyfit(re_intern_n1,mue_intern_n1,1,cov=True)
	[alpha_med_2_n1,beta_med_2_n1],cov_2_n1 = np.polyfit(re_extern_n1,mue_extern_n1,1,cov=True)
	[alpha_med_2_n1_res,beta_med_2_n1_res],cov_2_n1_res = np.polyfit(re_extern_n1_res,mue_extern_n1_res,1,cov=True)

	linha_intern_n1 = alpha_med_1_n1 * comp_12_linspace + beta_med_1_n1
	linha_extern_n1 = alpha_med_2_n1 * comp_12_linspace + beta_med_2_n1
	linha_extern_n1_res = alpha_med_2_n1_res * comp_12_linspace + beta_med_2_n1_res

	alpha_med_int_n1,beta_med_int_n1 = format(alpha_med_1_n1,'.3'),format(beta_med_1_n1,'.3') 
	alpha_med_ext_n1,beta_med_ext_n1 = format(alpha_med_2_n1,'.3'),format(beta_med_2_n1,'.3')
	alpha_med_ext_n1_res,beta_med_ext_n1_res = format(alpha_med_2_n1_res,'.3'),format(beta_med_2_n1_res,'.3')
	###	
	re_sersic_cd_kpc_n1=re_sersic_cd_kpc[sub_sample_n1]
	mue_sersic_cd_n1=mue_sersic_cd[sub_sample_n1]

	[alpha_cd_n1,beta_cd_n1],cov_cd_n1 = np.polyfit(re_sersic_cd_kpc_n1,mue_sersic_cd_n1,1,cov=True)

	linha_cd_n1 = alpha_cd_n1 * comp_12_linspace + beta_cd_n1

	alpha_cd_label_n1,beta_cd_label_n1 = format(alpha_cd_n1,'.3'),format(beta_cd_n1,'.3') 
	####

	if item != len(num_n1):
		label=f'{np.round(bins_med_n1[item-1][0],3)}<{param[1]}<{np.round(bins_med_n1[item-1][1],3)}'
	else:
		label=f'{np.round(bins_med_n1[item-1],3)}<{param[1]}'

	###############

	# fig,axs=plt.subplots(1,1,sharey=True,figsize=(7,7))
	# plt.suptitle(label)
	# axs.scatter(re_sersic_cd_kpc_n1,mue_sersic_cd_n1,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	# axs.plot(linspace_re, linha_cd, color='red', linestyle='-',alpha=0.9, label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
	# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	# axs.plot(linspace_re, linha_cd_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_cd_label_n1}$'+'\n'+fr'$\beta={beta_cd_label_n1}$')
	# axs.set_ylim(y2ss,y1ss)
	# axs.set_xlim(x1ss,x2ss)
	# axs.set_xlabel(xlabel)
	# axs.set_ylabel(ylabel)
	# axs.legend()
	# plt.savefig(f'{sample}_stats_observation_desi/analise_sersic_cds/{param[0]}_{item-1}.png')
	# plt.close()

	########################################

	fig,axs=plt.subplots(1,1,sharey=True,figsize=(7,7))
	plt.suptitle(label)
	axs.scatter(re_intern[sub_sample_n1],mue_intern[sub_sample_n1],marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	axs.plot(linspace_re, linha_interna, color='red', linestyle='-',alpha=0.9, label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
	axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	axs.plot(linspace_re, linha_intern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_int_n1}$'+'\n'+fr'$\beta={beta_med_int_n1}$')
	axs.set_ylim(y2ss,y1ss)
	axs.set_xlim(x1ss,x2ss)
	axs.set_xlabel(xlabel)
	axs.set_ylabel(ylabel)
	axs.legend()
	
	# plt.savefig(f'{sample}_stats_observation_desi/analise_comp_cds/intern/{param[0]}_intern_{item-1}.png')
	plt.close()

	fig,axs=plt.subplots(1,1,sharey=True,figsize=(7,7))
	plt.suptitle(label)
	axs.scatter(re_extern[sub_sample_n1],mue_extern[sub_sample_n1],marker='o',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	axs.plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
	axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	axs.plot(linspace_re, linha_extern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_ext_n1}$'+'\n'+fr'$\beta={beta_med_ext_n1}$')
	axs.set_ylim(y2ss,y1ss)
	axs.set_xlim(x1ss,x2ss)
	axs.set_xlabel(xlabel)
	axs.set_ylabel(ylabel)
	axs.legend()
	# plt.savefig(f'{sample}_stats_observation_desi/analise_comp_cds/extern/{param[0]}_extern_{item-1}.png')
	plt.close()

	fig,axs=plt.subplots(1,2,sharey=True,figsize=(15,10))
	plt.suptitle(label)
	axs[0].set_title('re < 0.8 (bololo)')
	axs[0].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.2,label='Sersic',color='green')
	axs[0].scatter(re_extern[sub_sample_n1],mue_extern[sub_sample_n1],marker='o',edgecolor='black',alpha=0.5,label=label_bojo,color='white')
	axs[0].plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
	axs[0].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	axs[0].plot(linspace_re, linha_extern_n1, color='orange', linestyle='-', label='Linha sub amostra re < 0.8'+'\n'+fr'$\alpha={alpha_med_ext_n1}$'+'\n'+fr'$\beta={beta_med_ext_n1}$')
	axs[0].set_ylim(y2ss,y1ss)
	axs[0].set_xlim(x1ss,x2ss)
	axs[0].set_xlabel(xlabel)
	axs[0].set_ylabel(ylabel)
	axs[0].legend()

	axs[1].set_title('re > 0.8 (resto)')
	axs[1].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.2,label='Sersic',color='green')
	axs[1].scatter(re_extern[sub_sample_n1_resto],mue_extern[sub_sample_n1_resto],marker='o',edgecolor='black',alpha=0.5,label=label_bojo,color='white')
	axs[1].plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
	axs[1].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	axs[1].plot(linspace_re, linha_extern_n1_res, color='orange', linestyle='-', label='Linha sub amostra re > 0.8'+'\n'+fr'$\alpha={alpha_med_ext_n1_res}$'+'\n'+fr'$\beta={beta_med_ext_n1_res}$')
	axs[1].set_ylim(y2ss,y1ss)
	axs[1].set_xlim(x1ss,x2ss)
	axs[1].set_xlabel(xlabel)
	axs[1].set_ylabel(ylabel)
	axs[1].legend()
	plt.show()
	# plt.savefig(f'{sample}_stats_observation_desi/analise_comp_cds/extern/{param[0]}_extern_{item-1}.png')
	plt.close()


	# fig,axs=plt.subplots(1,2,sharey=True,figsize=(10,5))
	# plt.suptitle(label)
	# axs[0].scatter(re_intern[sub_sample_n1],mue_intern[sub_sample_n1],marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	# axs[0].plot(linspace_re, linha_interna, color='red', linestyle='-',alpha=0.9, label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
	# axs[0].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	# axs[0].plot(linspace_re, linha_intern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_int_n1}$'+'\n'+fr'$\beta={beta_med_int_n1}$')
	# axs[0].set_ylim(y2ss,y1ss)
	# axs[0].set_xlim(x1ss,x2ss)
	# axs[0].set_xlabel(xlabel)
	# axs[0].set_ylabel(ylabel)
	# axs[0].legend()

	# axs[1].scatter(re_extern[sub_sample_n1],mue_extern[sub_sample_n1],marker='o',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	# axs[1].scatter(re_s_kpc[sub_sample_n1],mue_sersic[sub_sample_n1],marker='o',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	# axs[1].plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
	# axs[1].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	# axs[1].plot(linspace_re, linha_extern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_ext_n1}$'+'\n'+fr'$\beta={beta_med_ext_n1}$')
	# axs[1].set_ylim(y2ss,y1ss)
	# axs[1].set_xlim(x1ss,x2ss)
	# axs[1].set_xlabel(xlabel)
	# axs[1].set_ylabel(ylabel)
	# axs[1].legend()
	# 
	# # plt.savefig(f'{sample}_stats_observation_desi/analise_comp_cds/both/{param[0]}_{item-1}.png')
	# plt.close()

#GIF BUILDER

writer=PillowWriter(fps=0.5)
gif_name='WHL_stats_observation_desi/gif_n1_extern_comp_07.gif'

fig,axs=plt.subplots(1,2,sharey=True,figsize=(15,10))
# fig,axs=plt.subplots(1,1,sharey=True,figsize=(7,7))
with writer.saving(fig, gif_name,dpi=150):
	for item in num_n1:
		axs[0].clear()
		axs[1].clear()
		# sub_sample_n1=(idx_faixas_n1==item) & (n1_intern<=9.7)
		sub_sample_n1=(idx_faixas_n1==item) & (re_intern<0.7)
		sub_sample_n1_resto=(idx_faixas_n1==item) & (re_intern>0.7)

		# print(item,len(n1_intern[sub_sample_n1]))
		med_n1_sub_sample=np.average(n1_intern[sub_sample_n1])

		re_intern_n1,re_extern_n1,re_extern_n1_res=re_intern[sub_sample_n1],re_extern[sub_sample_n1],re_extern[sub_sample_n1_resto]
		mue_intern_n1,mue_extern_n1,mue_extern_n1_res=mue_intern[sub_sample_n1],mue_extern[sub_sample_n1],mue_extern[sub_sample_n1_resto]

		[alpha_med_1_n1,beta_med_1_n1],cov_1_n1 = np.polyfit(re_intern_n1,mue_intern_n1,1,cov=True)
		[alpha_med_2_n1,beta_med_2_n1],cov_2_n1 = np.polyfit(re_extern_n1,mue_extern_n1,1,cov=True)
		[alpha_med_2_n1_res,beta_med_2_n1_res],cov_2_n1_res = np.polyfit(re_extern_n1_res,mue_extern_n1_res,1,cov=True)

		linha_intern_n1 = alpha_med_1_n1 * comp_12_linspace + beta_med_1_n1
		linha_extern_n1 = alpha_med_2_n1 * comp_12_linspace + beta_med_2_n1
		linha_extern_n1_res = alpha_med_2_n1_res * comp_12_linspace + beta_med_2_n1_res

		alpha_med_int_n1,beta_med_int_n1 = format(alpha_med_1_n1,'.3'),format(beta_med_1_n1,'.3') 
		alpha_med_ext_n1,beta_med_ext_n1 = format(alpha_med_2_n1,'.3'),format(beta_med_2_n1,'.3')
		alpha_med_ext_n1_res,beta_med_ext_n1_res = format(alpha_med_2_n1_res,'.3'),format(beta_med_2_n1_res,'.3')
		if item != len(num_n1):
			label=f'{np.round(bins_med_n1[item-1][0],3)}<n1<{np.round(bins_med_n1[item-1][1],3)}'
		else:
			label=f'{np.round(bins_med_n1[item-1],3)}<n1'

		# fig,axs=plt.subplots(1,2,sharey=True,figsize=(15,10))
		plt.suptitle(label)
		axs[0].set_title('re < 0.7 (bololo)')
		axs[0].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.2,label='Sersic',color='green')
		axs[0].scatter(re_extern[sub_sample_n1],mue_extern[sub_sample_n1],marker='o',edgecolor='black',alpha=0.5,label=label_bojo,color='white')
		axs[0].plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
		axs[0].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
		axs[0].plot(linspace_re, linha_extern_n1, color='orange', linestyle='-', label='Linha sub amostra re < 0.8'+'\n'+fr'$\alpha={alpha_med_ext_n1}$'+'\n'+fr'$\beta={beta_med_ext_n1}$')
		axs[0].set_ylim(y2ss,y1ss)
		axs[0].set_xlim(x1ss,x2ss)
		axs[0].set_xlabel(xlabel)
		axs[0].set_ylabel(ylabel)
		axs[0].legend()

		axs[1].set_title('re > 0.7 (resto)')
		axs[1].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.2,label='Sersic',color='green')
		axs[1].scatter(re_extern[sub_sample_n1_resto],mue_extern[sub_sample_n1_resto],marker='o',edgecolor='black',alpha=0.5,label=label_bojo,color='white')
		axs[1].plot(linspace_re, linha_externa, color='red', linestyle='-',alpha=0.9, label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
		axs[1].plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
		axs[1].plot(linspace_re, linha_extern_n1_res, color='orange', linestyle='-', label='Linha sub amostra re > 0.8'+'\n'+fr'$\alpha={alpha_med_ext_n1_res}$'+'\n'+fr'$\beta={beta_med_ext_n1_res}$')
		axs[1].set_ylim(y2ss,y1ss)
		axs[1].set_xlim(x1ss,x2ss)
		axs[1].set_xlabel(xlabel)
		axs[1].set_ylabel(ylabel)
		axs[1].legend()
		writer.grab_frame()
		
		# plt.savefig(f'{sample}_stats_observation_desi/analise_comp_cds/extern/{param[0]}_extern_{item-1}.png')
		# plt.close()


		# plt.suptitle(label)
		# axs.scatter(re_intern[sub_sample_n1],mue_intern[sub_sample_n1],marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
		# axs.plot(linspace_re, linha_interna, color='red', linestyle='-',alpha=0.9, label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
		# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
		# axs.plot(linspace_re, linha_intern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_int_n1}$'+'\n'+fr'$\beta={beta_med_int_n1}$')
		# axs.set_ylim(y2ss,y1ss)
		# axs.set_xlim(x1ss,x2ss)
		# axs.set_xlabel(xlabel)
		# axs.set_ylabel(ylabel)
		# axs.legend()
		# writer.grab_frame()
		# 
		# # plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
plt.close(fig)
'''
################################################
#ANALISE POR INDICE DE SÉRSIC DO COMPONENTE INTERNO QUANTO AO VALOR DE N1
r'''
n1_intern=n1[cd_lim]
faixas_n1=np.percentile(n1_intern,np.linspace(0,90,5))
idx_faixas_n1=np.digitize(n1_intern,faixas_n1,right=False)
bins_med_n1=[]
for n in range(len(faixas_n1)-1):
	bins_med_n1.append((faixas_n1[n],faixas_n1[n+1]))
	# print(faixas_n1[n],'<n1<',faixas_n1[n+1])
bins_med_n1.append(faixas_n1[-1])
num_n1=np.unique(idx_faixas_n1)
# print(faixas_n1,num_n1,idx_faixas_n1)
print(len(bins_med_n1),len(num_n1))
for item in num_n1:
	sub_sample_n1=(idx_faixas_n1==item) & (n1_intern<=9.7)
	# print(item,len(n1_intern[sub_sample_n1]))
	med_n1_sub_sample=np.average(n1_intern[sub_sample_n1])

	re_intern_n1,re_extern_n1=re_intern[sub_sample_n1],re_extern[sub_sample_n1]
	mue_intern_n1,mue_extern_n1=mue_intern[sub_sample_n1],mue_extern[sub_sample_n1]

	[alpha_med_1_n1,beta_med_1_n1],cov_1_n1 = np.polyfit(re_intern_n1,mue_intern_n1,1,cov=True)
	[alpha_med_2_n1,beta_med_2_n1],cov_2_n1 = np.polyfit(re_extern_n1,mue_extern_n1,1,cov=True)

	linha_intern_n1 = alpha_med_1_n1 * comp_12_linspace + beta_med_1_n1
	linha_extern_n1 = alpha_med_2_n1 * comp_12_linspace + beta_med_2_n1

	alpha_med_int_n1,beta_med_int_n1 = format(alpha_med_1_n1,'.3'),format(beta_med_1_n1,'.3') 
	alpha_med_ext_n1,beta_med_ext_n1 = format(alpha_med_2_n1,'.3'),format(beta_med_2_n1,'.3')
	if item != len(num_n1):
		label=f'{np.round(bins_med_n1[item-1][0],3)}<n1<{np.round(bins_med_n1[item-1][1],3)}'
	else:
		label=f'{np.round(bins_med_n1[item-1],3)}<n1'

	fig,axs=plt.subplots(1,1,sharey=True,figsize=(7,7))
	plt.suptitle(label)
	axs.scatter(re_intern[sub_sample_n1],mue_intern[sub_sample_n1],marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='white')
	axs.plot(linspace_re, linha_interna, color='red', linestyle='-',alpha=0.9, label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
	axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.',alpha=0.9, label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	axs.plot(linspace_re, linha_intern_n1, color='black', linestyle='-', label='Linha sub amostra'+'\n'+fr'$\alpha={alpha_med_int_n1}$'+'\n'+fr'$\beta={beta_med_int_n1}$')
	axs.set_ylim(y2ss,y1ss)
	axs.set_xlim(x1ss,x2ss)
	axs.set_xlabel(xlabel)
	axs.set_ylabel(ylabel)
	axs.legend()
	plt.show()
	# plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
	plt.close()
'''
###############################################################
#GALÁXIAS COM NÚCLEOS KDC
r'''
####
#CORTE - DELTA BIC E N1
##OBJETOS COM N1<2.5

re_sersic_cd_kpc_low,re_sersic_kpc_low=re_s_kpc[low_lim_cd],re_s_kpc[low_lim_elip]
mue_sersic_cd_low,mue_sersic_low=mue_med_s[low_lim_cd],mue_med_s[low_lim_elip]

[alpha_cd_low,beta_cd_low],cov_cd_low = np.polyfit(re_sersic_cd_kpc_low,mue_sersic_cd_low,1,cov=True)
[alpha_s_low,beta_s_low],cov_s_low = np.polyfit(re_sersic_kpc_low,mue_sersic_low,1,cov=True)

linha_cd_low = alpha_cd_low * comp_12_linspace + beta_cd_low
linha_sersic_low = alpha_s_low * comp_12_linspace + beta_s_low

alpha_cd_low_label,beta_cd_low_label = format(alpha_cd_low,'.3'),format(beta_cd_low,'.3') 
alpha_ser_low_label,beta_ser_low_label = format(alpha_s_low,'.3'),format(beta_s_low,'.3')

fig,axs=plt.subplots(1,1,sharey=True,sharex=True,figsize=(10,10))
axs.scatter(re_sersic_cd_kpc_low,mue_sersic_cd_low,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
axs.plot(linspace_re, linha_cd_low, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_low_label}$'+'\n'+fr'$\beta={beta_cd_low_label}$')
axs.scatter(re_sersic_kpc_low,mue_sersic_low,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
axs.plot(linspace_re, linha_sersic_low, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_ser_low_label}$'+'\n'+fr'$\beta={beta_ser_low_label}$')
axs.set_ylim(y2s,y1s)
axs.set_xlim(x1s,x2s)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()

plt.savefig(f'{sample}_stats_observation_desi/p_value/rel_kormendy_cd_e_low.png')
plt.close()
####
##OBJETOS COM N1>2.5

re_sersic_cd_kpc_high,re_sersic_kpc_high=re_s_kpc[high_lim_cd],re_s_kpc[high_lim_elip]
mue_sersic_cd_high,mue_sersic_high=mue_med_s[high_lim_cd],mue_med_s[high_lim_elip]

[alpha_cd_high,beta_cd_high],cov_cd_high = np.polyfit(re_sersic_cd_kpc_high,mue_sersic_cd_high,1,cov=True)
[alpha_s_high,beta_s_high],cov_s_high = np.polyfit(re_sersic_kpc_high,mue_sersic_high,1,cov=True)

linha_cd_high = alpha_cd_high * comp_12_linspace + beta_cd_high
linha_sersic_high = alpha_s_high * comp_12_linspace + beta_s_high

alpha_cd_high_label,beta_cd_high_label = format(alpha_cd_high,'.3'),format(beta_cd_high,'.3') 
alpha_ser_high_label,beta_ser_high_label = format(alpha_s_high,'.3'),format(beta_s_high,'.3')

fig,axs=plt.subplots(1,1,sharey=True,sharex=True,figsize=(10,10))
axs.scatter(re_sersic_cd_kpc_high,mue_sersic_cd_high,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
axs.plot(linspace_re, linha_cd_high, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_high_label}$'+'\n'+fr'$\beta={beta_cd_high_label}$')
axs.scatter(re_sersic_kpc_high,mue_sersic_high,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
axs.plot(linspace_re, linha_sersic_high, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_ser_high_label}$'+'\n'+fr'$\beta={beta_ser_high_label}$')
axs.set_ylim(y2s,y1s)
axs.set_xlim(x1s,x2s)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
plt.show()
# plt.savefig(f'{sample}_stats_observation_desi/p_value/rel_kormendy_cd_e_high.png')
plt.close()
####
#CONJUNTO - CORTES POR DELTA BIC & N1 LOW - cD
re_intern_low,re_extern_low=re_1_kpc[low_lim_cd],re_2_kpc[low_lim_cd]
mue_intern_low,mue_extern_low=mue_med_1[low_lim_cd],mue_med_2[low_lim_cd]

[alpha_med_1_low,beta_med_1_low],cov_1_low = np.polyfit(re_intern_low,mue_intern_low,1,cov=True)
[alpha_med_2_low,beta_med_2_low],cov_2_low = np.polyfit(re_extern_low,mue_extern_low,1,cov=True)

linha_interna_low = alpha_med_1_low * comp_12_linspace + beta_med_1_low
linha_externa_low = alpha_med_2_low * comp_12_linspace + beta_med_2_low

alpha_med_int_low,beta_med_int_low = format(alpha_med_1_low,'.3'),format(beta_med_1_low,'.3') 
alpha_med_ext_low,beta_med_ext_low = format(alpha_med_2_low,'.3'),format(beta_med_2_low,'.3')

vec_alpha=[]
vec_beta=[]
vec_erro=[]

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_intern_low,mue_intern_low,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
axs.plot(linspace_re, linha_interna_low, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
axs.scatter(re_extern_low,mue_extern_low,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
axs.plot(linspace_re, linha_externa_low, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext_low}$'+'\n'+fr'$\beta={beta_med_ext_low}$')
# axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
# axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()

# plt.savefig(f'{sample}_stats_observation_desi/p_value/kormendy_rel_comp_low.png')
plt.close()

lim_gr_cd=lim_gr[low_lim_cd]
fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
sc=axs.scatter(re_intern_low[lim_gr_cd],mue_intern_low[lim_gr_cd],marker='s',edgecolor='black',label=label_bojo,vmin=-0.5,vmax=0.5,c=slope_disk[low_lim_cd][lim_gr_cd],cmap=cmap)
axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
# axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
# axs.plot(linspace_re, linha_interna, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
# axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
# axs.plot(linspace_re, linha_externa, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label('grad e')
plt.show()
# plt.savefig(f'{sample}_stats_observation_desi/p_value/pre_kormendy_rel_comp.png')
plt.close()
######
#CONJUNTO - CORTES POR DELTA BIC & N1 HIGH - cD

re_intern_high,re_extern_high=re_1_kpc[high_lim_cd],re_2_kpc[high_lim_cd]
mue_intern_high,mue_extern_high=mue_med_1[high_lim_cd],mue_med_2[high_lim_cd]

[alpha_med_1_high,beta_med_1_high],cov_1_high = np.polyfit(re_intern_high,mue_intern_high,1,cov=True)
[alpha_med_2_high,beta_med_2_high],cov_2_high = np.polyfit(re_extern_high,mue_extern_high,1,cov=True)

linha_interna_high = alpha_med_1_high * comp_12_linspace + beta_med_1_high
linha_externa_high = alpha_med_2_high * comp_12_linspace + beta_med_2_high

alpha_med_int_high,beta_med_int_high = format(alpha_med_1_high,'.3'),format(beta_med_1_high,'.3') 
alpha_med_ext_high,beta_med_ext_high = format(alpha_med_2_high,'.3'),format(beta_med_2_high,'.3')

fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,8))
# axs.scatter(re_intern_low,mue_intern_low,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
# axs.plot(linspace_re, linha_interna_low, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
# axs.scatter(re_extern_low,mue_extern_low,marker='o',edgecolor='black',alpha=0.3,label=label_env,color='red')
# axs.plot(linspace_re, linha_externa_low, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext_low}$'+'\n'+fr'$\beta={beta_med_ext_low}$')
axs.scatter(re_intern_high,mue_intern_high,marker='s',edgecolor='black',alpha=0.3,label=label_bojo,color='blue')
axs.plot(linspace_re, linha_interna_high, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_high}$'+'\n'+fr'$\beta={beta_med_int_high}$')
axs.scatter(re_extern_high,mue_extern_high,marker='s',edgecolor='black',alpha=0.3,label=label_env,color='red')
axs.plot(linspace_re, linha_externa_high, color='red', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext_high}$'+'\n'+fr'$\beta={beta_med_ext_high}$')

# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
# axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()

# plt.savefig(f'{sample}_stats_observation_desi/p_value/kormendy_rel_comp_low.png')
plt.close()

lim_m200_low=lim_m200[low_lim_cd]
lim_veldisp_low=lim_veldisp[low_lim_cd]
lim_logmass_low=lim_casjobs[low_lim_cd]

fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,8))
sc=axs.scatter(starmass[low_lim_cd][lim_logmass_low & lim_veldisp_low],vel_disp[low_lim_cd][lim_veldisp_low & lim_logmass_low],marker='s',edgecolor='black',label=label_bojo)#c=vel_disp[low_lim_cd][lim_veldisp_low],cmap=cmap)#np.log10(h_line[low_lim_cd][lim_halpha_low])
# axs.plot(linspace_re, linha_interna_low, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
# axs.set_ylim(y2ss,y1ss)
# axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label('vel disp')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/geral/morf_class_bojo_box_{str(file_names[n-1])}.png')
plt.close()


fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,8))
sc=axs.scatter(re_intern_low[lim_veldisp_low],mue_intern_low[lim_veldisp_low],marker='s',edgecolor='black',label=label_bojo,c=vel_disp[low_lim_cd][lim_veldisp_low],cmap=cmap)#np.log10(h_line[low_lim_cd][lim_halpha_low])
axs.plot(linspace_re, linha_interna_low, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label('vel disp')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/geral/morf_class_bojo_box_{str(file_names[n-1])}.png')
plt.close()


# data_sandro=np.asarray([cluster,ra,dec]).T
# print(data_sandro[high_lim_cd][(n1[high_lim_cd]==10) & (re_intern_high<0.5)])

high_n1_lim=(n1[high_lim_cd]==10) & (re_intern_high<0.5)


fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,8))
sc=axs.scatter(re_intern_high[(n1[high_lim_cd]==10) & (re_intern_high<0.5)],mue_intern_high[(n1[high_lim_cd]==10) & (re_intern_high<0.5)],marker='s',edgecolor='black',label=label_bojo)#,vmin=2.5,vmax=8.,c=n1[high_lim_cd & n1[high_lim_cd]>8.],cmap=cmap)
axs.plot(linspace_re, linha_interna_high, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_high}$'+'\n'+fr'$\beta={beta_med_int_high}$')
axs.set_ylim(y2ss,y1ss)
axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label('n1')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/geral/morf_class_bojo_box_{str(file_names[n-1])}.png')
plt.close()

lim_veldisp_high=lim_veldisp[high_lim_cd]
lim_logmass_high=lim_casjobs[high_lim_cd]

high_n1_lim_2=high_n1_lim[lim_logmass_high & lim_veldisp_high]

fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,8))
axs.scatter(starmass[high_lim_cd][lim_logmass_high & lim_veldisp_high],vel_disp[high_lim_cd][lim_logmass_high & lim_veldisp_high],marker='s',edgecolor='black',label=label_bojo)#c=vel_disp[low_lim_cd][lim_veldisp_low],cmap=cmap)#np.log10(h_line[low_lim_cd][lim_halpha_low])
axs.scatter(starmass[high_lim_cd][lim_logmass_high & lim_veldisp_high][high_n1_lim_2],vel_disp[high_lim_cd][lim_logmass_high & lim_veldisp_high][high_n1_lim_2],marker='s',color='red',edgecolor='black',label='bololo')#c=vel_disp[low_lim_cd][lim_veldisp_low],cmap=cmap)#np.log10(h_line[low_lim_cd][lim_halpha_low])
# axs.plot(linspace_re, linha_interna_low, color='red', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int_low}$'+'\n'+fr'$\beta={beta_med_int_low}$')
# axs.set_ylim(y2ss,y1ss)
# axs.set_xlim(x1ss,x2ss)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label('vel disp')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/geral/morf_class_bojo_box_{str(file_names[n-1])}.png')
plt.close()


fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(10,8))
# axs.set_title(title_vec[n-1])
# sc=axs.scatter(delta_bic_obs[idx_faixas==n],p_value[idx_faixas==n],c=bt_vec_12[idx_faixas==n],edgecolor='black',vmin=0,vmax=1,cmap=cmap)
axs.scatter(delta_bic_obs,p_value,color='white',edgecolor='black',alpha=0.9)
axs.scatter(delta_bic_obs[high_lim_cd][high_n1_lim],p_value[high_lim_cd][high_n1_lim],color='red',edgecolor='black',alpha=0.9)

axs.axhline(0.32,label=r'$1 \sigma$')
# axs.scatter(z_cuts,vec_cuts,color='red',edgecolor='black')
line_cmap=plt.colormaps['jet']
line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
#for i,corte in enumerate(vec_cuts):
axs.axvline(vec_cuts[-1],c=line_colors[0],label=f'{str(vec_cuts[-1])}')
axs.legend()
axs.set_xlabel(r'$\Delta BIC$')
axs.set_ylabel(r'$P_{value}$')
# axs.set_xlim(min(redshift),max(redshift))
axs.set_xlim(2000,-8000)
axs.set_ylim(0,1)
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label('B/T')

# plt.savefig(f'{sample}_stats_observation_desi/p_value/{file_names[n-1]}/p_value_delta_bic_bt.png')
plt.close(fig)
'''
############################################
#GRANDE BLOCO DA RELAÇÃO DE KORMENDY (PRE-CORTES)
#############################################
r'''
#CASO BOJO E ENV
## MUE 
alpha_bojo,beta_bojo=coef_bojo = np.polyfit(re_bojo_kpc,mue_bojo,1)
alpha_env,beta_env=coef_env = np.polyfit(re_env_kpc,mue_env,1)
alpha_s,beta_s=coef_s = np.polyfit(re_s_kpc,mue_s,1)

bojo_env_linspace = np.linspace(min(re_bojo_kpc),max(re_env_kpc), 100)

bojo_line = alpha_bojo * bojo_env_linspace + beta_bojo

env_line = alpha_env * bojo_env_linspace + beta_env

s_line = alpha_s * bojo_env_linspace + beta_s
#############
## MUE MED

alpha_med_bojo,beta_med_bojo=coef_med_bojo = np.polyfit(re_bojo_kpc,mue_med_bojo,1)
alpha_med_env,beta_med_env=coef_med_env = np.polyfit(re_env_kpc,mue_med_env,1)
alpha_med_s,beta_med_s=coef_med_s = np.polyfit(re_s_kpc,mue_med_s,1)

bojo_med_line =	 alpha_med_bojo * bojo_env_linspace + beta_med_bojo

env_med_line = alpha_med_env * bojo_env_linspace + beta_med_env

s_med_line = alpha_med_s * bojo_env_linspace + beta_med_s

#CASO COMP 1 E COMP 2
re_12_all_kpc=np.append(re_1_kpc,re_2_kpc)
## MUE 
alpha_1,beta_1=coef_comp_1 = np.polyfit(re_1_kpc,mue_1,1)
alpha_2,beta_2=coef_comp_2 = np.polyfit(re_2_kpc,mue_2,1)

comp_12_linspace = np.linspace(min(re_12_all_kpc),max(re_12_all_kpc), 100)

comp_1_line = alpha_1 * comp_12_linspace + beta_1

comp_2_line = alpha_2 * comp_12_linspace + beta_2

s_line = alpha_s * comp_12_linspace + beta_s
#############
## MUE MED

alpha_med_1,beta_med_1=coef_med_1 = np.polyfit(re_1_kpc,mue_med_1,1)
alpha_med_2,beta_med_2=coef_med_2 = np.polyfit(re_2_kpc,mue_med_2,1)

comp_1_med_line =	 alpha_med_1 * comp_12_linspace + beta_med_1

comp_2_med_line = alpha_med_2 * comp_12_linspace + beta_med_2

s_med_line = alpha_med_s * comp_12_linspace + beta_med_s

#GRÁFICOS ENVOLVENDO A RELAÇÃO DE KORMENDY
#RELAÇÃO SEM MAPA DE CORES

xlabel=r'$\log_{10} R_e (Kpc)$'
label_s='Sérsic'

#CONJUNTO PARA BOJO / ENV
label_bojo='Bojo'
label_env='Envelope'
re_intern=re_bojo_kpc
re_extern=re_env_kpc
linspace_int_ext=bojo_env_linspace
linha_interna_label='Linha Bojo'
linha_externa_label='Linha Envelope'
# linha_interna=bojo_line
# linha_externa=env_line
# linha_sersic=s_line
linha_interna=bojo_med_line
linha_externa=env_med_line
linha_sersic=s_med_line

# CONJUNTO PARA COMP 1 / COMP 2
# label_bojo='Comp 1'
# label_env='Comp 2'
# re_intern=re_1_kpc
# re_extern=re_2_kpc
# linha_interna_label='Linha comp 1'
# linha_externa_label='Linha comp 2'
# linspace_int_ext=comp_12_linspace
# linha_interna=comp_1_line
# linha_externa=comp_2_line
# linha_sersic=s_line
# linha_interna=comp_1_med_line
# linha_externa=comp_2_med_line
# linha_sersic=s_med_line

#CONJUNTO PARA <MU>
ylabel=r'$<\mu_e>$'
mue_intern=mue_med_bojo
mue_extern=mue_med_env
mue_sersic=mue_med_s

#CONJUNTO PARA MU
# ylabel=r'$\mu_e$'
# mue_intern=mue_bojo
# mue_extern=mue_env
# mue_sersic=mue_s

save_path=path_bojo_env_mue_med
# save_path=path_bojo_env_mue
# save_path=path_12_mue
# save_path=path_12_mue_med

#################################
if sample == 'L07':
	fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
	axs[0].set_title('E')
	axs[0].scatter(re_intern[elip_cut],mue_intern[elip_cut],marker='s',edgecolor='black',label=label_bojo,color='green')
	# axs[0].scatter(re_extern[elip_cut],mue_extern[elip_cut],marker='o',edgecolor='black',label=label_env,color='red')
	# axs[0].scatter(re_s_kpc[elip_cut],mue_sersic[elip_cut],marker='d',edgecolor='black',label='Sérsic',color='blue')
	x1,x2=axs[0].set_xlim()
	y1,y2=axs[0].set_ylim()
	axs[0].set_xlabel(xlabel)
	axs[0].set_ylabel(ylabel)
	axs[0].yaxis.set_inverted(True)
	axs[0].legend()

	axs[1].set_title('cD')
	axs[1].scatter(re_intern[cd_cut],mue_intern[cd_cut],marker='s',edgecolor='black',label=label_bojo,color='green')
	# axs[1].scatter(re_extern[cd_cut],mue_extern[cd_cut],marker='o',edgecolor='black',label=label_env,color='red')
	# axs[1].scatter(re_s_kpc[cd_cut],mue_sersic[cd_cut],marker='d',edgecolor='black',label='Sérsic',color='blue')
	axs[1].set_xlim(x1,x2)
	axs[1].set_ylim(y1,y2)
	axs[1].set_xlabel(xlabel)
	axs[1].set_ylabel(ylabel)
	axs[1].yaxis.set_inverted(True)
	axs[1].legend()

	axs[2].set_title('E/cD')
	axs[2].scatter(re_intern[ecd_cut | cde_cut],mue_intern[ecd_cut | cde_cut],marker='s',edgecolor='black',label=label_bojo,color='green')
	# axs[2].scatter(re_extern[ecd_cut | cde_cut],mue_extern[ecd_cut | cde_cut],marker='o',edgecolor='black',label=label_env,color='red')
	# axs[2].scatter(re_s_kpc[ecd_cut | cde_cut],mue_sersic[ecd_cut | cde_cut],marker='d',edgecolor='black',label='Sérsic',color='blue')
	axs[2].set_xlim(x1,x2)
	axs[2].set_ylim(y1,y2)
	axs[2].set_xlabel(xlabel)
	axs[2].set_ylabel(ylabel)
	axs[2].yaxis.set_inverted(True)
	axs[2].legend()

	
	# plt.savefig(f'{save_path}/kormendy_relation_l07_dimm.png')
	plt.close()

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,color='green')
axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,color='red')
axs.scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',color='blue')
x1,x2=axs.set_xlim()
y1,y2=axs.set_ylim()
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.yaxis.set_inverted(True)
axs.legend()
plt.savefig(f'{save_path}/kormendy_relation_dimm.png')
plt.close()

#RELAÇÃO DE KORMENDY PARA AS MORFOLOGIAS

fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))

axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,color='blue')
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,color='red')
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',color='green')
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()
plt.savefig(f'{save_path}/kormendy_relation_subplots_vline_dimm.png')
plt.close()


#DELTA BIC (BOJO,ENVELOPE,SERSIC)
fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))
axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c=delta_bic_obs,vmin=-5000,vmax=500,cmap=cmap)
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c=delta_bic_obs,vmin=-5000,vmax=500,cmap=cmap)
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',c=delta_bic_obs,vmin=-5000,vmax=500,cmap=cmap)
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()
cbar = plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$\Delta BIC$')
plt.savefig(f'{save_path}/kormendy_relation_delta_bic_vline_dimm.png')
plt.close()

fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c='green',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c='red',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs.scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sersic',c='orange',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs.plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs.plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs.plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.yaxis.set_inverted(True)
axs.legend()
plt.savefig(f'{save_path}/kormendy_relation_lines_dimm.png')
plt.close()


# fig,axs=plt.subplots(1,2,sharey=True,figsize=(10,10))
# axs[0].set_title(r'$\mu_e$')
# axs[0].scatter(re_1_kpc,mue_1,marker='s',edgecolor='black',label='Comp 1',c='green',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[0].scatter(re_2_kpc,mue_2,marker='o',edgecolor='black',label='Comp 2',c='red',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[0].scatter(re_s_kpc,mue_s,marker='d',edgecolor='black',label='Sersic',c='orange',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[0].plot(comp_12_linspace, comp_1_line, color='black', linestyle='-', label='Linha Comp 1')
# axs[0].plot(comp_12_linspace, comp_2_line, color='black', linestyle='--', label='Linha comp 2')
# axs[0].plot(comp_12_linspace, s_line, color='black', linestyle='-.', label='Linha Sersic')
# axs[0].set_xlabel(r'$R_e$')
# axs[0].set_ylabel(r'$\mu_e$')
# axs[0].yaxis.set_inverted(True)
# axs[0].legend()

# axs[1].set_title(r'$<\mu_e>$')
# axs[1].scatter(re_1_kpc,mue_med_1,marker='s',edgecolor='black',label='Comp 1',c='green',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[1].scatter(re_2_kpc,mue_med_2,marker='o',edgecolor='black',label='Comp 2',c='red',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[1].scatter(re_s_kpc,mue_med_s,marker='d',edgecolor='black',label='Sersic',c='orange',alpha=0.3)#,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
# axs[1].plot(comp_12_linspace, comp_1_med_line, color='black', linestyle='-', label='Linha Comp 1')
# axs[1].plot(comp_12_linspace, comp_2_med_line, color='black', linestyle='--', label='Linha comp 2')
# axs[1].plot(comp_12_linspace, s_med_line, color='black', linestyle='-.', label='Linha Sersic')
# axs[1].set_xlabel(r'$R_e$')
# axs[1].set_ylabel(r'$<\mu_e>$')
# axs[1].yaxis.set_inverted(True)
# axs[1].legend()
# plt.savefig(f'{paths_sons[0][1]}/kormendy_relation_lines_mu_mumed_dimm.png')
# plt.close()

#INDICE DE SÉRSIC (BOJO,ENVELOPE,SERSIC)

fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))

axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c=n_bojo,vmin=0.5,vmax=10,cmap=cmap)
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c=n_env,vmin=0.5,vmax=10,cmap=cmap)
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',c=n_s,vmin=0.5,vmax=10,cmap=cmap)
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()
cbar = plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n$')
plt.savefig(f'{save_path}/kormendy_relation_n_vline_dimm.png')
plt.close()

#B/T (BOJO,ENVELOPE,SERSIC)

fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))
axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='o',edgecolor='black',label='Sérsic',c=bt_vec_ss,vmin=0.,vmax=1,cmap=cmap)
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()

cbar = plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$B/T$')
plt.savefig(f'{save_path}/kormendy_relation_bt_vline_dimm.png')
plt.close()

#ELIPTICIDADE (BOJO,ENVELOPE,SERSIC)

fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))
axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c=e_bojo,vmin=0.,vmax=0.9,cmap=cmap)
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c=e_env,vmin=0.,vmax=0.9,cmap=cmap)
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',c=e_s,vmin=0.,vmax=0.9,cmap=cmap)
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()
cbar = plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$n$')

plt.savefig(f'{save_path}/kormendy_relation_elip_vline_dimm.png')
plt.close()

#BOX (BOJO,ENVELOPE,SERSIC)

fig,axs=plt.subplots(1,3,sharey=True,figsize=(15,5))
axs[0].scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,c=box_bojo,vmin=-0.25,vmax=0.25,cmap=cmap)
axs[0].plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
axs[0].set_ylim(y2,y1)
axs[0].set_xlim(x1,x2)
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend()

axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,c=box_env,vmin=-0.25,vmax=0.25,cmap=cmap)
axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
axs[1].set_ylim(y2,y1)
axs[1].set_xlim(x1,x2)
axs[1].set_xlabel(xlabel)
axs[1].set_ylabel(ylabel)
axs[1].legend()

sc=axs[2].scatter(re_s_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',c=box_s,vmin=-0.25,vmax=0.25,cmap=cmap)
axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
axs[2].set_ylim(y2,y1)
axs[2].set_xlim(x1,x2)
axs[2].set_xlabel(xlabel)
axs[2].set_ylabel(ylabel)
axs[2].legend()
cbar = plt.colorbar(sc, ax=axs[2])
cbar.set_label(r'$B$')
plt.savefig(f'{save_path}/kormendy_relation_box_vline_dimm.png')
plt.close()
'''
###################################################
#MAPA de cores no plano de razão de raios (bojo e envelope) X n_env
###################################################
r'''
#CONJUNTOS
##CASO BOJO/ENV
# vec_x=re_ratio_ss
# vec_x=np.log10(re_ratio_ss)
# vec_y=n_env
# # labelx=r'$R_{bojo}/R_{env}$'
# labelx=r'$\log_{10} (R_{bojo}/R_{env})$'
# labely=r'$n_{env}$'
# bt_ratio,n_interno,n_int_ext=bt_vec_ss,n_bojo,n_ratio_ss
# bt_ratio_label,n_interno_label,n_int_ext_label=r'$B/T$',r'$n_{bojo}$',r'$n_{bojo}/n_{env}$'
# save_path_geral=path_bojo_env_geral_linear
# save_path_subplot=path_bojo_env_subplot_linear
# save_path_geral=path_bojo_env_geral_vlog
# save_path_subplot=path_bojo_env_subplot_vlog

##CASO COMP 1 /COMP 2
# vec_x=re_ratio_12
vec_x=np.log10(re_ratio_12)
vec_y=n2
# labelx=r'$R_{1}/R_{2}$'
labelx=r'$\log_{10} (R_{1}/R_{2})$'
labely=r'$n_{2}$'
bt_ratio,n_interno,n_int_ext=bt_vec_12,n1,n_ratio_12
bt_ratio_label,n_interno_label,n_int_ext_label=r'$B/T_{12}$',r'$n_{1}$',r'$n_{1}/n_{2}$'
# save_path_geral=path_12_geral_linear
# save_path_subplot=path_12_subplot_linear
save_path_geral=path_12_geral_vlog
# save_path_subplot=path_12_subplot_vlog

############
#ÚNICO
##BT

fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=bt_ratio,edgecolor='black',vmin=0,vmax=1,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(bt_ratio_label)
plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_bt_geral.png')
plt.close()

#N_bojo
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=n_interno,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(n_interno_label)

plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_n_bojo_geral.png')
plt.close()

#DELTA BIC
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=delta_bic_obs,edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$\Delta BIC$')
plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_delta_bic_geral.png')
plt.close()

#N_bojo/N_env
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=n_int_ext,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(n_int_ext_label)
plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_n_ratio_geral.png')
plt.close()

#RFF_S 
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=np.log10(rff_s),edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$RFF_s$')
plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_rff_geral.png')
plt.close()

#RFF_SS/RFF_S 
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(vec_x,vec_y,c=rff_ratio,edgecolor='black',vmax=1.5,cmap=cmap)
axs.set_xlabel(labelx)
axs.set_ylabel(labely)
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$RFF_{ss}/RFF_s$')
plt.savefig(f'{save_path_geral}/scatter_re_ratio_n_env_rff_ratio_geral.png')
plt.close()

#2x2 -- SUBPLOTS
##BT

# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=bt_ratio,edgecolor='black',vmin=0,vmax=1,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=bt_ratio[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=bt_ratio[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()
# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=bt_ratio[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(bt_ratio_label)
# plt.savefig(f'{save_path_subplot}/scatter_re_ratio_n_env_bt.png')
# plt.close()

# ##N_BOJO
# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=n_interno,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=n_interno[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=n_interno[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()

# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=n_interno[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(n_interno_label)
# plt.savefig(f'{save_path_subplot}/scatter_re_n_env_bt_n_bojo.png')
# plt.close()

# ##N_BOJO/N_ENV

# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=n_int_ext,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=n_int_ext[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=n_int_ext[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()
# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=n_int_ext[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(n_int_ext_label)
# plt.savefig(f'{save_path_subplot}/scatter_re_n_env_bt_n_ratio.png')
# plt.close()

# ##DELTA BIC

# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=delta_bic_obs,edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=delta_bic_obs[elip_cut],marker='s',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=delta_bic_obs[cd_cut],marker='d',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()
# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=delta_bic_obs[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(r'$\Delta BIC$')
# plt.savefig(f'{save_path_subplot}/scatter_re_ratio_n_env_delta_bic.png')
# plt.close()

# #RFF OBSERVADO -- MODELO S

# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=np.log10(rff_s),edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=np.log10(rff_s[elip_cut]),marker='s',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=np.log10(rff_s[cd_cut]),marker='d',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()
# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=np.log10(rff_s[ecd_cut | cde_cut]),marker='o',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()
# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(r'$\log_{10} RFF_s$')
# plt.savefig(f'{save_path_subplot}/scatter_re_ratio_n_env_rff_s.png')
# plt.close()

# #RFF_SS/RFF_S

# fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
# sc=axs[0,0].scatter(vec_x,vec_y,c=rff_ratio,edgecolor='black',vmax=1.5,cmap=cmap)
# axs[0,0].set_ylabel(labely)
# axs[0,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[0,1].scatter(vec_x[elip_cut],vec_y[elip_cut],c=rff_ratio[elip_cut],marker='s',edgecolor='black',vmax=1.5,cmap=cmap,label='E')
# axs[0,1].set_ylabel(labely)
# axs[0,1].legend()
# axs[1,0].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,0].scatter(vec_x[cd_cut],vec_y[cd_cut],c=rff_ratio[cd_cut],marker='d',edgecolor='black',vmax=1.5,cmap=cmap,label='cD')
# axs[1,0].set_xlabel(labelx)
# axs[1,0].set_ylabel(labely)
# axs[1,0].legend()

# axs[1,1].scatter(vec_x,vec_y,c='white',edgecolor='black',alpha=0.4)
# axs[1,1].scatter(vec_x[ecd_cut | cde_cut],vec_y[ecd_cut | cde_cut],c=rff_ratio[ecd_cut | cde_cut],marker='o',edgecolor='black',vmax=1.5,cmap=cmap,label='E/cD')
# axs[1,1].set_xlabel(labelx)
# axs[1,1].set_ylabel(labely)
# axs[1,1].legend()

# cbar=plt.colorbar(sc, ax=axs)
# cbar.set_label(r'$RFF_{ss}/RFF_s$')
# plt.savefig(f'{save_path_subplot}/scatter_re_ratio_n_env_rff_ratio.png')
# plt.close()
'''
###################################################
#MAPA de cores no plano de razão de raios (re1/re2) X n_2
r'''
cmap=plt.colormaps['hot']
#ÚNICO
##BT

fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=bt_vec_12,edgecolor='black',vmin=0,vmax=1,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$B/T (12)$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n2_bt_12_geral.png')
plt.close()

#N_bojo
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=n1,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$n_{2}$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_1_n_2_geral.png')
plt.close()

#DELTA BIC
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=delta_bic_obs,edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$\Delta BIC$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_delta_bic_12_geral.png')
plt.close()

#N_bojo/N_env
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=n_ratio_12,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$n_{1}/n_{2}$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_n_ratio_12_geral.png')
plt.close()

#RFF_S 
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=np.log10(rff_s),edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$RFF_s$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_rff_12_geral.png')
plt.close()

#RFF_SS/RFF_S 
fig,axs=plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
sc=axs.scatter(np.log10(re_ratio_12),n2,c=rff_ratio,edgecolor='black',vmax=1.5,cmap=cmap)
axs.set_xlabel(r'$R_{1}/R_{2}$')
axs.set_ylabel(r'$n_{2}$')#B/T (S+S)$')
cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$RFF_{ss}/RFF_s$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_rff_ratio_12_geral.png')
plt.close()

#2x2
##BT

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=bt_vec_12,edgecolor='black',vmin=0,vmax=1,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=bt_vec_12[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=bt_vec_12[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=bt_vec_12[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=1,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$B/T (12)$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_bt_12.png')
plt.close()


##N_BOJO

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=n1,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=n1[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n1[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=n1[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$n_{2}$')

plt.savefig('L07_stats_observation_desi/scatter_re_n_2_n_1.png')
plt.close()

##N_BOJO/N_ENV

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=n_ratio_12,edgecolor='black',vmin=0,vmax=8,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=n_ratio_12[elip_cut],marker='s',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n_ratio_12[cd_cut],marker='d',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=n_ratio_12[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=0,vmax=8,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$n_{1}/n_{2}$')

plt.savefig('L07_stats_observation_desi/scatter_re_n_2_n_ratio_12.png')
plt.close()

##DELTA BIC

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=delta_bic_obs,edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=delta_bic_obs[elip_cut],marker='s',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=delta_bic_obs[cd_cut],marker='d',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=delta_bic_obs[ecd_cut | cde_cut],marker='o',edgecolor='black',vmin=-8000,vmax=2000,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$\Delta BIC$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_delta_bic_12.png')
plt.close()

#RFF OBSERVADO -- MODELO S

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=np.log10(rff_s),edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=np.log10(rff_s[elip_cut]),marker='s',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=np.log10(rff_s[cd_cut]),marker='d',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=np.log10(rff_s[ecd_cut | cde_cut]),marker='o',edgecolor='black',vmin=-2.5,vmax=-0.7,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$Log_{10} RFF_s$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_rff_s_12.png')
plt.close()

#RFF_SS/RFF_S

fig,axs=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10),constrained_layout=True)
sc=axs[0,0].scatter(np.log10(re_ratio_12),n2,c=rff_ratio,edgecolor='black',vmax=1.5,cmap=cmap)
# axs[0,0].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')

axs[0,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[0,1].scatter(np.log10(re_ratio_12[elip_cut]),n2[elip_cut],c=rff_ratio[elip_cut],marker='s',edgecolor='black',vmax=1.5,cmap=cmap,label='E')
# axs[0,1].set_xlabel(r'$R_{bojo}/R_{env}$')
axs[0,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[0,1].legend()

axs[1,0].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,0].scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=rff_ratio[cd_cut],marker='d',edgecolor='black',vmax=1.5,cmap=cmap,label='cD')
axs[1,0].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,0].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,0].legend()

axs[1,1].scatter(np.log10(re_ratio_12),n2,c='white',edgecolor='black',alpha=0.4)
axs[1,1].scatter(np.log10(re_ratio_12[ecd_cut | cde_cut]),n2[ecd_cut | cde_cut],c=rff_ratio[ecd_cut | cde_cut],marker='o',edgecolor='black',vmax=1.5,cmap=cmap,label='E/cD')
axs[1,1].set_xlabel(r'$R_{1}/R_{2}$')
axs[1,1].set_ylabel(r'$n_{2}$')#B/T (S+S)$')
axs[1,1].legend()

cbar=plt.colorbar(sc, ax=axs)
cbar.set_label(r'$RFF_{ss}/RFF_s$')

plt.savefig('L07_stats_observation_desi/scatter_re_ratio_n_2_rff_ratio_12.png')
plt.close()
'''
#########################################
#separação e preparação das faixas de redshift para a contrução do 
#corte em delta bic e redshift
r'''
faixas=np.percentile(redshift,np.linspace(0,90,10))
idx_faixas=np.digitize(redshift,faixas,right=False)
bins_med=[]
for z in range(len(faixas)-1):
	bins_med.append(np.round((faixas[z]+faixas[z+1])/2,3))

num=np.unique(idx_faixas)

bins_med=np.asarray([np.average(redshift[idx_faixas==idx]) for idx in num],dtype=float)

p5_vec=np.asarray([np.median(delta_bic_sim[idx_faixas==idx])-2*iqr(delta_bic_sim[idx_faixas==idx],scale='normal') for idx in num],dtype=float)
# p5_vec_psf=np.asarray([np.median(delta_bic_sim_psf[idx_faixas==idx])-2*iqr(delta_bic_sim_psf[idx_faixas==idx],scale='normal') for idx in num],dtype=float)
# p5_vec=np.asarray([np.percentile(delta_bic_sim[idx_faixas==idx],5) for idx in num],dtype=float)
# test = PySRRegressor(niterations=1000,unary_operators=["exp", "log",'log10','sqrt','log2'],binary_operators=["+", "-", "*", "/"])
# X=bins_med.reshape(-1,1)
# test.fit(X,p5_vec)

# pov,cov=scp.curve_fit(log_model,bins_med,p5_vec,p0=[-13.9,11.55])
# pov_psf,cov_psf=scp.curve_fit(log_model,bins_med,p5_vec_psf,p0=[-13.9,11.55])

# [-13.99127766  11.55777504]
#GRAFICO DA SEPARAÇÃO DE REDSHIFT REPLICAÇÃO DO GRAFICO DO PROFESSOR
# lim=(n1 < 2.5) & (n2 > 2.5)

# bins_delta=np.arange(0,1.05,0.05)#0.01,0.07,0.004)#-7500,1000,200)#np.arange(min(delta_bic[h]),max(delta_bic[h]),200)
# fig=plt.figure()
# ax=fig.add_subplot()
# ax.hist(rff_ss/rff_s,density=True,bins=bins_delta, alpha=0.6)
# ax.hist(rff_ss[lim]/rff_s[lim],histtype='step',density=True,color='red',bins=bins_delta)
# ax.set_xlabel('RFF')#r'$Delta BIC$')
# # ax.set_xlim(-8000,2000)
# plt.legend()

# # plt.savefig(f'{sample}_stats_observation_desi/best_ncomp.png')
# plt.close(fig)


fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(10,8))
axs.scatter(redshift,delta_bic_obs,color='white',edgecolor='black',label='Obs S',alpha=0.4)
# axs.scatter(redshift[lim],delta_bic_obs[lim],color='red',edgecolor='black',label='Obs S')

axs.scatter(bins_med,p5_vec,color='black',label=r'$5\%$')
x1,x2=axs.set_xlim()
xz=np.linspace(x1,x2,100)
axs.plot(xz,log_model(xz,*pov))
axs.legend()
axs.set_xlabel(r'$z$')
axs.set_ylabel(r'$\Delta BIC$')
axs.set_ylim(2000,-8000)

# axs[1].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S (psf)')
# # axs[1].scatter(redshift,delta_bic_obs,color='black',edgecolor='black',label='L07')
# axs[1].scatter(bins_med,p5_vec_psf,color='red',label=r'$5\%$')
# axs[1].plot(xz,log_model(xz,*pov_psf))
# axs[1].legend()
# axs[1].set_xlabel(r'$z$')
# axs[1].set_ylabel(r'$\Delta BIC$')
# axs[1].set_ylim(2000,-8000)
# plt.savefig('L07_stats_observation_desi/delta_bic_sim_full.png')
plt.show()
plt.close()

# fig,axs=plt.subplots(2,2,tight_layout=True,figsize=(10,8))
# axs[0,0].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',label='Simulação S')
# axs[0,0].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# x1,x2=axs[0,0].set_xlim()
# xz=np.linspace(x1,x2,100)
# axs[0,0].plot(xz,log_model(xz,*pov))
# axs[0,0].legend()
# axs[0,0].set_xlabel(r'$z$')
# axs[0,0].set_ylabel(r'$\Delta BIC$')
# axs[0,0].set_ylim(100,-100)

# axs[0,1].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S')
# axs[0,1].scatter(redshift[elip_cut],delta_bic_obs[elip_cut],color='green',edgecolor='black',label='E')
# axs[0,1].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# axs[0,1].plot(xz,log_model(xz,*pov))
# axs[0,1].legend()
# axs[0,1].set_xlabel(r'$z$')
# axs[0,1].set_ylabel(r'$\Delta BIC$')
# axs[0,1].set_ylim(100,-100)

# axs[1,0].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S')
# axs[1,0].scatter(redshift[ecd_cut | cde_cut],delta_bic_obs[ecd_cut | cde_cut],color='orange',edgecolor='black',label='E/cD')
# axs[1,0].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# axs[1,0].plot(xz,log_model(xz,*pov))
# axs[1,0].legend()
# axs[1,0].set_xlabel(r'$z$')
# axs[1,0].set_ylabel(r'$\Delta BIC$')
# axs[1,0].set_ylim(100,-100)

# axs[1,1].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S')
# axs[1,1].scatter(redshift[cd_cut],delta_bic_obs[cd_cut],color='blue',edgecolor='black',label='cD')
# axs[1,1].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# axs[1,1].plot(xz,log_model(xz,*pov))
# axs[1,1].legend()
# axs[1,1].set_xlabel(r'$z$')
# axs[1,1].set_ylabel(r'$\Delta BIC$')
# axs[1,1].set_ylim(100,-100)
# plt.savefig('L07_stats_observation_desi/delta_bic_sim_close.png')
# plt.close()

# fig,axs=plt.subplots(1,2,tight_layout=True,figsize=(10,8))
# axs[0].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',label='Simulação S')
# axs[0].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# x1,x2=axs[0].set_xlim()
# xz=np.linspace(x1,x2,100)
# axs[0].plot(xz,log_model(xz,*pov))
# axs[0].legend()
# axs[0].set_xlabel(r'$z$')
# axs[0].set_ylabel(r'$\Delta BIC$')
# axs[0].set_ylim(200,-200)

# axs[1].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S')
# axs[1].scatter(redshift,delta_bic_obs,color='black',edgecolor='black',label='L07')
# axs[1].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# axs[1].plot(xz,log_model(xz,*pov))
# axs[1].legend()
# axs[1].set_xlabel(r'$z$')
# axs[1].set_ylabel(r'$\Delta BIC$')
# axs[1].set_ylim(200,-200)
# plt.savefig('L07_stats_observation_desi/delta_bic_sim_mid_close.png')
# plt.close()

# fig,axs=plt.subplots(1,2,tight_layout=True,figsize=(10,8))
# axs[0].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',label='Simulação S')
# axs[0].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# x1,x2=axs[0].set_xlim()
# xz=np.linspace(x1,x2,100)
# axs[0].plot(xz,log_model(xz,*pov))
# axs[0].legend()
# axs[0].set_xlabel(r'$z$')
# axs[0].set_ylabel(r'$\Delta BIC$')
# axs[0].set_ylim(2000,-8000)

# axs[1].scatter(redshift,delta_bic_sim,color='white',edgecolor='black',alpha=0.4,label='Simulação S')
# axs[1].scatter(redshift,delta_bic_obs,color='black',edgecolor='black',label='L07')
# axs[1].scatter(bins_med,p5_vec,color='red',label=r'$5\%$')
# axs[1].plot(xz,log_model(xz,*pov))
# axs[1].legend()
# axs[1].set_xlabel(r'$z$')
# axs[1].set_ylabel(r'$\Delta BIC$')
# axs[1].set_ylim(2000,-8000)
# plt.savefig('L07_stats_observation_desi/delta_bic_sim_full.png')
# plt.close()
'''
#############################################
####################################################
#PRÉ ANÁLISE FEITA POR VALORES DE REDSHIFT COM AS DISTRIBUIÇÕES DO P VALUE
#PRÉ-CORTE EM DELTA BIC
r'''
for n in num:
	# os.makedirs(f'{sample}_stats_observation_desi/p_value/kormendy_rel',exist_ok=True)
	z_sub_sample_only=idx_faixas==n
	# z_cuts.append(np.average(redshift[z_sub_sample_only]))
	lim1=rff_ratio[z_sub_sample_only]>0.9
	delta_bic_obs_sub_sample=delta_bic_obs[z_sub_sample_only]
	delta_bic_obs_09=delta_bic_obs[z_sub_sample_only][lim1]
	delta_bic_obs_res=delta_bic_obs[z_sub_sample_only][~lim1]
	cut_lim=np.median(delta_bic_obs_09)-mad(delta_bic_obs_09)*1.4826*3.
	# vec_cuts.append(cut_lim)

	# cluster_n1=cluster[n1_small_cd]
	# ra_n1=ra[z_sub_sample_cd & (n1<2.5)]
	# dec_n1=dec[z_sub_sample_cd & (n1<2.5)]
	# redshift_n1=redshift[z_sub_sample_cd & (n1<2.5)]
	# make_image(sample,cluster_n1,n-1)
	# data_n1=np.asarray([cluster_n1,ra_n1,dec_n1,redshift_n1]).T
	# np.savetxt(f'{sample}_stats_observation_desi/data_indiv_n1_{n-1}.dat',data_n1,fmt='%s',newline='\n')
	#######

	z_sub_sample_elip=z_sub_sample_only & (delta_bic_obs>cut_lim)
	z_sub_sample_cd=z_sub_sample_only & (delta_bic_obs<cut_lim)
	n1_small=z_sub_sample_cd & (n1<2.5)
	n1_large=z_sub_sample_cd & (n1>2.5)
	n1_small_and_e=n1_small | z_sub_sample_elip
	####
	xlabel=r'$\log_{10} R_e (Kpc)$'
	ylabel=r'$<\mu_e>$'
	title_z=fr'{min(redshift[z_sub_sample_only])} $\leq$ z < {max(redshift[z_sub_sample_only])}'

	label_cd='cD'
	label_elip='E'
	linha_cd_label='Linha cD'
	linha_elip_label='Linha E'

	label_bojo='Comp 1'
	label_env='Comp 2'
	linha_interna_label='Linha comp 1'
	linha_externa_label='Linha comp 2'
	####

	#GERAL (por redshift) 

	##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & cD)
	re_sersic_cd_kpc,re_sersic_kpc=re_s_kpc[z_sub_sample_cd],re_s_kpc[z_sub_sample_elip]
	mue_sersic_cd,mue_sersic=mue_med_s[z_sub_sample_cd],mue_med_s[z_sub_sample_elip]

	re_12_all_kpc=np.append(re_sersic_cd_kpc,re_sersic_kpc)
	linspace_re= comp_12_linspace = np.linspace(min(re_12_all_kpc),max(re_12_all_kpc), 100)
	
	[alpha_cd,beta_cd],cov_cd = np.polyfit(re_sersic_cd_kpc,mue_sersic_cd,1,cov=True)
	[alpha_s,beta_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

	linha_cd = alpha_cd * comp_12_linspace + beta_cd
	linha_sersic = alpha_s * comp_12_linspace + beta_s

	alpha_cd_label,beta_cd_label = format(alpha_cd,'.3'),format(beta_cd,'.3') 
	alpha_ser_label,beta_ser_label = format(alpha_s,'.3'),format(beta_s,'.3')

	fig,axs=plt.subplots(1,1,sharey=True,sharex=True,figsize=(15,5))
	axs.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
	axs.plot(linspace_re, linha_cd, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
	axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
	axs.plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_ser_label}$'+'\n'+fr'$\beta={beta_ser_label}$')
	axs.set_ylim(y2s,y1s)
	axs.set_xlim(x1s,x2s)
	axs.set_xlabel(xlabel)
	axs.set_ylabel(ylabel)
	axs.legend()
	plt.close()

	fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
	axs[0].scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
	axs[0].plot(linspace_re, linha_cd, color='red', linestyle='-')
	axs[0].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.3,label='Sérsic',color='green')
	axs[0].plot(linspace_re, linha_sersic, color='red', linestyle='-.')
	axs[0].set_ylim(y2s,y1s)
	axs[0].set_xlim(x1s,x2s)
	axs[0].set_xlabel(xlabel)
	axs[0].set_ylabel(ylabel)
	axs[0].legend()

	axs[1].set_title(title_z,fontsize=15)
	axs[1].scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.5,label=label_cd,color='blue')
	axs[1].plot(linspace_re, linha_cd, color='red', linestyle='-', label=linha_cd_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
	axs[1].set_ylim(y2s,y1s)
	axs[1].set_xlim(x1s,x2s)
	axs[1].set_xlabel(xlabel)
	axs[1].set_ylabel(ylabel)
	axs[1].legend()

	axs[2].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',alpha=0.5,label='Sérsic',color='green')
	axs[2].plot(linspace_re, linha_sersic, color='red', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_ser_label}$'+'\n'+fr'$\beta={beta_ser_label}$')
	axs[2].set_ylim(y2s,y1s)
	axs[2].set_xlim(x1s,x2s)
	axs[2].set_xlabel(xlabel)
	axs[2].set_ylabel(ylabel)
	axs[2].legend()

	
	# plt.savefig(f'{sample}_stats_observation_desi/p_value/{str(file_names[n-1])}/rel_kormendy_s_e_cd_{str(file_names[n-1])}.png')
	# plt.savefig(f'{sample}_stats_observation_desi/p_value/rel_kormendy/rel_kormendy_s_e_cd_{str(file_names[n-1])}.png')
	plt.close()

	##COMPONENTES
	re_intern,re_extern,re_sersic_kpc=re_1_kpc[z_sub_sample_cd],re_2_kpc[z_sub_sample_cd],re_s_kpc[z_sub_sample_elip]
	mue_intern,mue_extern,mue_sersic=mue_med_1[z_sub_sample_cd],mue_med_2[z_sub_sample_cd],mue_med_s[z_sub_sample_elip]

	[alpha_med_1,beta_med_1],cov_1 = np.polyfit(re_intern,mue_intern,1,cov=True)
	[alpha_med_2,beta_med_2],cov_2 = np.polyfit(re_extern,mue_extern,1,cov=True)
	[alpha_med_s,beta_med_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

	linha_interna = alpha_med_1 * comp_12_linspace + beta_med_1
	linha_externa = alpha_med_2 * comp_12_linspace + beta_med_2
	linha_sersic = alpha_med_s * comp_12_linspace + beta_med_s
	###

	vec_erro.append((np.sqrt(np.diag(cov_1)),np.sqrt(np.diag(cov_2)),np.sqrt(np.diag(cov_s))))
	alpha_med_int,beta_med_int = format(alpha_med_1,'.3'),format(beta_med_1,'.3') 
	alpha_med_ext,beta_med_ext = format(alpha_med_2,'.3'),format(beta_med_2,'.3')
	alpha_med_ser,beta_med_ser = format(alpha_med_s,'.3'),format(beta_med_s,'.3')
	vec_alpha.append((alpha_med_int,alpha_med_ext,alpha_med_ser))
	vec_beta.append((beta_med_int,beta_med_ext,beta_med_ser))

	#SEPARAÇÃO COM N1
	##cD(SOMENTE N1) - E - SÉRSIC
	re_sersic_cd_kpc_n1,re_sersic_kpc=re_s_kpc[n1_small],re_s_kpc[z_sub_sample_elip]
	mue_sersic_cd_n1,mue_sersic=mue_med_s[n1_small],mue_med_s[z_sub_sample_elip]

	[alpha_med_cd_n1,beta_med_cd_n1],cov_cd_n1 = np.polyfit(re_sersic_cd_kpc_n1,mue_sersic_cd_n1,1,cov=True)
	[alpha_med_s,beta_med_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

	linha_cd_n1 = alpha_med_cd_n1 * comp_12_linspace + beta_med_cd_n1
	linha_sersic = alpha_med_s * comp_12_linspace + beta_med_s

	##cD(SEM N1) - E(COM AS N1) - SÉRSIC
	re_sersic_cd_kpc_clean,re_sersic_kpc_n1=re_s_kpc[n1_large],re_s_kpc[n1_small_and_e]
	mue_sersic_cd_clean,mue_sersic_n1=mue_med_s[n1_large],mue_med_s[n1_small_and_e]

	[alpha_med_cd_clean,beta_med_cd_clean],cov_cd_clean = np.polyfit(re_sersic_cd_kpc_clean,mue_sersic_cd_clean,1,cov=True)
	[alpha_med_s_n1,beta_med_s_n1],cov_s_n1 = np.polyfit(re_sersic_kpc_n1,mue_sersic_n1,1,cov=True)

	linha_cd_clean = alpha_med_cd_clean * comp_12_linspace + beta_med_cd_clean
	linha_sersic = alpha_med_s_n1 * comp_12_linspace + beta_med_s_n1

	##cD(SEM N1) - cD(SOMENTE N1) - SÉRSIC

	#linhas feitas acima só plot aqui 

	##cD(SOMENTE N1) - E - COMPONENTES
	re_intern_n1,re_extern_n1,re_sersic_kpc=re_1_kpc[n1_small],re_2_kpc[n1_small],re_s_kpc[z_sub_sample_elip]
	mue_intern_n1,mue_extern_n1,mue_sersic=mue_med_1[n1_small],mue_med_2[n1_small],mue_med_s[z_sub_sample_elip]

	[alpha_med_1_n1,beta_med_1_n1],cov_1_n1 = np.polyfit(re_intern_n1,mue_intern_n1,1,cov=True)
	[alpha_med_2_n1,beta_med_2_n1],cov_2_n1 = np.polyfit(re_extern_n1,mue_extern_n1,1,cov=True)
	[alpha_med_s,beta_med_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

	linha_interna_n1 = alpha_med_1_n1 * comp_12_linspace + beta_med_1_n1
	linha_externa_n1 = alpha_med_2_n1 * comp_12_linspace + beta_med_2_n1
	linha_sersic = alpha_med_s * comp_12_linspace + beta_med_s

	##cD(SEM N1) - E(COM AS N1) - COMPONENTES
	re_intern_clean,re_extern_clean,re_sersic_kpc=re_1_kpc[n1_large],re_2_kpc[n1_large],re_s_kpc[n1_small_and_e]
	mue_intern_clean,mue_extern_clean,mue_sersic=mue_med_1[n1_large],mue_med_2[n1_large],mue_med_s[n1_small_and_e]

	[alpha_med_1_clean,beta_med_1_clean],cov_1_clean = np.polyfit(re_intern_clean,mue_intern_clean,1,cov=True)
	[alpha_med_2_clean,beta_med_2_clean],cov_2_clean = np.polyfit(re_extern_clean,mue_extern_clean,1,cov=True)
	[alpha_med_s_n1,beta_med_s_n1],cov_s_n1 = np.polyfit(re_sersic_kpc_n1,mue_sersic_n1,1,cov=True)

	linha_interna_clean = alpha_med_1_clean * comp_12_linspace + beta_med_1_clean
	linha_externa_clean = alpha_med_2_clean * comp_12_linspace + beta_med_2_clean
	linha_sersic = alpha_med_s_n1 * comp_12_linspace + beta_med_s_n1


	##cD(SEM N1) - cD(SOMENTE N1) - COMPONENTES
	#feito nas linhas de cima

	#plot do n1 dos componentes	
	# sc=axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,vmin=0.5,vmax=5.,c=n1[z_sub_sample_cd],cmap=cmap)
	# axs.plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
	# axs.set_ylim(y2,y1)
	# axs.set_xlim(x1,x2)
	# axs.set_xlabel(xlabel)
	# axs.set_ylabel(ylabel)
	# axs.legend()
	# cbar=plt.colorbar(sc, ax=axs)
	# cbar.set_label('n1')


	# fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))

	# sc=axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,vmin=-0.5,vmax=0.5,c=box1[z_sub_sample_cd],cmap=cmap)
	# axs.plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label)
	# axs.set_ylim(y2,y1)
	# axs.set_xlim(x1,x2)
	# axs.set_xlabel(xlabel)
	# axs.set_ylabel(ylabel)
	# axs.legend()
	# cbar=plt.colorbar(sc, ax=axs)
	# cbar.set_label('boxiness')

	# # axs[1].scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,color=n1_env)
	# # axs[1].plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label)
	# # axs[1].set_ylim(y2,y1)
	# # axs[1].set_xlim(x1,x2)
	# # axs[1].set_xlabel(xlabel)
	# # axs[1].set_ylabel(ylabel)
	# # axs[1].legend()

	# # sc=axs[2].scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',color=n_sersic)
	# # axs[2].plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic')
	# # axs[2].set_ylim(y2,y1)
	# # axs[2].set_xlim(x1,x2)
	# # axs[2].set_xlabel(xlabel)
	# # axs[2].set_ylabel(ylabel)
	# # axs[2].legend()

	# plt.savefig(f'{sample}_stats_observation_desi/p_value/geral/morf_class_bojo_box_{str(file_names[n-1])}.png')
	# plt.close()

	# alpha_med_int,beta_med_int = format(alpha_med_1,'.3'),format(beta_med_1,'.3') 
	# alpha_med_ext,beta_med_ext = format(alpha_med_2,'.3'),format(beta_med_2,'.3')
	# alpha_med_ser,beta_med_ser = format(alpha_med_s,'.3'),format(beta_med_s,'.3')
	# vec_alpha.append((alpha_med_int,alpha_med_ext,alpha_med_ser))
	# vec_beta.append((beta_med_int,beta_med_ext,beta_med_ser))
	# fig,axs=plt.subplots(1,1,sharey=True,figsize=(10,10))
	# axs.scatter(re_intern,mue_intern,marker='s',edgecolor='black',label=label_bojo,color='blue')
	# axs.plot(linspace_int_ext, linha_interna, color='black', linestyle='-', label=linha_interna_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
	# axs.scatter(re_extern,mue_extern,marker='o',edgecolor='black',label=label_env,color='red')
	# axs.plot(linspace_int_ext, linha_externa, color='black', linestyle='--', label=linha_externa_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
	# axs.scatter(re_sersic_kpc,mue_sersic,marker='d',edgecolor='black',label='Sérsic',color='green')
	# axs.plot(linspace_int_ext, linha_sersic, color='black', linestyle='-.', label='Linha Sersic'+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
	# axs.set_ylim(y2,y1)
	# axs.set_xlim(x1,x2)
	# axs.set_xlabel(xlabel)
	# axs.set_ylabel(ylabel)
	# axs.legend()
	# 
	# # plt.savefig(f'{sample}_stats_observation_desi/p_value/morf_class_allin_{str(file_names[n-1])}.png')
	# plt.close()

vec_alpha=np.asarray(vec_alpha,dtype=float).T
vec_beta=np.asarray(vec_beta,dtype=float).T
vec_erro=np.asarray(vec_erro).T
print(vec_alpha,'\n',vec_beta,'\n',vec_erro)
fig,axs=plt.subplots(2,1,sharex=True)
for i in range(len(vec_alpha)):
	axs[0].errorbar(z_cuts,vec_alpha[i],yerr=vec_erro[0][i],marker='o')
axs[0].set_ylabel(r'$\alpha$')
for i in range(len(vec_alpha)):
	axs[1].errorbar(z_cuts,vec_beta[i],yerr=vec_erro[1][i],marker='o')
axs[1].set_xlabel(r'z')
axs[1].set_ylabel(r'$\beta$')

fig.legend(['Bojo','Envelope','Sérsic'])
plt.show()
plt.close()

# fig,axs=plt.subplots(1,1,tight_layout=True,figsize=(10,8))
# # axs.set_title(title_vec[n-1])
# # sc=axs.scatter(delta_bic_obs[idx_faixas==n],p_value[idx_faixas==n],c=bt_vec_12[idx_faixas==n],edgecolor='black',vmin=0,vmax=1,cmap=cmap)
# axs.scatter(delta_bic_obs,p_value,color='white',edgecolor='black',alpha=0.9)
# axs.axhline(0.32,label=r'$1 \sigma$')
# # axs.scatter(z_cuts,vec_cuts,color='red',edgecolor='black')
# line_cmap=plt.colormaps['jet']
# line_colors=line_cmap(np.linspace(0,1,len(vec_cuts)))
# for i,corte in enumerate(vec_cuts):
# 	axs.axvline(corte,c=line_colors[i],label=f'{str(corte)}')
# axs.legend()
# axs.set_xlabel(r'$\Delta BIC$')
# axs.set_ylabel(r'$P_{value}$')
# # axs.set_xlim(min(redshift),max(redshift))
# axs.set_xlim(2000,-8000)
# axs.set_ylim(0,1)

# # cbar=plt.colorbar(sc, ax=axs)
# # cbar.set_label('B/T')
# 
# # plt.savefig(f'{sample}_stats_observation_desi/p_value/{file_names[n-1]}/p_value_delta_bic_bt.png')
# plt.close(fig)
'''
##############################


#GRAFICO DE COMPARAÇÃO CLASSES T1 E T2 CONTRA E,cD e E/cD
r'''
fig,axs=plt.subplots(1,2,figsize=(15,5),tight_layout=True)
axs[0].scatter(redshift_l07,delta_bic_l07,alpha=0.1)
x1,x2=axs[0].set_xlim()
xz=np.linspace(x1,x2,100)
axs[0].scatter(redshift_l07[elip_cut],delta_bic_l07[elip_cut],edgecolor='black',color='blue',label=r'$E$')
axs[0].scatter(redshift_l07[cd_cut],delta_bic_l07[cd_cut],edgecolor='black',color='orange',label=r'$cD$')
axs[0].scatter(redshift_l07[ecd_cut | cde_cut],delta_bic_l07[ecd_cut | cde_cut],edgecolor='black',color='purple',label=r'$E/cD & cD/E$')
axs[0].plot(xz,exp_func(xz,*pov),color='black',label='5%')
axs[0].legend()
axs[0].set_xlabel(r'$z$')
axs[0].set_ylabel(r'$Delta BIC$')
axs[0].set_ylim(2000,-8000)

axs[1].scatter(redshift_l07,delta_bic_l07,edgecolor='black')
x1,x2=axs[1].set_xlim()
xz=np.linspace(x1,x2,100)
axs[1].scatter(redshift_l07[bic_low_cut],delta_bic_l07[bic_low_cut],edgecolor='black',color='green',label=r'$T_1$')
axs[1].scatter(redshift_l07[bic_high_cut],delta_bic_l07[bic_high_cut],edgecolor='black',color='red',label=r'$T_2$')
axs[1].plot(xz,exp_func(xz,*pov),color='black',label='5%')
axs[1].legend()
axs[1].set_xlabel(r'$z$')
axs[1].set_ylabel(r'$Delta BIC$')
axs[1].set_ylim(2000,-8000)
plt.savefig('L07_stats_observation_desi/delta_bic_z_t1t2_morf.png')
plt.close()
'''
r'''
#RFF - ETA/ COM AS CLASSIFICAÇÕES MORFOLOGICAS

fig1,ax=plt.subplots(1,3,sharex=True,tight_layout=True,figsize=(15,5))
vec=[0.1,0.5,0.9]
for item in vec:
	ax[0].plot(np.log10(x),x-item*x,label=str(item))
	ax[1].plot(np.log10(x),x-item*x,label=str(item))
	ax[2].plot(np.log10(x),x-item*x,label=str(item))

ax[0].scatter(np.log10(rff_l07[elip_cut]),eta_l07[elip_cut],c='green',s=20,edgecolors='black',label=r'$E$')
ax[0].legend()
ax[0].set_ylabel(r'$\eta$')

ax[1].scatter(np.log10(rff_l07[cd_cut]),eta_l07[cd_cut],c='red',s=20,edgecolors='black',label=r'$cD$')
ax[1].legend()
ax[1].label_outer()

ax[2].scatter(np.log10(rff_l07[ecd_cut | cde_cut]),eta_l07[ecd_cut | cde_cut],c='black',s=20,edgecolors='black',label=r'$E/cD$')
ax[2].legend()
ax[2].label_outer()

plt.setp(ax,xlim=([-2.5,-0.7]),ylim=([-0.01,0.1]))
plt.setp(ax,xlabel=(r'$\log\,RFF$'))

plt.savefig('L07_stats_observation_desi/rffxeta_morf_types.png')
plt.close()

#RFF - ETA/ GRUPO T1 COM AS MORFOLOGIAS cD e E

vec_lim=[elip_cut & bic_low_cut,cd_cut & bic_low_cut, (ecd_cut | cde_cut) & bic_low_cut]


fig1,ax=plt.subplots(1,3,sharex=True,tight_layout=True,figsize=(15,5))
vec=[0.1,0.5,0.9]
for item in vec:
	ax[0].plot(np.log10(x),x-item*x,label=str(item))
	ax[1].plot(np.log10(x),x-item*x,label=str(item))
	ax[2].plot(np.log10(x),x-item*x,label=str(item))

ax[0].scatter(np.log10(rff_l07[vec_lim[0]]),eta_l07[vec_lim[0]],c='green',s=20,edgecolors='black',label=r'$T_1(E)$')
ax[0].legend()
ax[0].set_ylabel(r'$\eta$')

ax[1].scatter(np.log10(rff_l07[vec_lim[1]]),eta_l07[vec_lim[1]],c='red',s=20,edgecolors='black',label=r'$T_1(cD)$')
ax[1].legend()
ax[1].label_outer()

ax[2].scatter(np.log10(rff_l07[vec_lim[2]]),eta_l07[vec_lim[2]],c='black',s=20,edgecolors='black',label=r'$T_1(E/cD)$')
ax[2].legend()
ax[2].label_outer()

plt.setp(ax,xlim=([-2.5,-0.7]),ylim=([-0.01,0.1]))
plt.setp(ax,xlabel=(r'$\log\,RFF$'))

plt.savefig('L07_stats_observation_desi/rffxeta_morf_types_t1.png')
plt.close()

#RFF - ETA/ GRUPO T2 COM AS MORFOLOGIAS cD e E

vec_lim=[elip_cut & bic_high_cut,cd_cut & bic_high_cut, (ecd_cut | cde_cut) & bic_high_cut]
print(len(rff_l07[vec_lim[1]]))
fig1,ax=plt.subplots(1,3,sharex=True,tight_layout=True,figsize=(15,5))
vec=[0.1,0.5,0.9]
for item in vec:
	ax[0].plot(np.log10(x),x-item*x,label=str(item))
	ax[1].plot(np.log10(x),x-item*x,label=str(item))
	ax[2].plot(np.log10(x),x-item*x,label=str(item))

ax[0].scatter(np.log10(rff_l07[vec_lim[0]]),eta_l07[vec_lim[0]],c='green',s=20,edgecolors='black',label=r'$T_2(E)$')
ax[0].legend()
ax[0].set_ylabel(r'$\eta$')

ax[1].scatter(np.log10(rff_l07[vec_lim[1]]),eta_l07[vec_lim[1]],c='red',s=20,edgecolors='black',label=r'$T_2(cD)$')
ax[1].legend()
ax[1].label_outer()

ax[2].scatter(np.log10(rff_l07[vec_lim[2]]),eta_l07[vec_lim[2]],c='black',s=20,edgecolors='black',label=r'$T_2(E/cD)$')
ax[2].legend()
ax[2].label_outer()

plt.setp(ax,xlim=([-2.5,-0.7]),ylim=([-0.01,0.1]))
plt.setp(ax,xlabel=(r'$\log\,RFF$'))

# plt.savefig('L07_stats_observation_desi/rffxeta_morf_types_t2.png')
plt.close()


# print('',pearsonr(np.log10(rff_obs),delta_ss))
# print('\n',pearsonr(eta_obs_s,delta_ss))
# print('###########')
# print('',pearsonr(np.log10(rff_obs[se_cut]),delta_se[se_cut]))
# print('\n',pearsonr(eta_obs_s[se_cut],delta_se[se_cut]))
# print('###########')
# print('',pearsonr(np.log10(rff_obs[ss_cut]),delta_ss[ss_cut]))
# print('\n',pearsonr(eta_obs_s[ss_cut],delta_ss[ss_cut]))
# print('###########')
'''

#CONFECÇÃO DAS MATRIZES CONFUSÃO/TABELAS DE REDSHIFT CONTRA AS CLASSES QUE ENCONTRAMOS
#VARIANDO OS CORTES POR REDSHIFT
r'''
# sample='YANG'
# # bins_vec=[3,4,5]
# bins_vec=[5,7,10]

# found_comp_vec=[[] for i in range(5)]
# table_vec=[]
# labels_tab=[]
# for w,pasta in enumerate(bins_vec):
# 	faixas=np.percentile(redshift_yang,np.linspace(0,100,pasta))
# 	idx_faixas=np.digitize(redshift_yang,faixas)
# 	bins_med=[]
# 	for z in range(len(faixas)-1):
# 		bins_med.append(np.round((faixas[z]+faixas[z+1])/2,3))
# 	labels_tab.append(np.asarray(bins_med,dtype=str))
# 	count_class=[]
# 	for q in range(pasta-1):
# 		g=3
# 		caminho=f'{sample}_stats_observation_desi/redshift_gmm/{pasta}_bins/{q+1}/n_{g}_comp_stats.dat'
# 		with open(f'{caminho}','r') as data_boots:
# 			lines=[linha.strip().split() for linha in data_boots]
# 			clean_list = [rodada for rodada in lines if 'nan' not in rodada]
# 			ncols=len(clean_list[0])
# 			clean_lines=np.asarray([np.asarray(linha,dtype=float) for linha in clean_list if len(linha) == ncols])
# 			clean_lines=clean_lines.T
# 		vec_sep=[' ' for i in range(g)]
# 		for sep in range(3,len(clean_lines)):
# 			x=clean_lines[sep]
# 			vec_sep[sep-3]=np.average(x)
# 		vec_lim_gmm=[vec_sep[0],vec_sep[1]]
# 		delta_bic_bin=delta_bic_yang[idx_faixas==q+1]

# 		s_cut=delta_bic_bin>vec_lim_gmm[0]
# 		half_cut=(delta_bic_bin<vec_lim_gmm[0]) & (delta_bic_bin>vec_lim_gmm[1])
# 		ss_cut=delta_bic_bin<vec_lim_gmm[1]

# 		bins_delta=np.arange(-7500,10000,200)#np.arange(min(delta_bic[h]),max(delta_bic[h]),200)
# 		fig=plt.figure()
# 		ax=fig.add_subplot()
# 		ax.set_title(f'{sample} em torno de z = {bins_med[q]}')
# 		ax.hist(delta_bic_bin[s_cut], bins=bins_delta, alpha=0.5, label='Grupo 1')
# 		ax.hist(delta_bic_bin[half_cut], bins=bins_delta, alpha=0.5, label='Grupo 2')
# 		ax.hist(delta_bic_bin[ss_cut], bins=bins_delta, alpha=0.5, label='Grupo 3')
# 		ax.axvline(vec_lim_gmm[0],c='black',linestyle='--',label=f'{vec_lim_gmm[0]:.3f}')
# 		ax.axvline(vec_lim_gmm[1],c='black',linestyle='-.',label=f'{vec_lim_gmm[1]:.3f}')
# 		ax.set_xlabel(r'$Delta BIC$')
# 		ax.set_xlim(-10000,5000)
# 		plt.legend()
# 		
# 		plt.savefig(f'{sample}_stats_observation_desi/redshift_gmm/graficos_yang/best_ncomp_{pasta-1}_{q}.png')
# 		plt.close(fig)


# 		bcgs_grupo=len(delta_bic_bin)
# 		s_grupo=len(delta_bic_bin[s_cut])/bcgs_grupo
# 		half_grupo=len(delta_bic_bin[half_cut])/bcgs_grupo
# 		ss_grupo=len(delta_bic_bin[ss_cut])/bcgs_grupo

# 		vec_tab=[100.*s_grupo,100.*half_grupo,100.*ss_grupo]
# 		vec_tab=np.asarray(vec_tab)
		
# 		count_class.append(vec_tab)
# 	table_vec.append(np.asarray(count_class))


# labels_y = ['S','S/S+S','S+S']
# # table_vec=[table_l07,table_whl,table_yang]

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle(sample)
# for ax, table, labels_x in zip(axs, table_vec, labels_tab):
# 	mapa= sns.heatmap(table,annot=True,cmap='Reds',xticklabels=labels_y,yticklabels=labels_x,cbar=False,ax=ax)
# 	ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  
# 	ax.set_ylabel(r'$z$',rotation=0)
# plt.tight_layout()
# 
# plt.savefig(f'{sample}_stats_observation_desi/redshift_gmm/graficos_yang/tabela_{sample}_confusion.png')


	# fig, ax = plt.subplots()
	# for g_idx, vet in enumerate(xx_temp_l07):
	# 	vet_arr = np.asarray(vet, dtype=float).T
	# 	vet_err = np.asarray(xx_err_l07[g_idx], dtype=float).T
	# 	vet_std = np.asarray(xx_std_l07[g_idx], dtype=float).T
	# 	vet_arr = vet_arr[:-1]
	# 	vet_err = vet_err[:-1]
	# 	vet_std = vet_std[:-1]
	# 	for id_err,item in enumerate(vet_arr):
	# 		err_item=vet_err[id_err]
	# 		std_item=vet_std[id_err]
	# 		line_std=ax.errorbar(bins_med_l07,item,yerr=std_item,ecolor='0.8',fmt='none')
	# 		line_err=ax.errorbar(bins_med_l07,item,yerr=err_item,markersize=5,fmt='o-')
	# 	ax.set_title(f"Bins {pasta-1} em z | {g_idx+2} grupos | {g_idx+1} corte(s)")
	# 	ax.xaxis.set_inverted(True)
	# 	ax.set_xlabel(r'$z$')
	# 	ax.set_ylabel('Ponto de corte')
	# 	ax.grid(True, alpha=0.3)
	# 	fig.legend([line_std],[r'$sigma$'],loc=('outside upper right'))
	# 	plt.show()
	# 	# plt.savefig(f'{sample}_stats_observation_desi/redshift_gmm/graficos_yang/{pasta-1}_bins_{g_idx+2}_comp.png')
	# 	plt.close()
'''
#comparação do l07 via faixas de redshift
r'''
sample='L07'
bins_vec=[5]
# bins_vec=[10]

found_comp_vec=[[] for i in range(5)]
for w,pasta in enumerate(bins_vec):
	faixas=np.percentile(redshift_l07,np.linspace(0,100,pasta))
	bins_med_l07=[]
	for z in range(len(faixas)-1):
		bins_med_l07.append(np.round((faixas[z]+faixas[z+1])/2,3))
	xx_temp_l07=[[] for _ in range(9)]
	xx_err_l07=[[]for _ in range(9)]
	xx_std_l07=[[]for _ in range(9)]
	for q in range(pasta-1):
		for g in range(2,11):
			caminho=f'{sample}_stats_observation_desi/redshift_gmm/{pasta}_bins/{q+1}/n_{g}_comp_stats.dat'
			with open(f'{caminho}','r') as data_boots:
				lines=[linha.strip().split() for linha in data_boots]
				clean_list = [rodada for rodada in lines if 'nan' not in rodada]
				ncols=len(clean_list[0])
				clean_lines=np.asarray([np.asarray(linha,dtype=float) for linha in clean_list if len(linha) == ncols])
				clean_lines=clean_lines.T
			vec_sep=[' ' for i in range(g)]
			vec_err=[' ' for j in range(g)]
			vec_std=[' ' for q in range(g)]
			for sep in range(3,len(clean_lines)):
				x=clean_lines[sep]
				# print(len(x))
				vec_sep[sep-3]=np.average(x)
				vec_err[sep-3]=np.divide(np.std(x),np.sqrt(len(x)))
				vec_std[sep-3]=np.std(x)
			xx_temp_l07[g-2].append(vec_sep)
			xx_err_l07[g-2].append(vec_err)
			xx_std_l07[g-2].append(vec_std)

	xx_temp_l07=[xx_temp_l07[1]]
	xx_err_l07=[xx_err_l07[1]]
	xx_std_l07=[xx_std_l07[1]]

	xx_temp_whl=[xx_temp_whl[1]]
	xx_err_whl=[xx_err_whl[1]]
	xx_std_whl=[xx_std_whl[1]]

	fig, ax = plt.subplots()
	for g_idx, vet in enumerate(xx_temp_l07):
		vet_arr = np.asarray(vet, dtype=float).T
		vet_err = np.asarray(xx_err_l07[g_idx], dtype=float).T
		vet_std = np.asarray(xx_std_l07[g_idx], dtype=float).T
		vet_arr = vet_arr[:-1]
		vet_err = vet_err[:-1]
		vet_std = vet_std[:-1]
		for id_err,item in enumerate(vet_arr):
			err_item=vet_err[id_err]
			std_item=vet_std[id_err]
			line_std=ax.errorbar(bins_med_l07,item,yerr=std_item,ecolor='0.8',fmt='none')
			line_err=ax.errorbar(bins_med_l07,item,yerr=err_item,markersize=5,fmt='o-')
	for g_idx, vet in enumerate(xx_temp_whl):
		vet_arr = np.asarray(vet, dtype=float).T
		vet_err = np.asarray(xx_err_whl[g_idx], dtype=float).T
		vet_std = np.asarray(xx_std_whl[g_idx], dtype=float).T
		vet_arr = vet_arr[:-1]
		vet_err = vet_err[:-1]
		vet_std = vet_std[:-1]
		for id_err,item in enumerate(vet_arr):
			err_item=vet_err[id_err]
			std_item=vet_std[id_err]
			line_std=ax.errorbar(bins_med_whl,item,yerr=std_item,ecolor='0.8',fmt='none')
			line_err=ax.errorbar(bins_med_whl,item,yerr=err_item,markersize=5,fmt='o-')

		ax.set_title(f"Bins {pasta-1} em z | {g_idx+2} grupos | {g_idx+1} corte(s)")
		ax.xaxis.set_inverted(True)
		ax.set_xlabel(r'$z$')
		ax.set_ylabel('Ponto de corte')
		ax.grid(True, alpha=0.3)
		fig.legend([line_std],[r'$sigma$'],loc=('outside upper right'))
		plt.show()
		# plt.savefig(f'{sample}_stats_observation_desi/redshift_gmm/graficos_yang/{pasta-1}_bins_{g_idx+2}_comp.png')
		plt.close()
#
'''

#PEDAÇO DA CONFECÇÃO DOS VALORES MÉDIOS DOS PONTOS DE CORTE
#QUE VIERAM DOS BOOTSTRAPS PARA AS AMOSTRAS
r'''

			# n_cluster_boots=np.loadtxt(caminho,dtype=float).T
			# found_cluster_boots=set(n_cluster_boots.astype(int))
			# print(n_cluster_boots)
		# with open(caminho) as f:
		# 	linhas_boots = [linha.strip().split() for linha in f]

		# data_cluster_boots=[[] for i in range(len(found_cluster_boots))]
		# for j in range(len(n_cluster_boots)):
		# 	for i,item in enumerate(found_cluster_boots):
		# 		if len(linhas_boots[j])-2 == item:
		# 			data_cluster_boots[i].append(linhas_boots[j])

		# data_cluster_boots=[np.asarray(data_cluster_boots[k],dtype=float) for k in range(len(data_cluster_boots))]
		# #BOOTSTRAP
		# vec_frac_ncl=[]
		# med_vec=[[] for i in range(6)]
		# for i,item in enumerate(found_cluster_boots):
		# 	if item < 6:
		# 		componentes=n_cluster_boots==item
		# 		ncl=len(n_cluster_boots[componentes])
		# 		frac_ncl=np.divide(ncl,len(n_cluster_boots))
		# 		vec_frac_ncl.append(frac_ncl)
		# 		# print(f'nlc {item} = {ncl}')
		# 		# print(f'fração de ncl {item}:{frac_ncl}')
		# 		if ncl == 0:
		# 			pass
		# 		else:
		# 			for j in range(item):
		# 				lim_comp=data_cluster_boots[i].T[j+2]
		# 				if item == 1:
		# 					med_lim,median_lim,std_lim=np.average(lim_comp),np.median(lim_comp),np.std(lim_comp)
		# 					med_vec[i].append(med_lim)
		# 					# print(f'Ponto médio {j+1} = {med_lim}')
		# 					# print(f'Mediana {j+1} = {median_lim}')
		# 					# print(f'Desvio Padrão {j+1} = {std_lim}')
		# 				else:
		# 					med_lim,median_lim,std_lim=np.average(lim_comp),np.median(lim_comp),np.std(lim_comp)
		# 					med_vec[i].append(med_lim)
		# 					# print(f'Ponto médio {j+1} = {med_lim}')
		# 					# print(f'Mediana {j+1} = {median_lim}')
		# 					# print(f'Desvio Padrão {j+1} = {std_lim}')

		# 	else:
		# 		componentes=n_cluster_boots>=item
		# 		ncl=len(n_cluster_boots[componentes])
		# 		frac_ncl=np.divide(ncl,len(n_cluster_boots))
		# 		# print(f'nlc {item} = {ncl}')
		# 		# print(f'fração de ncl {item}:{frac_ncl}')
		# 		# print('##########')
		# 		if ncl == 0:
		# 			pass
		# 		else:
		# 			for j in range(item):
		# 				lim_comp=data_cluster_boots[i].T[j+2]
		# 				med_lim,median_lim,std_lim=np.average(lim_comp),np.median(lim_comp),np.std(lim_comp)
		# 				# print(f'Ponto médio {j+1} = {med_lim}')
		# 				# print(f'Mediana {j+1} = {median_lim}')
		# 				# print(f'Desvio Padrão {j+1} = {std_lim}')
		# vec_frac_ncl=np.asarray(vec_frac_ncl)
		# id_best=np.argmax(vec_frac_ncl)
		# max_frac=vec_frac_ncl[id_best]
		# med_best=np.asarray(med_vec[id_best])
		# med_best=med_best[np.argsort(-med_best)]
		# print(vec_frac_ncl[id_best],med_best)

		# # s_cut=delta_bic_bin>med_best[0]
		# # half_cut=(delta_bic_bin<med_best[0]) & (delta_bic_bin>med_best[1])
		# # ss_cut=delta_bic_bin<med_best[1]

		# bins_delta=np.arange(-7500,1000,200)
		# fig=plt.figure()
		# ax=fig.add_subplot()

		# ax.boxplot()

		# # cores=['green','blue','orange','red','purple','pink','gray']
		# # lines=['--','-.',':','--','-.']
		# # ax.hist(delta_bic_bin, bins=bins_delta, color='white',edgecolor='black', alpha=0.5, label=fr'$z={bin_value}$')
		# # ax.hist([],color='white',label=f'fração de {len(med_best)} = {max_frac}')
		# # for h in range(len(med_best)):
		# # 	try:
		# # 		if h+1 != len(med_best):
		# # 			ax.axvline(med_best[h],c=cores[h],linestyle=lines[h],label=f'{med_best[h]}')
		# # 	except:
		# # 		pass
		# # ax.set_xlabel(r'$Delta BIC$')
		# # ax.set_xlim(-10000,5000)
		# # plt.legend()
		# 
		# # plt.savefig(f'{path}{savedir}/hist_try_{n}.png')
		# plt.close(fig)


		# print(20*'#####')
'''
#MATRIZ CONFUSÃO DAS CLASSES MORFOLOGICAS DO L07
r''' 
vec_lim_gmm_l07=[-727.1392870144092,-6408.608391715923]
s_cut=delta_bic_l07>vec_lim_gmm_l07[0]
half_cut=(delta_bic_l07<vec_lim_gmm_l07[0]) & (delta_bic_l07>vec_lim_gmm_l07[1])
ss_cut=delta_bic_l07<vec_lim_gmm_l07[1]

# elip_cut=tipo_morf=='E'
# cd_cut=tipo_morf=='cD'
# ecd_cut=tipo_morf=='E/cD'
# cde_cut=tipo_morf=='cD/E'

# sample=len(cluster)
# e_sample,cd_sample,inc_sample=len(cluster[elip_cut])+len(cluster[ecd_cut]),len(cluster[cd_cut])+len(cluster[cde_cut]),len(cluster[ecd_cut])+len(cluster[cde_cut])

# s_all=len(cluster[s_cut])
# s_elip=len(cluster[elip_cut & s_cut])
# s_cds=len(cluster[cd_cut & s_cut])
# s_inc=len(cluster[cde_cut & s_cut])+len(cluster[ecd_cut & s_cut])

# half_all=len(cluster[half_cut])
# half_eli=len(cluster[elip_cut & half_cut])
# hal
f_cd=len(cluster[cd_cut & half_cut])
# half_inc=len(cluster[cde_cut & half_cut])+len(cluster[ecd_cut & half_cut])

# ss_all=len(cluster[ss_cut])
# ss_elip=len(cluster[elip_cut & ss_cut])
# ss_cds=len(cluster[cd_cut & ss_cut])
# ss_inc=len(cluster[cde_cut & ss_cut])+len(cluster[ecd_cut & ss_cut])

# #########################################################################1
# #POR CENTAGENS
# #percentuais totais
# s_budget=100.*(s_all/sample)
# half_budget=100*(half_all/sample)
# ss_budget=100.*(ss_all/sample)
# #acertos/erro
# #S
# s_elip_budget=100.*(s_elip/e_sample)
# s_cd_budget=100.*(s_cds/cd_sample)
# s_misc_budget=100.*(s_inc/inc_sample)

# #SE
# half_elip_budget=100.*(half_eli/e_sample)
# half_cd_budget=100.*(half_cd/cd_sample)
# half_misc_budget=100.*(half_inc/inc_sample)

# #SS
# ss_cd_budget=100.*(ss_cds/cd_sample)
# ss_elip_budget=100.*(ss_elip/e_sample)
# ss_misc_budget=100.*(ss_inc/inc_sample)

# print(20*'#')
# print(f'\tall\tE\tcD\t E/cD & cD/E')
# print(f'S\t{s_all}\t{s_elip}\t{s_cds}\t{s_inc}')
# print(f'S+S\t{ss_all}\t{ss_elip}\t{ss_cds}\t{ss_inc}\n')
# print(f'S+S/S\t{half_all}\t{half_eli}\t{half_cd}\t{half_inc}\n')

# print('PERCENTUAL DE OBJETOS DA AMOSTRA RELATIVO AO MELHOR MODELO')
# print(f'S {s_budget:.3f}% \t S/S+S {half_budget:.3f}% \t S+S {ss_budget:.3f}%')

# vec=np.asarray([[ss_elip,ss_inc,ss_cds],[half_eli,half_inc,half_cd],[s_elip,s_inc,s_cds]])
# labels = ['cD','E/cD & cD/E','E']
# plt.figure(figsize=(6, 4))
# sns.heatmap(vec, annot=True, fmt='d', cmap='Reds',
#             xticklabels=np.flip(labels), yticklabels=labels)
# plt.savefig('L07_stats_observation_desi/confusion_matrix_morf.png')
'''
#CONSTRUÇÃO DA MATRIZ CONFUSÃO DAS AMOSTRAS QUANTO A SUAS FAIXAS DE REDSHIFT E OS CORTES 
# REFERENTES A TODA A AMOSTRA OBSERVADA PARA QUALQUER Z
r'''
#WHL
# data_whl=np.loadtxt(f'WHL_profit_observation.dat',dtype=str).T
# data_z_whl=np.loadtxt(f'WHL_clean_redshift.dat',dtype=str).T
# data_eta_whl=np.loadtxt(f'ass_WHL.dat').T

# bic_sersic_whl=data_whl[11].astype(float)
# bic_sersic_duplo_whl=data_whl[47].astype(float)
# delta_bic_whl=bic_sersic_duplo_whl - bic_sersic_whl

# redshift_whl=data_z_whl[1].astype(float)
# ##################################################################################
# #YANG
# data_yang=np.loadtxt(f'YANG_profit_observation.dat',dtype=str).T
# data_z_yang=np.loadtxt(f'YANG_clean_redshift.dat',dtype=str).T
# data_eta_yang=np.loadtxt(f'ass_YANG.dat').T

# bic_sersic_yang=data_yang[11].astype(float)
# bic_sersic_duplo_yang=data_yang[47].astype(float)
# delta_bic_yang=bic_sersic_duplo_yang - bic_sersic_yang

# redshift_yang=data_z_yang[1].astype(float)

# vec_lim_gmm_l07=[-727.1392870144092,-6408.608391715923]
# vec_lim_gmm_whl=[-141.3564488639911,-1173.166491957352]
# vec_lim_gmm_yang=[-290.48036438822186,-2463.7849736758526]

# s_cut_l07=delta_bic_l07>vec_lim_gmm_l07[0]
# half_cut_l07=(delta_bic_l07<vec_lim_gmm_l07[0]) & (delta_bic_l07>vec_lim_gmm_l07[1])
# ss_cut_l07=delta_bic_l07<vec_lim_gmm_l07[1]

# s_cut_whl=delta_bic_whl>vec_lim_gmm_whl[0]
# half_cut_whl=(delta_bic_whl<vec_lim_gmm_whl[0]) & (delta_bic_whl>vec_lim_gmm_whl[1])
# ss_cut_whl=delta_bic_whl<vec_lim_gmm_whl[1]

# s_cut_yang=delta_bic_yang>vec_lim_gmm_yang[0]
# half_cut_yang=(delta_bic_yang<vec_lim_gmm_yang[0]) & (delta_bic_yang>vec_lim_gmm_yang[1])
# ss_cut_yang=delta_bic_yang<vec_lim_gmm_yang[1]

# redshift_all=np.linspace(np.min(redshift_l07),np.max(redshift_yang),10000)

# faixas=np.percentile(redshift_all,np.linspace(0,100,10))

# idx_faixas_l07=np.digitize(redshift_l07,faixas)
# idx_faixas_whl=np.digitize(redshift_whl,faixas)
# idx_faixas_yang=np.digitize(redshift_yang,faixas)


# bins_med=[]
# for z in range(len(faixas)-1):
# 	bins_med.append(np.round((faixas[z]+faixas[z+1])/2,3))

# table_l07=[]
# table_whl=[]
# table_yang=[]
# print(bins_med)
# for i in range(1,10):

# 	obj_z_l07=idx_faixas_l07==i
# 	obj_z_whl=idx_faixas_whl==i
# 	obj_z_yang=idx_faixas_yang==i
# 	#L07
# 	try:
# 		bcgs_grupo_l07=len(delta_bic_l07[obj_z_l07])
# 		s_grupo_l07=len(delta_bic_l07[s_cut_l07 & obj_z_l07])/bcgs_grupo_l07
# 		half_grupo_l07=len(delta_bic_l07[half_cut_l07 & obj_z_l07])/bcgs_grupo_l07
# 		ss_grupo_l07=len(delta_bic_l07[ss_cut_l07 & obj_z_l07])/bcgs_grupo_l07
# 	except:
# 		s_grupo_l07=len(delta_bic_l07[s_cut_l07 & obj_z_l07])
# 		half_grupo_l07=len(delta_bic_l07[half_cut_l07 & obj_z_l07])
# 		ss_grupo_l07=len(delta_bic_l07[ss_cut_l07 & obj_z_l07])

# 	vec_l07=[100.*s_grupo_l07,100.*half_grupo_l07,100.*ss_grupo_l07]
# 	vec_l07=np.asarray(vec_l07)
# 	table_l07.append(vec_l07)

# 	#WHL
# 	try:
# 		bcgs_grupo_whl=len(delta_bic_whl[obj_z_whl])
# 		s_grupo_whl=len(delta_bic_whl[s_cut_whl & obj_z_whl])/bcgs_grupo_whl
# 		half_grupo_whl=len(delta_bic_whl[half_cut_whl & obj_z_whl])/bcgs_grupo_whl
# 		ss_grupo_whl=len(delta_bic_whl[ss_cut_whl & obj_z_whl])/bcgs_grupo_whl
# 	except:
# 		s_grupo_whl=len(delta_bic_whl[s_cut_whl & obj_z_whl])
# 		half_grupo_whl=len(delta_bic_whl[half_cut_whl & obj_z_whl])
# 		ss_grupo_whl=len(delta_bic_whl[ss_cut_whl & obj_z_whl])

# 	vec_whl=[100.*s_grupo_whl,100.*half_grupo_whl,100.*ss_grupo_whl]
# 	vec_whl=np.asarray(vec_whl)
# 	table_whl.append(vec_whl)

# 	#YANG
# 	try:
# 		bcgs_grupo_yang=len(delta_bic_yang[obj_z_yang])
# 		s_grupo_yang=len(delta_bic_yang[s_cut_yang & obj_z_yang])/bcgs_grupo_yang
# 		half_grupo_yang=len(delta_bic_yang[half_cut_yang & obj_z_yang])/bcgs_grupo_yang
# 		ss_grupo_yang=len(delta_bic_yang[ss_cut_yang & obj_z_yang])/bcgs_grupo_yang
# 	except:
# 		s_grupo_yang=len(delta_bic_yang[s_cut_yang & obj_z_yang])
# 		half_grupo_yang=len(delta_bic_yang[half_cut_yang & obj_z_yang])
# 		ss_grupo_yang=len(delta_bic_yang[ss_cut_yang & obj_z_yang])

# 	vec_yang=[100.*s_grupo_yang,100.*half_grupo_yang,100.*ss_grupo_yang]
# 	vec_yang=np.asarray(vec_yang)
# 	table_yang.append(vec_yang)

# table_l07=np.asarray(table_l07)
# table_whl=np.asarray(table_whl)
# table_yang=np.asarray(table_yang)

# labels_x = np.asarray(bins_med,dtype=str)
# print(labels_x)
# labels_y = ['S','S/S+S','S+S']

# titulos=['L07','WHL','YANG']

# table_vec=[table_l07,table_whl,table_yang]

# fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# for ax, table, title in zip(axs, table_vec, titulos):
# 	mapa= sns.heatmap(table,annot=True,cmap='Reds',xticklabels=labels_y,yticklabels=labels_x,cbar=False,ax=ax)
# 	ax.set_title(title)
# 	#ax.set_yticklabels(ax.get_yticks(), rotation=0)  
# 	ax.set_ylabel(r'$z$',rotation=0)
# plt.tight_layout()
# 
# plt.savefig('tabela_amostra_confusion.png')
'''
########################################################################
#CÓDIGO ORIGINAL COM A CONFECÇÃO DA FIGURA DE PONTOS DE CORTE EM DELTA BIC DOS MODELOS
r'''
# 	os.makedirs(f'WHL_stats_observation_desi/redshift_gmm/5_bins/{i}',exist_ok=True)
# 	for n in range(1000):
# 		print('bin',i,'BOOTSTRAP',n+1)
# 		random_delta_bic=np.random.choice(bic_vec,len(bic_vec))
# 		gmm_bootstrap('WHL',random_delta_bic,i,n,'5')

# faixas=np.percentile(redshift,np.linspace(0,100,7))
# idx_faixas=np.digitize(redshift,faixas)
# for i in range(1,7):
# 	os.makedirs(f'WHL_stats_observation_desi/redshift_gmm/7_bins/{i}',exist_ok=True)
# 	bic_vec=delta_bic[idx_faixas==i]
# 	for n in range(1000):
# 		print('bin',i,'BOOTSTRAP',n+1)
# 		random_delta_bic=np.random.choice(bic_vec,len(bic_vec))
# 		gmm_bootstrap('WHL',random_delta_bic,i,n,'7')

# faixas=np.percentile(redshift,np.linspace(0,100,10))
# idx_faixas=np.digitize(redshift,faixas)
# for i in range(1,10):
# 	os.makedirs(f'WHL_stats_observation_desi/redshift_gmm/10_bins/{i}',exist_ok=True)
# 	bic_vec=delta_bic[idx_faixas==i]
# 	for n in range(1000):
# 		print('bin',i,'BOOTSTRAP',n+1)
# 		random_delta_bic=np.random.choice(bic_vec,len(bic_vec))
# 		gmm_bootstrap('WHL',random_delta_bic,i,n,'10')

# # bins_delta=np.arange(-7500,1000,200)#np.arange(min(delta_bic[h]),max(delta_bic[h]),200)
# # fig=plt.figure()
# # ax=fig.add_subplot()
# # ax.hist(delta_bic[delta_bic>vec_lim_gmm[0]], bins=bins_delta, alpha=0.5, label='Grupo 1')
# # ax.hist(delta_bic[(delta_bic<vec_lim_gmm[0]) & (delta_bic>vec_lim_gmm[1])], bins=bins_delta, alpha=0.5, label='Grupo 2')
# # ax.hist(delta_bic[delta_bic<vec_lim_gmm[1]], bins=bins_delta, alpha=0.5, label='Grupo 3')
# # ax.axvline(vec_lim_gmm[0],c='black',linestyle='--',label=f'{vec_lim_gmm[0]:.3f}')
# # ax.axvline(vec_lim_gmm[1],c='black',linestyle='-.',label=f'{vec_lim_gmm[1]:.3f}')
# # ax.set_xlabel(r'$Delta BIC$')
# # ax.set_xlim(-10000,5000)
# # plt.legend()
# # 
# # plt.savefig(f'{sample}_stats_observation_desi/best_ncomp.png')
# # plt.close(fig)
'''

#ÁNALISE DOS DADOS DE AORDO COM OS PARAMETROS DE CADA MODELO OBTIDO PARA O MELHOR OBJETO,
#POSSO VIR A USAR NOS DADOS NOVOS EM QUESTÃO DE SEPARAÇÃO DOS GRUPOS VIA REDSHIFT
r'''
#analise_kde(delta_bic,delta_bic[region_vec[4]][rff_obs[region_vec[4]]<np.percentile(rff_obs,10)],delta_ss[region_vec[5]][rff_obs[region_vec[5]]<np.percentile(rff_obs,10)],r'$Delta{SS - S}$','delta_ss_s',[-8000,max(delta_ss)])

############
# #FLUX FRACS S+E
# analise_kde(flux_frac_se_1,flux_frac_se_1[region_vec[2]],flux_frac_se_1[region_vec[3]],r'${Flux_{S}}/{Flux_{S+E}}$','flux_frac_se_1',[0,1])
# analise_kde(flux_frac_se_2,flux_frac_se_2[region_vec[2]],flux_frac_se_2[region_vec[3]],r'${Flux_{E}}/{Flux_{S+E}}$','flux_frac_se_2',[0,1])

# #FLUX FRACS S+S
# analise_kde(flux_frac_ss_1,flux_frac_ss_1[region_vec[4]],flux_frac_ss_1[region_vec[5]],r'${Flux_{S1}}/{Flux_{S+S}}$','flux_frac_ss_1',[0,1])
# analise_kde(flux_frac_ss_2,flux_frac_ss_2[region_vec[4]],flux_frac_ss_2[region_vec[5]],r'${Flux_{S2}}/{Flux_{S+S}}$','flux_frac_ss_2',[0,1])

# #############################
# #DELTA'S BICS 
# #S+E
# analise_kde(delta_se,delta_se[region_vec[2]],delta_se[region_vec[3]],r'$Delta{SE - S}$','delta_se_s',[-6000,max(delta_se)])
# ##S+S
# analise_kde(delta_ss,delta_ss[region_vec[4]][rff_obs[region_vec[4]]<np.percentile(rff_obs,10)],delta_ss[region_vec[5]][rff_obs[region_vec[5]]<np.percentile(rff_obs,10)],r'$Delta{SS - S}$','delta_ss_s',[-8000,max(delta_ss)])
# #############################
# #RAZÃO DE RAIOS
# ##S+E
# analise_kde(re_ratio_se,re_ratio_se[region_vec[2]],re_ratio_se[region_vec[3]],r'$Re_b/Re_e$','kde_re_ratio_se',[0,3])
# ##S+S
# analise_kde(re_ratio_ss,re_ratio_ss[region_vec[4]],re_ratio_ss[region_vec[5]],r'$Re_1/Re_2$','kde_re_ratio_ss',[0,3])
# #####################################################
# #RAZÃO BT
# ##S+E
# analise_kde(bt_vec_se,bt_vec_se[region_vec[2]],bt_vec_se[region_vec[3]],r'$B/T (S+E)$','kde_bt_se',[0,1])
# ##S+S
# analise_kde(bt_vec_ss,bt_vec_ss[region_vec[4]],bt_vec_ss[region_vec[5]],r'$B/T (S+S)$','kde_bt_ss',[0,1])
# #RAZÃO DE ÍNDICES DE SÉRSIC 
# ##S+S
# analise_kde(n_ratio,n_ratio[region_vec[4]],n_ratio[region_vec[5]],r'$n_1/n_2 (S+S)$','kde_n_ratio_ss',[0,10])
# #ÍNDICE DE SÉRSIC INTERNO E EXTERNO (1 E 2)
# analise_kde(n1,n1[region_vec[4]],n1[region_vec[5]],r'$n_1 (S+S)$','kde_n_1_ss',[0,10])
# analise_kde(n2,n2[region_vec[4]],n2[region_vec[5]],r'$n_2 (S+S)$','kde_n_2_ss',[0,10])
# #RAZÃO AXIAL DOS COMPONENTES INTERNOS E EXTERNOS
# ##S+E
# analise_kde(e_s,e_s[region_vec[2]],e_s[region_vec[3]],r'$e_s (S+E)$','kde_e_s_se',[0,1])
# analise_kde(e_d,e_d[region_vec[4]],e_d[region_vec[5]],r'$e_e (S+E)$','kde_e_e_se',[0,1])
# ##S+S
# analise_kde(e1,e1[region_vec[4]],e1[region_vec[5]],r'$e_1 (S+S)$','kde_e_1_ss',[0,1])
# analise_kde(e2,e2[region_vec[4]],e2[region_vec[5]],r'$e_2 (S+S)$','kde_e_2_ss',[0,1])
# #RAZÃO DE RFF

# analise_kde(rff_ratio_sse[se_cut],rff_ratio_sse[region_vec[2]],rff_ratio_sse[region_vec[3]],r'$RFF_{S}/RFF_{S+E} (S+E)$','kde_rff_ratio_sse',[min(rff_ratio_sse),10])
# analise_kde(rff_ratio_sss[ss_cut],rff_ratio_sss[region_vec[4]],rff_ratio_sss[region_vec[5]],r'$RFF_{S}/RFF_{S+S} (S+S)$','kde_rff_ratio_sss',[min(rff_ratio_sss),10])
# print(cluster[ss_cut][rff_ratio_sss[ss_cut]>3],tipo_morf_new[ss_cut][rff_ratio_sss[ss_cut]>3])

# analise_kde(z_new,z_new[region_vec[2]],z_new[region_vec[3]],'redshift','redshift_se',[min(z_new),max(z_new)])
# analise_kde(z_new,z_new[region_vec[4]],z_new[region_vec[5]],'redshift','redshift_ss',[min(z_new),max(z_new)])
# # ####################################################################################
# #SCATTER PLOTS

# ###S+E

# ##RAZÃO DE RAIOS CONTRA RAZÃO DE RFF

# fig,axs_se=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_se[0].scatter(re_ratio_se[region_vec[3]],rff_ratio_sse[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[0].scatter(re_ratio_se[region_vec[2]],rff_ratio_sse[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[0].set_xlim(0,2.5)
# axs_se[0].set_ylim(0.8,2.5)
# axs_se[0].set_xlabel(r'$Re_b/Re_e$')
# axs_se[0].set_ylabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[0].legend()

# axs_se[1].scatter(re_ratio_se[region_vec[2]],rff_ratio_sse[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[1].set_xlim(0,2.5)
# axs_se[1].set_ylim(0.8,2.5)
# axs_se[1].set_xlabel(r'$Re_b/Re_e$')
# axs_se[1].legend()

# axs_se[2].scatter(re_ratio_se[region_vec[3]],rff_ratio_sse[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[2].set_xlim(0,2.5)
# axs_se[2].set_ylim(0.8,2.5)
# axs_se[2].set_xlabel(r'$Re_b/Re_e$')
# axs_se[2].legend()
# plt.savefig('observation/scatter_re_ratio_rff_ratio_se.png')
# plt.close()

# ##RAZÃO DE RAIOS CONTRA RAZÃO BT

# fig,axs_se=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_se[0].scatter(re_ratio_se[region_vec[3]],bt_vec_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[0].scatter(re_ratio_se[region_vec[2]],bt_vec_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[0].set_xlim(0,2.5)
# axs_se[0].set_ylim(0,1)
# axs_se[0].set_xlabel(r'$Re_b/Re_e$')
# axs_se[0].set_ylabel(r'$B/T (S+E)$')
# axs_se[0].legend()

# axs_se[1].scatter(re_ratio_se[region_vec[2]],bt_vec_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[1].set_xlim(0,2.5)
# axs_se[1].set_ylim(0,1)
# axs_se[1].set_xlabel(r'$Re_b/Re_e$')
# axs_se[1].legend()

# axs_se[2].scatter(re_ratio_se[region_vec[3]],bt_vec_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[2].set_xlim(0,2.5)
# axs_se[2].set_ylim(0,1)
# axs_se[2].set_xlabel(r'$Re_b/Re_e$')
# axs_se[2].legend()

# 
# plt.savefig('observation/scatter_re_ratio_bt_se.png')
# plt.close()

# ##RAZÃO DE RAIOS CONTRA DELTA BIC

# fig,axs_se=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_se[0].scatter(re_ratio_se[region_vec[3]],delta_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[0].scatter(re_ratio_se[region_vec[2]],delta_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[0].set_xlim(0,2.5)
# # axs_se[0].set_ylim(-6000,1)
# axs_se[0].set_xlabel(r'$Re_b/Re_e$')
# axs_se[0].set_ylabel(r'$delta BIC (S+E - S)$')
# axs_se[0].legend()

# axs_se[1].scatter(re_ratio_se[region_vec[2]],delta_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[1].set_xlim(0,2.5)
# # axs_se[1].set_ylim(-6000,1)
# axs_se[1].set_xlabel(r'$Re_b/Re_e$')
# axs_se[1].legend()

# axs_se[2].scatter(re_ratio_se[region_vec[3]],delta_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[2].set_xlim(0,2.5)
# # axs_se[2].set_ylim(-6000,1)
# axs_se[2].set_xlabel(r'$Re_b/Re_e$')
# axs_se[2].legend()

# plt.savefig('observation/scatter_re_ratio_delta_se.png')
# plt.close()


# ##RAZÃO DE RFF CONTRA DELTA BIC

# fig,axs_se=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_se[0].scatter(rff_ratio_sse[region_vec[3]],delta_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[0].scatter(rff_ratio_sse[region_vec[2]],delta_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[0].set_xlim(0.8,3)
# # axs_se[0].set_ylim(-6000,1)
# axs_se[0].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[0].set_ylabel(r'$delta BIC (S+E - S)$')
# axs_se[0].legend()

# axs_se[1].scatter(rff_ratio_sse[region_vec[2]],delta_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[1].set_xlim(0.8,3)
# # axs_se[1].set_ylim(-6000,1)
# axs_se[1].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[1].legend()

# axs_se[2].scatter(rff_ratio_sse[region_vec[3]],delta_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[2].set_xlim(0.8,3)
# # axs_se[2].set_ylim(-6000,1)
# axs_se[2].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[2].legend()

# 
# plt.savefig('observation/scatter_delta_rff_ratio_se.png')
# plt.close()


# ##RAZÃO DE RFF CONTRA BT

# fig,axs_se=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_se[0].scatter(rff_ratio_sse[region_vec[3]],bt_vec_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[0].scatter(rff_ratio_sse[region_vec[2]],bt_vec_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[0].set_xlim(0.8,3)
# axs_se[0].set_ylim(0,1)
# axs_se[0].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[0].set_ylabel(r'$B/T (S+E)$')
# axs_se[0].legend()

# axs_se[1].scatter(rff_ratio_sse[region_vec[2]],bt_vec_se[region_vec[2]],color='green',edgecolor='black',label='E')
# axs_se[1].set_xlim(0.8,3)
# axs_se[1].set_ylim(0,1)
# axs_se[1].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[1].legend()

# axs_se[2].scatter(rff_ratio_sse[region_vec[3]],bt_vec_se[region_vec[3]],color='orange',edgecolor='black',label='cD')
# axs_se[2].set_xlim(0.8,3)
# axs_se[2].set_ylim(0,1)
# axs_se[2].set_xlabel(r'$RFF_{S}/RFF_{S+E}$')
# axs_se[2].legend()

# 
# plt.savefig('observation/scatter_rff_ratio_bt_se.png')
# plt.close()


# ###S+S

# ##RAZÃO DE RAIOS CONTRA RAZÃO DE RFF

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(re_ratio_ss[region_vec[5]],rff_ratio_sss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(re_ratio_ss[region_vec[4]],rff_ratio_sss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(0,2.5)
# axs_ss[0].set_ylim(0.8,2.5)
# axs_ss[0].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[0].set_ylabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[0].legend()

# axs_ss[1].scatter(re_ratio_ss[region_vec[4]],rff_ratio_sss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(0,2.5)
# axs_ss[1].set_ylim(0.8,2.5)
# axs_ss[1].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[1].legend()

# axs_ss[2].scatter(re_ratio_ss[region_vec[5]],rff_ratio_sss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(0,2.5)
# axs_ss[2].set_ylim(0.8,2.5)
# axs_ss[2].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[2].legend()
# plt.savefig('observation/scatter_re_ratio_rff_ratio_ss.png')
# plt.close()

# ##RAZÃO DE RAIOS CONTRA RAZÃO BT

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(re_ratio_ss[region_vec[5]],bt_vec_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(re_ratio_ss[region_vec[4]],bt_vec_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(0,2.5)
# axs_ss[0].set_ylim(0,1)
# axs_ss[0].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[0].set_ylabel(r'$B/T (S+S)$')
# axs_ss[0].legend()

# axs_ss[1].scatter(re_ratio_ss[region_vec[4]],bt_vec_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(0,2.5)
# axs_ss[1].set_ylim(0,1)
# axs_ss[1].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[1].legend()

# axs_ss[2].scatter(re_ratio_ss[region_vec[5]],bt_vec_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(0,2.5)
# axs_ss[2].set_ylim(0,1)
# axs_ss[2].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[2].legend()

# 
# plt.savefig('observation/scatter_re_ratio_bt_ss.png')
# plt.close()

# ##RAZÃO DE RAIOS CONTRA DELTA BIC

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(re_ratio_ss[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(re_ratio_ss[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(0,2.5)
# # axs_ss[0].set_ylim(-6000,1)
# axs_ss[0].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[0].set_ylabel(r'$delta BIC (S+S - S)$')
# axs_ss[0].legend()

# axs_ss[1].scatter(re_ratio_ss[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(0,2.5)
# # axs_ss[1].set_ylim(-6000,1)
# axs_ss[1].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[1].legend()

# axs_ss[2].scatter(re_ratio_ss[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(0,2.5)
# # axs_ss[2].set_ylim(-6000,1)
# axs_ss[2].set_xlabel(r'$Re_1/Re_2$')
# axs_ss[2].legend()

# plt.savefig('observation/scatter_re_ratio_delta_ss.png')
# plt.close()


# ##RAZÃO DE RFF CONTRA DELTA BIC

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(rff_ratio_sss[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(rff_ratio_sss[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(0.8,3)
# axs_ss[0].set_ylim(-6000,100)
# axs_ss[0].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[0].set_ylabel(r'$delta BIC (S+S - S)$')
# axs_ss[0].legend()

# axs_ss[1].scatter(rff_ratio_sss[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(0.8,3)
# axs_ss[1].set_ylim(-6000,100)
# axs_ss[1].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[1].legend()

# axs_ss[2].scatter(rff_ratio_sss[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(0.8,3)
# axs_ss[2].set_ylim(-6000,100)
# axs_ss[2].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[2].legend()

# 
# plt.savefig('observation/scatter_delta_rff_ratio_ss.png')
# plt.close()


# ##RAZÃO DE RFF CONTRA BT

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(rff_ratio_sss[region_vec[5]],bt_vec_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(rff_ratio_sss[region_vec[4]],bt_vec_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(0.8,3)
# axs_ss[0].set_ylim(0,1)
# axs_ss[0].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[0].set_ylabel(r'$B/T (S+S)$')
# axs_ss[0].legend()

# axs_ss[1].scatter(rff_ratio_sss[region_vec[4]],bt_vec_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(0.8,3)
# axs_ss[1].set_ylim(0,1)
# axs_ss[1].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[1].legend()

# axs_ss[2].scatter(rff_ratio_sss[region_vec[5]],bt_vec_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(0.8,3)
# axs_ss[2].set_ylim(0,1)
# axs_ss[2].set_xlabel(r'$RFF_{S}/RFF_{S+S}$')
# axs_ss[2].legend()

# 
# plt.savefig('observation/scatter_rff_ratio_bt_ss.png')
# plt.close()

# ##RAZÃO DE n CONTRA DELTA

# fig,axs_ss=plt.subplots(1,3,figsize=(15,5),sharey=True)
# axs_ss[0].scatter(n_ratio[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[0].scatter(n_ratio[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[0].set_xlim(-0.1,10)
# # axs_ss[0].set_ylim(0,1)
# axs_ss[0].set_xlabel(r'$n_1/n_2$')
# axs_ss[0].set_ylabel(r'$delta (S+S - S)$')
# axs_ss[0].legend()

# axs_ss[1].scatter(n_ratio[region_vec[4]],delta_ss[region_vec[4]],color='green',edgecolor='black',label='E')
# axs_ss[1].set_xlim(-0.1,10)
# # axs_ss[1].set_ylim(0,1)
# axs_ss[1].set_xlabel(r'$n_1/n_2$')
# axs_ss[1].legend()

# axs_ss[2].scatter(n_ratio[region_vec[5]],delta_ss[region_vec[5]],color='orange',edgecolor='black',label='cD')
# axs_ss[2].set_xlim(-0.1,10)
# # axs_ss[2].set_ylim(0,1)
# axs_ss[2].set_xlabel(r'$n_1/n_2$')
# axs_ss[2].legend()

# #plt.show()
# #plt.savefig('observation/scatter_rff_ratio_bt_ss.png')
# plt.close()

# ##ASSIMETRIA CONTRA DELTA BIC

# 
# ##S/S+E
# ##S/S+S

# #print(len(delta_ss[region_vec[5]][delta_ss[region_vec[5]]<-500]))#,len(cluster[delta_ss[region_vec[5]]<-500]))
# # plt.figure()
# # plt.suptitle('Melhor modelo S+S - cD')
# # plt.hist(delta_ss[region_vec[5]],bins=bins1,histtype='step')
# # plt.xlabel(r'$Delta SS - S$')
# 
# # plt.close()
# #print(cluster[ss_cut bt_vec_ss[ss_cut] <0.1])
# #print(cluster[bt_vec_ss <0.1],bt_vec_ss[bt_vec_ss <0.1],cluster[region_vec[4]])

# # plt.figure()
# # # plt.scatter(delta_ss[region_vec[5]],bt_vec_ss[region_vec[5]],label='cD')
# # # plt.scatter(delta_ss[region_vec[4]],bt_vec_ss[region_vec[4]],label='E')
# # # plt.scatter(np.log10(re_ratio[region_vec[5]]),bt_vec_ss[region_vec[5]],label='cD')
# # # plt.scatter(np.log10(re_ratio[region_vec[4]]),bt_vec_ss[region_vec[4]],label='E')
# # # plt.scatter((n_ratio[(region_vec[5])]),re_ratio[(region_vec[5])],label='cD')
# # # plt.scatter((n_ratio[(region_vec[4])]),re_ratio[(region_vec[4])],label='E')

# # plt.legend()
# # #plt.xlim(-8000,0)
# # #plt.show()
# # plt.close()


# ##########################################################################1
# # #POR CENTAGENS
# # #percentuais totais
# # s_budget=100.*(s_all/sample)
# # se_budget=100.*(se_all/sample)
# # ss_budget=100.*(ss_all/sample)

# # #acertos/erro
# # #S
# # s_acerto=100.*(s_true/e_sample)
# # s_erro=100.*(s_false/cd_sample)
# # s_misc=100.*(s_inc/inc_sample)

# # #SE
# # se_acerto=100.*(se_true/cd_sample)
# # se_erro=100.*(se_false/e_sample)
# # se_misc=100.*(se_inc/inc_sample)

# # #SS
# # ss_acerto=100.*(ss_true/cd_sample)
# # ss_erro=100.*(ss_false/e_sample)
# # ss_misc=100.*(ss_inc/inc_sample)

# # print(20*'#')
# # print(f'\tall\tE\tcD\t E/cD & cD/E')#{f_values[idx_f]}')
# # print(f'S\t{s_all}\t{s_true}\t{s_false}\t{s_inc}')
# # print(f'S+E\t{se_all}\t{se_false}\t{se_true}\t{se_inc}')
# # print(f'S+S\t{ss_all}\t{ss_false}\t{ss_true}\t{ss_inc}\n')

# # print('PERCENTUAL DE OBJETOS DA AMOSTRA RELATIVO AO MELHOR MODELO')
# # print(f'S {s_budget:.3f}% \t S+E {se_budget:.3f}% \t S+S {ss_budget:.3f}%')

# # print('PERCENTUAL DE MELHOR MODELO S PARA E(E & E/cD)/cD(cD & cD/E)/OBJETOS INCERTOS (E/cD OU cD/E)')
# # print(f'Elipticas {s_acerto:.3f}% \t cD {s_erro:.3f}% \t E/cD {s_misc:.3f}%')

# # print('PERCENTUAL DE MELHOR MODELO S+E PARA E(E & E/cD)/cD(cD & cD/E)/OBJETOS INCERTOS (E/cD OU cD/E)')
# # print(f'Elipticas {se_erro:.3f}% \t cD {se_acerto:.3f}% \t E/cD {se_misc:.3f}%')

# # print('PERCENTUAL DE MELHOR MODELO S+S PARA E(E & E/cD)/cD(cD & cD/E)/OBJETOS INCERTOS (E/cD OU cD/E)')
# # print(f'Elipticas {ss_erro:.3f}% \t cD {ss_acerto:.3f}% \t E/cD {ss_misc:.3f}%')



# # fig1,axs=plt.subplots(2,2)
# # axs[0,0].scatter(np.log10(rff_obs[elip_cut & s_cut]),re_kpc[elip_cut & s_cut],c='green',marker='o',edgecolors='black')
# # axs[0,0].scatter(np.log10(rff_obs[elip_cut & se_cut]),re_kpc[elip_cut & se_cut],c='blue',marker='o',edgecolors='black')
# # axs[0,0].scatter(np.log10(rff_obs[elip_cut & ss_cut]),re_kpc[elip_cut & ss_cut],c='red',marker='o',edgecolors='black')
# # axs[0,0].set_xlim([-2.2,-0.5])
# # axs[0,0].set_ylim([0.2,3.5])
# # axs[0,0].set_xlabel(r'$log,RFF$')
# # axs[0,0].set_ylabel(r'$log,Re$')

# # axs[0,1].scatter(np.log10(rff_obs[cd_cut & s_cut]),re_kpc[cd_cut & s_cut],c='green',marker='s',edgecolors='black')
# # axs[0,1].scatter(np.log10(rff_obs[cd_cut & se_cut]),re_kpc[cd_cut & se_cut],c='blue',marker='s',edgecolors='black')
# # axs[0,1].scatter(np.log10(rff_obs[cd_cut & ss_cut]),re_kpc[cd_cut & ss_cut],c='red',marker='s',edgecolors='black')
# # axs[0,1].set_xlim([-2.2,-0.5])
# # axs[0,1].set_ylim([0.2,3.5])
# # axs[0,1].set_xlabel(r'$log,RFF$')
# # axs[0,1].set_ylabel(r'$log,Re$')

# # axs[1,0].scatter(np.log10(rff_obs[ecd_cut & s_cut]),re_kpc[ecd_cut & s_cut],c='green',marker='^',edgecolors='black')
# # axs[1,0].scatter(np.log10(rff_obs[ecd_cut & se_cut]),re_kpc[ecd_cut & se_cut],c='blue',marker='^',edgecolors='black')
# # axs[1,0].scatter(np.log10(rff_obs[ecd_cut & ss_cut]),re_kpc[ecd_cut & ss_cut],c='red',marker='^',edgecolors='black')
# # axs[1,0].set_xlim([-2.2,-0.5])
# # axs[1,0].set_ylim([0.2,3.5])
# # axs[1,0].set_xlabel(r'$log,RFF$')
# # axs[1,0].set_ylabel(r'$log,Re$')


# # axs[1,1].scatter(np.log10(rff_obs[elip_cut & s_cut]),re_kpc[elip_cut & s_cut],c='green',marker='o',edgecolors='black',label='S(E)')
# # axs[1,1].scatter(np.log10(rff_obs[elip_cut & se_cut]),re_kpc[elip_cut & se_cut],c='blue',marker='o',edgecolors='black',label='S+E(E)')
# # axs[1,1].scatter(np.log10(rff_obs[elip_cut & ss_cut]),re_kpc[elip_cut & ss_cut],c='red',marker='o',edgecolors='black',label='S+S(E)')
# # axs[1,1].scatter(np.log10(rff_obs[cd_cut & s_cut]),re_kpc[cd_cut & s_cut],c='green',marker='s',edgecolors='black',label='S(cD)')
# # axs[1,1].scatter(np.log10(rff_obs[cd_cut & se_cut]),re_kpc[cd_cut & se_cut],c='blue',marker='s',edgecolors='black',label='S+E(cD)')
# # axs[1,1].scatter(np.log10(rff_obs[cd_cut & ss_cut]),re_kpc[cd_cut & ss_cut],c='red',marker='s',edgecolors='black',label='S+S(cD)')
# # axs[1,1].scatter(np.log10(rff_obs[ecd_cut & s_cut]),re_kpc[ecd_cut & s_cut],c='green',marker='^',edgecolors='black',label='S(E/cD)')
# # axs[1,1].scatter(np.log10(rff_obs[ecd_cut & se_cut]),re_kpc[ecd_cut & se_cut],c='blue',marker='^',edgecolors='black',label='S+E(E/cD)')
# # axs[1,1].scatter(np.log10(rff_obs[ecd_cut & ss_cut]),re_kpc[ecd_cut & ss_cut],c='red',marker='^',edgecolors='black',label='S+S(E/cD)')
# # axs[1,1].set_xlim([-2.2,-0.5])
# # axs[1,1].set_ylim([0.2,3.5])
# # axs[1,1].set_xlabel(r'$log,RFF$')
# # axs[1,1].set_ylabel(r'$log,Re$')
# # fig1.legend(loc='outside upper right',fontsize='x-small')
# # plt.savefig(f'{save_file}/log_rffxre_kpc.png')
# # plt.close()
# # ###############################################

# # fig=plt.figure()
# # plt.scatter(np.log10(rff_obs[elip_cut & s_cut]),re_kpc[elip_cut & s_cut],c='green',marker='o',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[elip_cut & se_cut]),re_kpc[elip_cut & se_cut],c='blue',marker='o',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[elip_cut & ss_cut]),re_kpc[elip_cut & ss_cut],c='red',marker='o',edgecolors='black')
# # plt.xlim([-2.2,-0.5])
# # plt.ylim([0.2,3.5])
# # plt.xlabel(r'$log,RFF$')
# # plt.ylabel(r'$log,Re$')
# # fig.legend(loc='outside upper right',fontsize='x-small')
# # plt.savefig(f'{save_file}/log_rffxre_kpc_elip.png')
# # plt.close(fig)

# # fig=plt.figure()
# # plt.scatter(np.log10(rff_obs[cd_cut & s_cut]),re_kpc[cd_cut & s_cut],c='green',marker='s',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[cd_cut & se_cut]),re_kpc[cd_cut & se_cut],c='blue',marker='s',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[cd_cut & ss_cut]),re_kpc[cd_cut & ss_cut],c='red',marker='s',edgecolors='black')
# # plt.xlim([-2.2,-0.5])
# # plt.ylim([0.2,3.5])
# # plt.xlabel(r'$log,RFF$')
# # plt.ylabel(r'$log,Re$')
# # fig.legend(loc='outside upper right',fontsize='x-small')
# # fig.savefig(f'{save_file}/log_rffxre_kpc_cd.png')
# # plt.close(fig)

# # fig=plt.figure()
# # plt.scatter(np.log10(rff_obs[ecd_cut & s_cut]),re_kpc[ecd_cut & s_cut],c='green',marker='^',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[ecd_cut & se_cut]),re_kpc[ecd_cut & se_cut],c='blue',marker='^',edgecolors='black')
# # plt.scatter(np.log10(rff_obs[ecd_cut & ss_cut]),re_kpc[ecd_cut & ss_cut],c='red',marker='^',edgecolors='black')
# # plt.xlim([-2.2,-0.5])
# # plt.ylim([0.2,3.5])
# # plt.xlabel(r'$log,RFF$')
# # plt.ylabel(r'$log,Re$')
# # fig.legend(loc='outside upper right',fontsize='x-small')
# # plt.savefig(f'{save_file}/log_rffxre_kpc_ecd.png')
# # plt.close(fig)

# # fig=plt.figure()
# # plt.scatter(np.log10(rff_obs[elip_cut & s_cut]),re_kpc[elip_cut & s_cut],c='green',marker='o',edgecolors='black',label='S(E)')
# # plt.scatter(np.log10(rff_obs[elip_cut & se_cut]),re_kpc[elip_cut & se_cut],c='blue',marker='o',edgecolors='black',label='S+E(E)')
# # plt.scatter(np.log10(rff_obs[elip_cut & ss_cut]),re_kpc[elip_cut & ss_cut],c='red',marker='o',edgecolors='black',label='S+S(E)')
# # plt.scatter(np.log10(rff_obs[cd_cut & s_cut]),re_kpc[cd_cut & s_cut],c='green',marker='s',edgecolors='black',label='S(cD)')
# # plt.scatter(np.log10(rff_obs[cd_cut & se_cut]),re_kpc[cd_cut & se_cut],c='blue',marker='s',edgecolors='black',label='S+E(cD)')
# # plt.scatter(np.log10(rff_obs[cd_cut & ss_cut]),re_kpc[cd_cut & ss_cut],c='red',marker='s',edgecolors='black',label='S+S(cD)')
# # plt.scatter(np.log10(rff_obs[ecd_cut & s_cut]),re_kpc[ecd_cut & s_cut],c='green',marker='^',edgecolors='black',label='S(E/cD)')
# # plt.scatter(np.log10(rff_obs[ecd_cut & se_cut]),re_kpc[ecd_cut & se_cut],c='blue',marker='^',edgecolors='black',label='S+E(E/cD)')
# # plt.scatter(np.log10(rff_obs[ecd_cut & ss_cut]),re_kpc[ecd_cut & ss_cut],c='red',marker='^',edgecolors='black',label='S+S(E/cD)')
# # #plt.scatter(np.log10(rff_obs),zhao_optim(np.log10(rff_obs)),c='black')
# # plt.xlim([-2.2,-0.5])
# # plt.ylim([0.2,3.5])
# # plt.xlabel(r'$log,RFF$')
# # plt.ylabel(r'$log,Re$')
# # fig.legend(loc='outside upper right',fontsize='x-small')
# # plt.savefig(f'{save_file}/log_rffxre_kpc_all.png')
# # #plt.show()
# # plt.close(fig)
# # ##################
'''

r'''
bic_sersic_l07_se=data_l07_se[11].astype(float)
bic_sersic_duplo_l07_se=data_l07_se[47].astype(float)
delta_bic_l07_se=bic_sersic_duplo_l07_se - bic_sersic_l07_se
rff_l07_se=data_l07_se[49].astype(float)

bic_sersic_l07_se_v2=data_l07_se_v2[11].astype(float)
bic_sersic_duplo_l07_se_v2=data_l07_se_v2[47].astype(float)
delta_bic_l07_se_v2=bic_sersic_duplo_l07_se_v2 - bic_sersic_l07_se_v2
rff_l07_se_v2=data_l07_se_v2[49].astype(float)

bins_delta=np.arange(0.0,0.07,0.002)#np.arange(-7500,1000,200)#np.arange(min(delta_bic[h]),max(delta_bic[h]),200)
fig,ax=plt.subplots(1,3)
# ax=fig.add_subplot(2,1)
ax[0].hist(rff_l07, bins=bins_delta,color='green',label='chute galfit')
# ax[0].set_xlabel(r'$\Delta BIC$')
ax[0].set_xlabel(r'$RFF$')
ax[0].legend()
ax[1].hist(rff_l07_se, bins=bins_delta,color='red',label='chute sextractor')
# ax[1].set_xlabel(r'$\Delta BIC$')
ax[1].set_xlabel(r'$RFF$')
# ax.set_xlim(-10000,5000)
ax[1].legend()

ax[2].hist(rff_l07_se_v2, bins=bins_delta,color='blue',label='chute sextractor')
# ax[1].set_xlabel(r'$\Delta BIC$')
ax[2].set_xlabel(r'$RFF$')
# ax.set_xlim(-10000,5000)
ax[2].legend()

plt.show()
# plt.savefig(f'{sample}_stats_observation_desi/best_ncomp.png')
plt.close(fig)

plt.figure()
plt.scatter(delta_bic_l07,delta_bic_l07_se_v2)


plt.close()

plt.figure()
plt.scatter(delta_bic_l07,delta_bic_l07_se_v2)


plt.close()
lim=np.abs(np.abs(delta_bic_l07_se_v2) - np.abs(delta_bic_l07))
# print(cluster[lim > 5000])
# cluster_n1=cluster[cd_lim]
# ra_n1=ra[cd_lim]
# dec_n1=dec[cd_lim]
# redshift_n1=redshift[cd_lim]
# n1_cd=n1[cd_lim]
# data_n1=np.asarray([cluster_n1,ra_n1,dec_n1,redshift_n1,n1_cd]).T
# np.savetxt(f'data_indiv_cd_{sample}.dat',data_n1,fmt='%s',newline='\n')
'''
r'''
plt.figure()
plt.scatter(magabs_e,n_s[elip_lim & lim_casjobs],color='green',edgecolor='black',label='E')
plt.scatter(magabs_cd_big,n_s[lim_cd_big & lim_casjobs],color='red',edgecolor='black',label='True cD')
plt.scatter(magabs_cd_small,n_s[lim_cd_small & lim_casjobs],color='blue',edgecolor='black',label='E(EL)')
plt.xlim(max(magabs),min(magabs))
plt.ylim(min(n_s),max(n_s))
plt.xlabel(r'$Mag_\odot$')
plt.ylabel(r'$n$')
plt.legend()

plt.close()
'''
