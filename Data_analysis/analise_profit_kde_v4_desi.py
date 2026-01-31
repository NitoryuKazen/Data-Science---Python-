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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
#DEFINIÇÃO DOS PARÂMETROS/LIMITES/DISTRIBUIÇÕES DE INTERESSE E FATORES PARA OS PLOTS KDE
#LAPELAS

xlabel=r'$\log_{10} R_e (Kpc)$'
ylabel=r'$<\mu_e>$'

label_2c='2C'
label_cd='cD'
label_elip='E'
label_elip_el='E(EL)'
linha_cd_label='Linha cD'
linha_2c_label='Linha 2C'
linha_elip_label='Linha E'
linha_elip_el_label='Linha E(EL)'

label_bojo_2c='Comp 1 2C.'
label_env_2c='Comp 2 2C.'
label_bojo_cd='Comp 1 cD.'
label_env_cd='Comp 2 cD.'
label_bojo_el='Comp 1 E(EL)'
label_env_el='Comp 2 E(EL)'

linha_interna_2c_label='Linha comp 1 cD'
linha_externa_2c_label='Linha comp 2 cD'

linha_interna_cd_label='Linha comp 1 cD'
linha_externa_cd_label='Linha comp 2 cD'

linha_interna_el_label='Linha comp 1 E(EL)'
linha_externa_el_label='Linha comp 2 E(EL)'

names_simples=['2C','E(EL)','cD','E']
names_comps=['2C','E(EL)','cD','E']
names_morf=['cD','E/cD & cD/E','E']
cores=['black','blue','red','green']
line_width=[4,1,1,1]
alpha_vec=[0.4,1,1,1]

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
os.makedirs(f'{sample}stats_observation_desi/test_ks/indice_sersic',exist_ok=True)
#MODELO SIMPLES
plt.figure()
plt.title('Indice de Sérsic - Modelos Simples - p_value')
sns.heatmap(ks_n_sersic[0],xticklabels=names_simples,yticklabels=names_simples,annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Modelos Simples - D_value')
sns.heatmap(ks_n_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_simples_dvalue.png')
plt.close()

##
#MODELO COMPOSTO - COMPONENTE INTERNO
plt.figure()
plt.title('Indice de Sérsic - Componente interno - p_value')
sns.heatmap(ks_n_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Componente interno - D_value')
sns.heatmap(ks_n_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_interno_dvalue.png')
plt.close()

##
#MODELO COMPOSTO - COMPONENTE EXTERNO
plt.figure()
plt.title('Indice de Sérsic - Componente externo - p_value')
sns.heatmap(ks_n_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Indice de Sérsic - Componente externo - D_value')
sns.heatmap(ks_n_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Modelos Simples')
for i,dist in enumerate(vec_kde_n_simples):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_n_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_sersic[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_n_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_sersic[1]:.3e}\n' f'K-S(E,cD)={vec_ks_n_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_n_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Componente interno')
for i,dist in enumerate(vec_kde_n_intern):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_n_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_interno[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_n_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_intern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_n_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_intern_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_n_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_intern_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Indice de Sérsic - Componente externo')
for i,dist in enumerate(vec_kde_n_extern):
	axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_n_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$n$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_n_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_n_extern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_n_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_n_extern_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_n_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_n_extern_2c[2]:.3e}')
fig.text(0.75, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_kde.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_EL_E_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_n_simples_zhao):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_n_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_n_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_n_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_n_intern_cD):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_n_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_n_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_intern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_n_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Indice de Sérsic - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_n_extern_cD):
		axs.plot(n_linspace,dist(n_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_n_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_n_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$n$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_n_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_n_extern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_n_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_n_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_externo_EL_E_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/indice_sersic/n_simples_E_kde_cD_misc_zhao.png')
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
os.makedirs(f'{sample}stats_observation_desi/test_ks/ax_ratio',exist_ok=True)

##
plt.figure()
plt.title('Razão Axial - Modelos Simples - p_value')
sns.heatmap(ks_axrat_sersic[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Modelos Simples - D_value')
sns.heatmap(ks_axrat_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_simples_dvalue.png')
plt.close()

##
plt.figure()
plt.title('Razão Axial - Componente interno - p_value')
sns.heatmap(ks_axrat_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Componente interno - D_value')
sns.heatmap(ks_axrat_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_interno_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Razão Axial - Componente externo - p_value')
sns.heatmap(ks_axrat_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Razão Axial - Componente externo - D_value')
sns.heatmap(ks_axrat_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Razão Axial - Modelos Simples')
for i,dist in enumerate(vec_kde_axrat_simples):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_axrat_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_sersic[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_axrat_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_sersic[1]:.3e}\n' f'K-S(E,cD)={vec_ks_axrat_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_axrat_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Razão Axial - Componente interno')
for i,dist in enumerate(vec_kde_axrat_intern):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_axrat_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_interno[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_axrat_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_intern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_axrat_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_intern_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_axrat_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_intern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Razão Axial - Componente externo')
for i,dist in enumerate(vec_kde_axrat_extern):
	axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_axrat_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$q$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_axrat_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_axrat_extern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_axrat_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_axrat_extern_2c[0]:.3e}\n' f'K-S(2C,cD)={vec_ks_axrat_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_axrat_extern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_kde.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_EL_E_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_simples_zhao):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_axrat_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$q$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_axrat_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_axrat_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_intern_cD):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_axrat_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$q_1$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_axrat_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_intern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_axrat_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Razão Axial Sérsic - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_axrat_extern_cD):
		axs.plot(axrat_linspace,dist(axrat_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_axrat_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_axrat_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$q_2$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_axrat_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_axrat_extern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_axrat_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_axrat_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_externo_EL_E_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/ax_ratio/axrat_simples_E_kde_cD_misc_zhao.png')
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
os.makedirs(f'{sample}stats_observation_desi/test_ks/box',exist_ok=True)

##
plt.figure()
plt.title('Boxiness - Modelos Simples - p_value')
sns.heatmap(ks_box_sersic[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_simples_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Modelos Simples - D_value')
sns.heatmap(ks_box_sersic[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_simples_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Boxiness - Componente interno - p_value')
sns.heatmap(ks_box_intern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_interno_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Componente interno - D_value')
sns.heatmap(ks_box_intern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_interno_dvalue.png')
plt.close()
##
plt.figure()
plt.title('Boxiness - Componente externo - p_value')
sns.heatmap(ks_box_extern[0],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_pvalue.png')
plt.close()

plt.figure()
plt.title('Boxiness - Componente externo - D_value')
sns.heatmap(ks_box_extern[1],xticklabels=names_comps,yticklabels=names_comps, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Boxiness - Modelos Simples')
for i,dist in enumerate(vec_kde_box_simples):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_box_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_sersic[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_box_sersic[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_sersic[1]:.3e}\n' f'K-S(E,cD)={vec_ks_box_sersic[2]:.3e}')
info_labels_2c = (f'K-S(2C,cD) ={vec_ks_box_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_simples_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE INTERNO

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Boxiness - Componente interno')
for i,dist in enumerate(vec_kde_box_intern):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_box_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_interno[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_box_intern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_intern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_box_intern[2]:.3e}')
info_labels_2c = (f'K-S(2C,cD) ={vec_ks_box_intern_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_intern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_intern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_interno_kde.png')
plt.close()

#MODELO COMPOSTO - COMPONENTE EXTERNO
fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Boxiness - Componente externo')
for i,dist in enumerate(vec_kde_box_extern):
	axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
	axs.axvline(vec_med_box_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_externo[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$a_4/a$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_box_extern[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_box_extern[1]:.3e}\n' f'K-S(E,cD)={vec_ks_box_extern[2]:.3e}')
info_labels_2c = (f'K-S(2C,cD) ={vec_ks_box_extern_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_box_extern_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_box_extern_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_kde.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_EL_E_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_comp_zhao_pvalue.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness Sérsic - Modelos Simples - cD Zhao')
	for i,dist in enumerate(vec_kde_box_simples_zhao):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_box_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_sersic_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4/a$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_box_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_box_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_simples_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness - Componente interno - cD Zhao')
	for i,dist in enumerate(vec_kde_box_intern_cD):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_box_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_interno_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$(a_4/a)_1$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_box_intern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_intern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_box_intern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_intern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_interno_kde_cD_zhao.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Boxiness - Componente externo - cD Zhao')
	for i,dist in enumerate(vec_kde_box_extern_cD):
		axs.plot(box_linspace,dist(box_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_comps[i]}')
		axs.axvline(vec_med_box_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_box_externo_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$(a_4/a)_2$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_box_extern_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_box_extern_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_box_extern_cD_zhao[1]:.3e}\n' f'K-S(2C,E)={ks_box_extern_cD_zhao[0]:.3e}')
	fig.text(0.75, 0.94, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_externo_EL_E_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/box/box_simples_E_kde_cD_misc_zhao.png')
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
	os.makedirs(f'{sample}stats_observation_desi/test_ks/rff_ratio',exist_ok=True)
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/rff_ratio/rff_ratio_morf_cd.png')
	plt.close()

	fig,axs=plt.subplots(1,1)
	for i,dist in enumerate(vec_rff_ratio_elip_kde_zhao):
		axs.plot(rff_ratio_linspace,dist(rff_ratio_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_morf[i]}')
		axs.axvline(vec_med_rff_ratio_elip_zhao[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_rff_ratio_elip_zhao[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$RFF_{ss}/RFF_s$')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/rff_ratio/rff_ratio_elip_all_morf.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/a3',exist_ok=True)
##
plt.figure()
plt.title('a3 - p_value')
sns.heatmap(ks_a3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_pvalue.png')
plt.close()

plt.figure()
plt.title('a3 - D_value')
sns.heatmap(ks_a3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope a3 - p_value')
sns.heatmap(ks_slope_a3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_pvalue.png')
plt.close()

plt.figure()
plt.title('slope a3 - D_value')
sns.heatmap(ks_slope_a3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('a3 médio')
for i,dist in enumerate(vec_kde_a3):
	axs.plot(a3_linspace,dist(a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_a3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$a_3$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_a3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_a3[1]:.3e}\n' f'K-S(E,cD)={vec_ks_a3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_a3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_a3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_a3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope a3')
for i,dist in enumerate(vec_kde_slope_a3):
	axs.plot(slope_a3_linspace,dist(slope_a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_a3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha a_3$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_slope_a3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_a3[1]:.3e}\n' f'K-S(E,cD)={vec_ks_slope_a3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_a3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_a3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_a3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_simples_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a3 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_a3_med_cD):
		axs.plot(a3_linspace,dist(a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_a3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_med_a3_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_a3_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_med_a3_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_a3_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_med_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/a3_simples_kde_e_morf_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a3 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_a3_cD):
		axs.plot(slope_a3_linspace,dist(slope_a3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_a3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_3$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_slope_a3_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_a3_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_slope_a3_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_a3_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a3/slope_a3_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/a4',exist_ok=True)

##
plt.figure()
plt.title('a4 - p_value')
sns.heatmap(ks_a4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_pvalue.png')
plt.close()

plt.figure()
plt.title('a4 - D_value')
sns.heatmap(ks_a4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope a4 - p_value')
sns.heatmap(ks_slope_a4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_pvalue.png')
plt.close()

plt.figure()
plt.title('slope a4 - D_value')
sns.heatmap(ks_slope_a4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('a4 médio')
for i,dist in enumerate(vec_kde_a4):
	axs.plot(a4_linspace,dist(a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_a4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a4[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$a_4$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_a4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_a4[1]:.3e}\n' f'K-S(E,cD)={vec_ks_a4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_a4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_a4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_a4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope a4')
for i,dist in enumerate(vec_kde_slope_a4):
	axs.plot(slope_a4_linspace,dist(slope_a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_a4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a4[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha a_4$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_slope_a4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_a4[1]:.3e}\n' f'K-S(E,cD)={vec_ks_slope_a4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_a4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_a4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_a4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_simples_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('a4 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_a4_med_cD):
		axs.plot(a4_linspace,dist(a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_a4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_a4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_med_a4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_a4_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_med_a4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_a4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_med_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/a4_simples_kde_e_morf_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope a4 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_a4_cD):
		axs.plot(slope_a4_linspace,dist(slope_a4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_a4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_a4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$a_4$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_slope_a4_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_a4_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_slope_a4_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_a4_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/a4/slope_a4_kde_e_morf_zhao.png')
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


os.makedirs(f'{sample}stats_observation_desi/test_ks/b3',exist_ok=True)

##
plt.figure()
plt.title('b3 - p_value')
sns.heatmap(ks_b3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_pvalue.png')
plt.close()

plt.figure()
plt.title('b3 - D_value')
sns.heatmap(ks_b3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope b3 - p_value')
sns.heatmap(ks_slope_b3[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_pvalue.png')
plt.close()

plt.figure()
plt.title('slope b3 - D_value')
sns.heatmap(ks_slope_b3[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_dvalue.png')
plt.close()

#MODELO SIMPLES - MÉDIO

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('b3 médio')
for i,dist in enumerate(vec_kde_b3):
	axs.plot(b3_linspace,dist(b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_b3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$b_3$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_b3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_b3[1]:.3e}\n' f'K-S(E,cD)={vec_ks_b3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_b3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_b3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_b3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope b3')
for i,dist in enumerate(vec_kde_slope_b3):
	axs.plot(slope_b3_linspace,dist(slope_b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_b3[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b3[i]:.3e}$')
axs.legend()
axs.set_xlabel(r'$\alpha b_3$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_slope_b3[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_b3[1]:.3e}\n' f'K-S(E,cD)={vec_ks_slope_b3[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_b3_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_b3_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_b3_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_simples_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_med_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b3 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_b3_med_cD):
		axs.plot(b3_linspace,dist(b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_b3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_med_b3_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_b3_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_med_b3_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_b3_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_med_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/b3_simples_kde_e_morf_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b3 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_b3_cD):
		axs.plot(slope_b3_linspace,dist(slope_b3_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_b3_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b3_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_3$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_slope_b3_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_slope_b3_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_slope_b3_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_slope_b3_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b3/slope_b3_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/b4',exist_ok=True)
##
plt.figure()
plt.title('b4 - p_value')
sns.heatmap(ks_b4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_pvalue.png')
plt.close()

plt.figure()
plt.title('b4 - D_value')
sns.heatmap(ks_b4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_dvalue.png')
plt.close()
##
plt.figure()
plt.title('slope b4 - p_value')
sns.heatmap(ks_slope_b4[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_pvalue.png')
plt.close()

plt.figure()
plt.title('slope b4 - D_value')
sns.heatmap(ks_slope_b4[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_dvalue.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('b4 médio')
for i,dist in enumerate(vec_kde_b4):
	axs.plot(b4_linspace,dist(b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_b4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b4[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$b_4$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_b4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_b4[1]:.3e}\n' f'K-S(E,cD)={vec_ks_b4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_b4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_b4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_b4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_simples_kde.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope b4')
for i,dist in enumerate(vec_kde_slope_b4):
	axs.plot(slope_b4_linspace,dist(slope_b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_b4[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b4[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\alpha b_4$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_slope_b4[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_b4[1]:.3e}\n' f'K-S(E,cD)={vec_ks_slope_b4[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_b4_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_b4_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_b4_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_simples_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_med_kde_comp_zhao.png')
	plt.close()
	######
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('b4 médio - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_b4_med_cD):
		axs.plot(b4_linspace,dist(b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_b4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_b4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_med_b4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_med_b4_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_med_b4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_med_b4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_med_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/b4_simples_kde_e_morf_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope b4 - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_b4_cD):
		axs.plot(slope_b4_linspace,dist(slope_b4_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_b4_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_b4_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$b_4$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_slope_b4_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_slope_b4_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_slope_b4_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_slope_b4_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/b4/slope_b4_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/test_gr',exist_ok=True)

plt.figure()
plt.title('slope gr - Modelos Simples - p_value')
sns.heatmap(ks_slope_gr[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_pvalue.png')
plt.close()

plt.figure()
plt.title('slope gr - Modelos Simples - D_value')
sns.heatmap(ks_slope_gr[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_dvalue.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('slope g-r')
for i,dist in enumerate(vec_kde_slope_gr):
	axs.plot(slope_gr_linspace,dist(slope_gr_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_slope_gr[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_gr[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\alpha g-r$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_slope_gr[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_slope_gr[1]:.3e}\n' f'K-S(E,cD)={vec_ks_slope_gr[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_slope_gr_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_slope_gr_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_slope_gr_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_simples_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('slope gr - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_slope_gr_cD):
		axs.plot(slope_gr_linspace,dist(slope_gr_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_slope_gr_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_slope_gr_cD[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(r'$\alpha g-r$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_slope_gr_cD_zhao[2]:.3e}  ' f'K-S(E,E(EL))={ks_slope_gr_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_slope_gr_cD_zhao[1]:.3e} ' f'K-S(E,2C)={ks_slope_gr_cD_zhao[0]:.3e}')
	fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/test_gr/slope_gr_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/grad_e',exist_ok=True)

plt.figure()
plt.title('Gradiente de Elipticidade - p_value')
sns.heatmap(ks_grad_e[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_pvalue.png')
plt.close()

plt.figure()
plt.title('Gradiente de Elipticidade - D_value')
sns.heatmap(ks_grad_e[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_dvalue.png')
plt.close()

#MODELO SIMPLES - SLOPE

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Gradiente de Elipticidade')
for i,dist in enumerate(vec_kde_grad_e):
	axs.plot(grad_e_linspace,dist(grad_e_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_grad_e[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_e[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\nabla e$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_grad_e[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_grad_e[1]:.3e}\n' f'K-S(E,cD)={vec_ks_grad_e[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_grad_e_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_grad_e_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_grad_e_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Elipticidade - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_grad_e_cD):
		axs.plot(grad_e_linspace,dist(grad_e_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_grad_e_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_e_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla e$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_grad_e_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_grad_e_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_grad_e_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_grad_e_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_e/grad_e_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/grad_pa',exist_ok=True)

plt.figure()
plt.title('Gradiente de PA - p_value')
sns.heatmap(ks_grad_e[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_pvalue.png')
plt.close()

plt.figure()
plt.title('Gradiente de PA - D_value')
sns.heatmap(ks_grad_e[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_dvalue.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Gradiente de PA')
for i,dist in enumerate(vec_kde_grad_pa):
	axs.plot(grad_pa_linspace,dist(grad_pa_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_grad_pa[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_pa[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\nabla PA$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_grad_pa[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_grad_pa[1]:.3e}\n' f'K-S(E,cD)={vec_ks_grad_pa[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_grad_pa_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_grad_pa_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_grad_pa_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Gradiente Posição Angular - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_grad_pa_cD):
		axs.plot(grad_pa_linspace,dist(grad_pa_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_grad_pa_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_grad_pa_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\nabla PA$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_grad_pa_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_grad_pa_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_grad_pa_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_grad_pa_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/grad_pa/grad_pa_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/magabs',exist_ok=True)

plt.figure()
plt.title('Magnitude Absoluta - p_value')
sns.heatmap(ks_magabs[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_pvalue.png')
plt.close()

plt.figure()
plt.title('Magnitude Absoluta - D_value')
sns.heatmap(ks_magabs[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Magnitude Absoluta')
for i,dist in enumerate(vec_kde_magabs_simples[1:]):
	axs.plot(magabs_linspace,dist(magabs_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_magabs[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_magabs[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$Mag_\odot$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_magabs[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_magabs[1]:.3e}\n' f'K-S(E,cD)={vec_ks_magabs[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Magnitude Absoluta - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_magabs_cD):
		axs.plot(magabs_linspace,dist(magabs_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_magabs_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_magabs_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$Mag_\odot$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_magabs_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_magabs_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_magabs_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_magabs_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/magabs/magabs_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/starmass',exist_ok=True)

plt.figure()
plt.title('Massa estelar - p_value')
sns.heatmap(ks_starmass[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_pvalue.png')
plt.close()

plt.figure()
plt.title('Massa estelar - D_value')
sns.heatmap(ks_starmass[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Massa estelar')
for i,dist in enumerate(vec_kde_starmass[1:]):
	axs.plot(starmass_linspace,dist(starmass_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_starmass[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_starmass[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\log M_\odot$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_starmass[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_starmass[1]:.3e}\n' f'K-S(E,cD)={vec_ks_starmass[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Massa estelar - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_starmass_cD):
		axs.plot(starmass_linspace,dist(starmass_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_starmass_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_starmass_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_starmass_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_starmass_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_starmass_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass/starmass_kde_e_morf_zhao.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/age',exist_ok=True)

plt.figure()
plt.title('Idade Estelar - p_value')
sns.heatmap(ks_age[0],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds_r')
plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_pvalue.png')
plt.close()

plt.figure()
plt.title('Idade Estelar - D_value')
sns.heatmap(ks_age[1],xticklabels=names_simples,yticklabels=names_simples, annot=True, cmap='Reds')
plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_dvalue.png')
plt.close()

#MODELO SIMPLES

fig,axs=plt.subplots(1,1,figsize=(10,5))#,layout='constrained')
plt.title('Idade Estelar')
for i,dist in enumerate(vec_kde_age[1:]):
	axs.plot(age_linspace,dist(age_linspace),color=cores[1:][i],label=f'{names_simples[1:][i]}')
	axs.axvline(vec_med_age[1:][i],color=cores[1:][i],ls='--',label=fr'$\mu = {vec_med_age[1:][i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$\tau (Gyr)$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_age[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_age[1]:.3e}\n' f'K-S(E,cD)={vec_ks_age[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Idade estelar - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_age_cD):
		axs.plot(age_linspace,dist(age_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_age_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_age_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$\log M_{\bigstar}$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_age_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_age_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_age_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_age_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/age/age_kde_e_morf_zhao.png')
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
ax_center.scatter(m200_cd_big,starmass_cd_big,marker='o',edgecolor='black',label='cD',color='red')
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


plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_m200.png')
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
ax_center.scatter(starmass_cd_big,age_cd_big,marker='o',edgecolor='black',label='cD',color='red')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_age.png')
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

axs[1].scatter(m200_cd_big,mass_c1_cd_big,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big[0]:.3f}$\pm${np.sqrt(cov_c1_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big[1]:.3f}$\pm${np.sqrt(cov_c1_big[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c1.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_elip_elc2.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE EXTERNO')

axs[0].scatter(m200_cd_small,mass_c2_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small[0]:.3f}$\pm${np.sqrt(cov_c2_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small[1]:.3f}$\pm${np.sqrt(cov_c2_small[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c2_cd_big,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big[0]:.3f}$\pm${np.sqrt(cov_c2_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big[1]:.3f}$\pm${np.sqrt(cov_c2_big[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c2.png')
plt.close()

fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES)')

axs[0,0].scatter(m200_cd_small,mass_c1_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small[0]:.3f}$\pm${np.sqrt(cov_c1_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small[1]:.3f}$\pm${np.sqrt(cov_c1_small[1,1]):.3f}')
axs[0,0].set_ylim(9.2,12.2)
axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

axs[0,1].scatter(m200_cd_big,mass_c1_cd_big,marker='o',edgecolor='black',label='cD',color='red')
axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big[0]:.3f}$\pm${np.sqrt(cov_c1_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big[1]:.3f}$\pm${np.sqrt(cov_c1_big[1,1]):.3f}')
axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,0].scatter(m200_cd_small,mass_c2_cd_small,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small[0]:.3f}$\pm${np.sqrt(cov_c2_small[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small[1]:.3f}$\pm${np.sqrt(cov_c2_small[1,1]):.3f}')
axs[1,0].set_ylabel(r'$\log M_{\odot}$')
axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,1].scatter(m200_cd_big,mass_c2_cd_big,marker='o',edgecolor='black',label='cD',color='red')
axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big[0]:.3f}$\pm${np.sqrt(cov_c2_big[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big[1]:.3f}$\pm${np.sqrt(cov_c2_big[1,1]):.3f}')
axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps.png')
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

axs[1].scatter(m200_cd_big,mass_c1_cd_big_corr,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_corr[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c1_corr.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_elip_elc2_corr.png')
plt.close()

fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
plt.suptitle('COMPONENTE EXTERNO - corrigida')

axs[0].scatter(m200_cd_small,mass_c2_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0].set_ylim(9.2,12.2)
axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0].set_ylabel(r'$\log M_{\bigstar}$')
axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_corr[1,1]):.3f}')

axs[1].scatter(m200_cd_big,mass_c2_cd_big_corr,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_corr[1,1]):.3f}')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c2_corr.png')
plt.close()

fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - corrigida')

axs[0,0].scatter(m200_cd_small,mass_c1_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_corr[1,1]):.3f}')
axs[0,0].set_ylim(9.2,12.2)
axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

axs[0,1].scatter(m200_cd_big,mass_c1_cd_big_corr,marker='o',edgecolor='black',label='cD',color='red')
axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_corr[1,1]):.3f}')
axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,0].scatter(m200_cd_small,mass_c2_cd_small_corr,marker='o',edgecolor='black',alpha=0.6,label=label_elip_el,color='blue')
axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_corr[1,1]):.3f}')
axs[1,0].set_ylabel(r'$\log M_{\odot}$')
axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

axs[1,1].scatter(m200_cd_big,mass_c2_cd_big_corr,marker='o',edgecolor='black',label='cD',color='red')
axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_corr[1,1]):.3f}')
axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

fig.legend()
plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps_corr.png')
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

	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_m200_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_m200_dual_nossa.png')
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
	ax_center.scatter(m200_cd_big_cD,starmass_cd_big_cD,marker='o',edgecolor='black',label='cD[cD]',color='red')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_m200_cDs_zhao.png')
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

	plt.savefig(f'{sample}stats_observation_desi/test_ks/age_starmass_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_age_dual_nossa.png')
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
	ax_center.scatter(starmass_cd_big_cD,age_cd_big_cD,marker='o',edgecolor='black',label='cD[cD]',color='red')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/starmass_age_cDs_zhao.png')
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

	axs[1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD,marker='o',edgecolor='black',alpha=0.6,label='cD[cD]',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c1_cD.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE EXTERNO -- SEM CORREÇÃO-cD Zhao- nossa cD vs E(EL)')

	axs[0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c2_cD.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_elip_elc2_cD.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - cDs do ZHAO')

	axs[0,0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD,marker='o',edgecolor='black',label='cD[cD]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD,marker='o',edgecolor='black',label='cD[cD]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps_cD.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps_nos_zhao.png')
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

	axs[1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD_corr,marker='o',edgecolor='black',alpha=0.6,label='cD[cD]',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c1_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE EXTERNO -- CORRIGIDA-cD Zhao- nossa cD vs E(EL)')

	axs[0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar}$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[1,1]):.3f}')

	axs[1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD_corr,marker='o',edgecolor='black',alpha=0.6,label='cD',color='red')
	axs[1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_c2_cD_corr.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_elip_elc2_cD_corr.png')
	plt.close()

	fig,axs=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO (FIGURAS SUPERIORES) -- EXTERNO (FIGURAS INFERIORES) - cDs do ZHAO - CORRIGIDA')

	axs[0,0].scatter(m200_cd_small_cD,mass_c1_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[0,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_small_cD_corr[1,1]):.3f}')
	axs[0,0].set_ylim(9.2,12.2)
	axs[0,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')
	axs[0,0].set_ylabel(r'$\log M_{\bigstar}$')

	axs[0,1].scatter(m200_cd_big_cD,mass_c1_cd_big_cD_corr,marker='o',edgecolor='black',label='cD[cD]',color='red')
	axs[0,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c1_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c1_big_cD_corr[1,1]):.3f}')
	axs[0,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,0].scatter(m200_cd_small_cD,mass_c2_cd_small_cD_corr,marker='o',edgecolor='black',alpha=0.6,label=f'{label_elip_el}[cD]',color='blue')
	axs[1,0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_small_cD_corr),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_small_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_small_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_small_cD_corr[1,1]):.3f}')
	axs[1,0].set_ylabel(r'$\log M_{\odot}$')
	axs[1,0].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	axs[1,1].scatter(m200_cd_big_cD,mass_c2_cd_big_cD_corr,marker='o',edgecolor='black',label='cD[cD]',color='red')
	axs[1,1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_big_cD_corr),color='black',ls='-',label=fr'$\alpha$={ajust_c2_big_cD_corr[0]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_big_cD_corr[1]:.3f}$\pm${np.sqrt(cov_c2_big_cD_corr[1,1]):.3f}')
	axs[1,1].set_xlabel(r'$\log M_{200} (10^{14} M_\odot)$')

	fig.legend()
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps_cD_corr.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/m200_mass_comps_nos_zhao_corr.png')
	plt.close()

#
###########################################################
#LINHA DO H ALPHA
h_line_cd,h_line_cd_small,h_line_cd_big,h_line_e=h_line[cd_lim_halpha],h_line[cd_lim_halpha_small],h_line[cd_lim_halpha_big],h_line[elip_lim_halpha]
ks_h_line=ks_calc([h_line_cd,h_line_cd_small,h_line_cd_big,h_line_e])

vec_ks_h_line=ks_h_line[0][1][2],ks_h_line[0][1][3],ks_h_line[0][2][3]
vec_ks_h_line_2c=ks_h_line[0][0][1],ks_h_line[0][0][2],ks_h_line[0][0][3]

kde_h_line=kde(h_line)
h_line_factor = kde_h_line.factor
h_line_linspace=np.linspace(min(h_line),4.5,3000)
h_line_kde_cd,h_line_kde_cd_small,h_line_kde_cd_big,h_line_kde_e=kde(h_line_cd,bw_method=h_line_factor),kde(h_line_cd_small,bw_method=h_line_factor),kde(h_line_cd_big,bw_method=h_line_factor),kde(h_line_e,bw_method=h_line_factor)

vec_med_h_line=med_h_line_cd,med_h_line_cd_small,med_h_line_cd_big,med_h_line_e=np.average(h_line_cd),np.average(h_line_cd_small),np.average(h_line_cd_big),np.average(h_line_e)
vec_kde_h_line=h_line_kde_cd,h_line_kde_cd_small,h_line_kde_cd_big,h_line_kde_e

os.makedirs(f'{sample}stats_observation_desi/test_ks/h_line',exist_ok=True)

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Linha do h_alpha')
for i,dist in enumerate(vec_kde_h_line):
	axs.plot(h_line_linspace,dist(h_line_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_h_line[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_h_line[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$H_\alpha$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_h_line[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_h_line[1]:.3e}\n' f'K-S(E,cD)={vec_ks_h_line[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_h_line_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_h_line_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_h_line_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/h_line/hline_kde.png')
plt.close()

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

os.makedirs(f'{sample}stats_observation_desi/test_ks/bt',exist_ok=True)

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('RAZÃO BT -- SEM CORREÇÃO')
for i,dist in enumerate(vec_kde_bt):
	axs.plot(bt_linspace,dist(bt_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_bt[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_bt[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$B/T$')
plt.savefig(f'{sample}stats_observation_desi/test_ks/bt/bt_kde.png')
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
plt.savefig(f'{sample}stats_observation_desi/test_ks/bt/bt_corr_kde.png')
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

os.makedirs(f'{sample}stats_observation_desi/test_ks/conc',exist_ok=True)

fig,axs=plt.subplots(1,1,figsize=(10,5))
plt.title('Concentração')
for i,dist in enumerate(vec_kde_conc):
	axs.plot(conc_linspace,dist(conc_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
	axs.axvline(vec_med_conc[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_conc[i]:.3f}$')
axs.legend()
axs.set_xlabel(r'$R_{90}/R_{50}$')
info_labels = (f'K-S(cD,E(EL)) ={vec_ks_conc[0]:.3e}\n' f'K-S(E,E(EL))={vec_ks_conc[1]:.3e}\n' f'K-S(E,cD)={vec_ks_conc[2]:.3e}')
info_labels_2c = (f'K-S(2C,E(EL)) ={vec_ks_conc_2c[0]:.3e}\n' f'K-S(2C,E(EL))={vec_ks_conc_2c[1]:.3e}\n' f'K-S(2C,E)={vec_ks_conc_2c[2]:.3e}')
fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
fig.text(0.1, 0.95, info_labels_2c, fontsize=11, va='center', ha='left')
plt.savefig(f'{sample}stats_observation_desi/test_ks/conc/conc_kde.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/conc/conc_kde_comp_zhao.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title('Concentração - cDs do ZHAO')
	for i,dist in enumerate(vec_kde_conc_cD):
		axs.plot(conc_linspace,dist(conc_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_conc_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_conc_cD[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(r'$R_{90}/R_{50}$')
	info_labels = (f'K-S(cD,E(EL)) ={ks_conc_cD_zhao[2]:.3e}\n' f'K-S(E,E(EL))={ks_conc_cD_zhao[3]:.3e}\n' f'K-S(E,cD)={ks_conc_cD_zhao[1]:.3e}\n' f'K-S(E,2C)={ks_conc_cD_zhao[0]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{sample}stats_observation_desi/test_ks/conc/conc_kde_cD_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/test_ks/conc/conc_kde_e_morf_zhao.png')
	plt.close()


###################################################################################
###################################################################################
###################################################################################
#INVESTIGAÇÃO DO KORMENDY

x1ss,x2ss=(-0.28322921653227484,2.716867381942037)
y1ss,y2ss=(16.824997655983005,26.727550943781317)
#################################################
##TRADICIONAL - MUE MÉDIO EM RELAÇÃO AO MODELO DE DOIS COMPONENTES
#################################################

#PARÂMETROS DA RELAÇÃO DE KORMENDY
#GERAL
linspace_re= comp_12_linspace = np.linspace(min(re_s_kpc),max(re_s_kpc),100)

##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & 2C)
re_sersic_cd_kpc,re_sersic_kpc=re_s_kpc[cd_lim],re_s_kpc[elip_lim]
mue_sersic_cd,mue_sersic=mue_med_s[cd_lim],mue_med_s[elip_lim]

[alpha_cd,beta_cd],cov_cd = np.polyfit(re_sersic_cd_kpc,mue_sersic_cd,1,cov=True)
[alpha_s,beta_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

linha_cd = alpha_cd * comp_12_linspace + beta_cd
linha_sersic = alpha_s * comp_12_linspace + beta_s

alpha_cd_label,beta_cd_label = format(alpha_cd,'.3'),format(beta_cd,'.3') 
alpha_ser_label,beta_ser_label = format(alpha_s,'.3'),format(beta_s,'.3')
#COMPONENTES INTERNO-EXTERNO-SÉRSIC - CORTE POR DELTA BIC
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
#####################################################
lim_small_center=lim_cd_small[cd_lim]
lim_big_center=lim_cd_big[cd_lim]
print(len(re_sersic_cd_kpc[lim_cd_small[cd_lim]]),len(cd_lim),len(cd_lim[lim_cd_small]))

#LINHAS SERSIC - E(EL) & cD
re_sersic_cd_kpc_small,re_sersic_cd_kpc_big=re_sersic_cd_kpc[lim_small_center],re_sersic_cd_kpc[lim_big_center]
mue_sersic_cd_small,mue_sersic_cd_big=mue_sersic_cd[lim_small_center],mue_sersic_cd[lim_big_center]

[alpha_cd_small,beta_cd_small],cov_cd_small = np.polyfit(re_sersic_cd_kpc_small,mue_sersic_cd_small,1,cov=True)
[alpha_cd_big,beta_cd_big],cov_cd_big = np.polyfit(re_sersic_cd_kpc_big,mue_sersic_cd_big,1,cov=True)

linha_cd_small = alpha_cd_small * comp_12_linspace + beta_cd_small
linha_cd_big = alpha_cd_big * comp_12_linspace + beta_cd_big

alpha_cd_label_small,beta_cd_label_small = format(alpha_cd_small,'.3'),format(beta_cd_small,'.3') 
alpha_cd_label_big,beta_cd_label_big = format(alpha_cd_big,'.3'),format(beta_cd_big,'.3') 
#####

#LINHAS COMPONENTES INTERNA E EXTERNA - E(EL) & cD
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

#SERSIC SIMPLES -- E(EL),cD,2c
#RAIO EFETIVO
re_kde_cd_kpc_small,re_kde_cd_kpc_big,re_kde_cd_kpc=kde(re_sersic_cd_kpc_small,bw_method=re_factor),kde(re_sersic_cd_kpc_big,bw_method=re_factor),kde(re_sersic_cd_kpc,bw_method=re_factor)
#BRILHO EFETIVO MÉDIO
mue_kde_cd_kpc_small,mue_kde_cd_kpc_big,mue_kde_cd_kpc=kde(mue_sersic_cd_small,bw_method=mue_factor),kde(mue_sersic_cd_big,bw_method=mue_factor),kde(mue_sersic_cd,bw_method=mue_factor)

#GERAL(2C) -- COMPONENTES -- INTERNO,EXTERNO,SERSIC
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

######################################
#GRÁFICO JOINTPLOT DA RELAÇÃO DE KORMENDY
os.makedirs(f'{sample}stats_observation_desi/kormendy_rel',exist_ok=True)

##SÉRSIC -- 2C (SMALL E BIG) vs ELIPTICAS
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_2c,color='black')
ax_center.plot(linspace_re, linha_cd, color='black', linestyle='-', label=linha_2c_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc(re_linspace),color='black')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc),ls='--',color='black',label=fr'$\mu={format(np.average(re_sersic_cd_kpc),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='black')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='black',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample_sersic.png')
plt.close()

#SÉRSIC -- E vs E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_small,mue_sersic_cd_small,marker='s',edgecolor='black',alpha=0.3,label=label_elip_el,color='blue')
ax_center.plot(linspace_re, linha_cd_small, color='blue', linestyle='-', label=linha_elip_el_label+'\n'+fr'$\alpha={alpha_cd_label_small}$'+'\n'+fr'$\beta={beta_cd_label_small}$')
ax_center.legend(fontsize='small')
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
ax_righty.plot(mue_kde_cd_kpc_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)
plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample_sersic_small.png')
plt.close()

#SERSIC -- E vs cD
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_big,mue_sersic_cd_big,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_big, color='red', linestyle='-', label=linha_cd_label+' std\n'+fr'$\alpha={alpha_cd_label_big}$'+'\n'+fr'$\beta={beta_cd_label_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_sersic_cd_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample_sersic_big.png')
plt.close()

#######################################
#POR COMPONENTES EM RELAÇÃO A SÉRSIC
##GERAL - 2C(cD & E(EL))
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_2c,color='blue')
ax_center.plot(linspace_re, linha_interna, color='blue', linestyle='--', label=linha_interna_2c_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
ax_center.scatter(re_extern,mue_extern,marker='s',edgecolor='black',alpha=0.3,label=label_env_2c,color='red')
ax_center.plot(linspace_re, linha_externa, color='red', linestyle='-.', label=linha_externa_2c_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
ax_center.legend(fontsize='small')
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
plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample.png')
plt.close()

####
#SERSIC vs COMPONENTES E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
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

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_small.png')
plt.close()

####
#SERSIC vs COMPONENTES cD 
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
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

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_big.png')
plt.close()

##COMPONENTES CDS
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_big_only.png')
plt.close()

##COMPONENTES E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='red')
ax_topx.axvline(np.average(re_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_small_only.png')
plt.close()

##COMPONENTE - INTERNO & EXTERNO - CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='red')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comps_small_big.png')
plt.close()

##COMPONENTE INTERNO CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='red')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comp_intern_small_big.png')
plt.close()

##COMPONENTE EXTERNO CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comps_extern_small_big.png')
plt.close()

# PARTE DO L07
if sample == 'L07':
	label_cd_cd='cD[cD]'
	label_elip_cd='E[cD]'
	label_elip_el_cd='E(EL)[cD]'
	linha_cd_label_cd='Linha cD[cD]'
	linha_elip_label_cd='Linha E[cD]'
	linha_elip_el_label_cd='Linha E(EL)[cD]'

	label_bojo_cd_cd='Comp 1 cD[cD]'
	label_env_cd_cd='Comp 2 cD[cD]'
	label_bojo_el_cd='Comp 1 E(EL)[cD]'
	label_env_el_cd='Comp 2 E(EL)[cD]'

	linha_interna_2c_label_cd='Linha comp 1 cD[cD]'
	linha_externa_2c_label_cd='Linha comp 2 cD[cD]'

	linha_interna_cd_label_cd='Linha comp 1 cD[cD]'
	linha_externa_cd_label_cd='Linha comp 2 cD[cD]'

	linha_interna_el_label_cd='Linha comp 1 E(EL)[cD]'
	linha_externa_el_label_cd='Linha comp 2 E(EL)[cD]'

	##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & 2C)
	re_sersic_cd_kpc_small_cd,re_sersic_cd_kpc_big_cd,re_sersic_kpc_cd=re_s_kpc[lim_cd_small & cd_cut],re_s_kpc[lim_cd_big & cd_cut],re_s_kpc[elip_lim & cd_cut]
	mue_sersic_cd_small_cd,mue_sersic_cd_big_cd,mue_sersic_cd=mue_med_s[lim_cd_small & cd_cut],mue_med_s[lim_cd_big & cd_cut],mue_med_s[elip_lim & cd_cut]

	[alpha_cd_small_cd,beta_cd_small_cd],cov_cd_small_cd = np.polyfit(re_sersic_cd_kpc_small_cd,mue_sersic_cd_small_cd,1,cov=True)
	[alpha_cd_big_cd,beta_cd_big_cd],cov_cd_big_cd = np.polyfit(re_sersic_cd_kpc_big_cd,mue_sersic_cd_big_cd,1,cov=True)
	[alpha_s_cd,beta_s_cd],cov_s_cd = np.polyfit(re_sersic_kpc_cd,mue_sersic_cd,1,cov=True)

	linha_cd_small_cd = alpha_cd_small_cd * comp_12_linspace + beta_cd_small_cd
	linha_cd_big_cd = alpha_cd_big_cd * comp_12_linspace + beta_cd_big_cd
	linha_sersic_cd = alpha_s_cd * comp_12_linspace + beta_s_cd

	alpha_cd_label_small_cd,beta_cd_label_small_cd = format(alpha_cd_small_cd,'.3'),format(beta_cd_small_cd,'.3') 
	alpha_cd_label_big_cd,beta_cd_label_big_cd = format(alpha_cd_big_cd,'.3'),format(beta_cd_big_cd,'.3') 
	alpha_ser_label_cd,beta_ser_label_cd = format(alpha_s_cd,'.3'),format(beta_s_cd,'.3')

	#####################################################
	#LINHAS COMPONENTES INTERNA E EXTERNA - E(EL) - SUBGRUPO CDS DO ZHAO
	re_intern_small_cd,re_extern_small_cd=re_1_kpc[lim_cd_small & cd_cut],re_2_kpc[lim_cd_small & cd_cut]
	mue_intern_small_cd,mue_extern_small_cd=mue_med_1[lim_cd_small & cd_cut],mue_med_2[lim_cd_small & cd_cut]

	[alpha_med_1_small_cd,beta_med_1_small_cd],cov_1_small_cd = np.polyfit(re_intern_small_cd,mue_intern_small_cd,1,cov=True)
	[alpha_med_2_small_cd,beta_med_2_small_cd],cov_2_small_cd = np.polyfit(re_extern_small_cd,mue_extern_small_cd,1,cov=True)

	linha_intern_small_cd = alpha_med_1_small_cd * comp_12_linspace + beta_med_1_small_cd
	linha_extern_small_cd = alpha_med_2_small_cd * comp_12_linspace + beta_med_2_small_cd

	alpha_med_int_small_cd,beta_med_int_small_cd = format(alpha_med_1_small_cd,'.3'),format(beta_med_1_small_cd,'.3') 
	alpha_med_ext_small_cd,beta_med_ext_small_cd = format(alpha_med_2_small_cd,'.3'),format(beta_med_2_small_cd,'.3')
	##
	#LINHAS COMPONENTES INTERNA E EXTERNA cD - SUBGRUPO CDS DO ZHAO
	re_intern_big_cd,re_extern_big_cd=re_1_kpc[lim_cd_big & cd_cut],re_2_kpc[lim_cd_big & cd_cut]
	mue_intern_big_cd,mue_extern_big_cd=mue_med_1[lim_cd_big & cd_cut],mue_med_2[lim_cd_big & cd_cut]

	[alpha_med_1_big_cd,beta_med_1_big_cd],cov_1_big_cd = np.polyfit(re_intern_big_cd,mue_intern_big_cd,1,cov=True)
	[alpha_med_2_big_cd,beta_med_2_big_cd],cov_2_big_cd = np.polyfit(re_extern_big_cd,mue_extern_big_cd,1,cov=True)

	linha_intern_big_cd = alpha_med_1_big_cd * comp_12_linspace + beta_med_1_big_cd
	linha_extern_big_cd = alpha_med_2_big_cd * comp_12_linspace + beta_med_2_big_cd

	alpha_med_int_big_cd,beta_med_int_big_cd = format(alpha_med_1_big_cd,'.3'),format(beta_med_1_big_cd,'.3') 
	alpha_med_ext_big_cd,beta_med_ext_big_cd = format(alpha_med_2_big_cd,'.3'),format(beta_med_2_big_cd,'.3')

	#RAIO EFETIVO
	re_kde_cd_kpc_small_cd,re_kde_cd_kpc_big_cd,re_kde_sersic_kpc_cd=kde(re_sersic_cd_kpc_small_cd,bw_method=re_factor),kde(re_sersic_cd_kpc_big_cd,bw_method=re_factor),kde(re_sersic_kpc_cd,bw_method=re_factor)
	#BRILHO EFETIVO MÉDIO
	mue_kde_cd_kpc_small_cd,mue_kde_cd_kpc_big_cd,mue_kde_sersic_kpc_cd=kde(mue_sersic_cd_small_cd,bw_method=mue_factor),kde(mue_sersic_cd_big_cd,bw_method=mue_factor),kde(mue_sersic_cd,bw_method=mue_factor)
	##
	#ELIPTICAS (EXTRA-LIGHT) -- COMPONENTES -- INTERNO,EXTERNO
	re_kde_intern_small_cd,re_kde_extern_small_cd=kde(re_intern_small_cd,bw_method=re_factor),kde(re_extern_small_cd,bw_method=re_factor)
	mue_kde_intern_small_cd,mue_kde_extern_small_cd=kde(mue_intern_small_cd,bw_method=mue_factor),kde(mue_extern_small_cd,bw_method=mue_factor)
	##
	#cDs CONVENCIONAIS -- COMPONENTES -- INTERNO,EXTERNO
	re_kde_intern_big_cd,re_kde_extern_big_cd=kde(re_intern_big_cd,bw_method=re_factor),kde(re_extern_big_cd,bw_method=re_factor)
	mue_kde_intern_big_cd,mue_kde_extern_big_cd=kde(mue_intern_big_cd,bw_method=mue_factor),kde(mue_extern_big_cd,bw_method=mue_factor)

	#SÉRSIC -- E vs E(EL) - SUBGRUPO CDS DO ZHAO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_sersic_kpc_cd,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_cd,color='green')
	ax_center.plot(linspace_re, linha_sersic_cd, color='red', linestyle='-', label=linha_elip_label_cd+'\n'+fr'$\alpha={alpha_ser_label_cd}$'+'\n'+fr'$\beta={beta_ser_label_cd}$')
	ax_center.scatter(re_sersic_cd_kpc_small_cd,mue_sersic_cd_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_cd_small_cd, color='blue', linestyle='-', label=linha_elip_el_label_cd+'\n'+fr'$\alpha={alpha_cd_label_small_cd}$'+'\n'+fr'$\beta={beta_cd_label_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_sersic_kpc_cd(re_linspace),color='green')
	ax_topx.plot(re_linspace,re_kde_cd_kpc_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_sersic_kpc_cd),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc_cd),'.3')}$')
	ax_topx.axvline(np.average(re_sersic_cd_kpc_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_sersic_kpc_cd(mue_linspace),mue_linspace,color='green')
	ax_righty.plot(mue_kde_cd_kpc_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_sersic_cd_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample_sersic_small_cd_zhao.png')
	plt.close()

	#SERSIC -- E vs cD - SUBGRUPO CDS DO ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_sersic_kpc_cd,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_cd,color='green')
	ax_center.plot(linspace_re, linha_sersic_cd, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_ser_label_cd}$'+'\n'+fr'$\beta={beta_ser_label_cd}$')
	ax_center.scatter(re_sersic_cd_kpc_big_cd,mue_sersic_cd_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd_cd,color='blue')
	ax_center.plot(linspace_re, linha_cd_big_cd, color='red', linestyle='-', label=linha_cd_label_cd+'\n'+fr'$\alpha={alpha_cd_label_big_cd}$'+'\n'+fr'$\beta={beta_cd_label_big_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_sersic_kpc_cd(re_linspace),color='green')
	ax_topx.plot(re_linspace,re_kde_cd_kpc_big_cd(re_linspace),color='red')
	ax_topx.axvline(np.average(re_sersic_kpc_cd),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc_cd),'.3')}$')
	ax_topx.axvline(np.average(re_sersic_cd_kpc_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_sersic_kpc_cd(mue_linspace),mue_linspace,color='green')
	ax_righty.plot(mue_kde_cd_kpc_big_cd(mue_linspace),mue_linspace,color='red')
	ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_sersic_cd_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(mue_sersic_cd_big_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_sample_sersic_big_cds_zhao.png')
	plt.close()

	##COMPONENTE - INTERNO & EXTERNO - CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
	ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
	ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
	ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
	ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comps_small_big_cds_zhao.png')
	plt.close()
 
	##COMPONENTE INTERNO CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
	ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comp_intern_small_big_cds_zhao.png')
	plt.close()

	##COMPONENTE EXTERNO CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
	ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
	ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/rel_kormendy_comps_extern_small_big_cds_zhao.png')
	plt.close()
##RAZÕES ENTRE PARAMETROS
data_to_look=[bt_vec_corr,bt_vec_12,n_ratio_12,re_ratio_12,rff_ratio,chi2_ratio]
vmin_to_look=[0,0,1,1,0,0.9]
vmax_to_look=[1,1,1.5,1.5,0.9,1.5]
labels_to_look=[r'$B/T$',r'$B/T$',r'$n_1/n_2$',r'$Re_1/Re_2 (Kpc)$',r'$RFF_{S+S}/RFF_S$',r'$\chi^2_{S+S}/\chi^2_S$']
save_names=['bt_vec_corr','bt_vec_12','n_ratio_12','re_ratio_12','rff_ratio','chi2_ratio']
for i,item in enumerate(data_to_look):
	print(i,len(item[lim_cd_big]))
	fig = plt.figure(figsize=(16,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 4, width_ratios=[1,5,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center_left = fig.add_subplot(gs[1,1])
	ax_center_right = fig.add_subplot(gs[1,2],sharex=ax_center_left,sharey=ax_center_left)
	ax_topx_left = fig.add_subplot(gs[0,1],sharex=ax_center_left)
	ax_topx_right = fig.add_subplot(gs[0,2],sharex=ax_center_left)
	ax_righty = fig.add_subplot(gs[1,3],sharey=ax_center_right)

	ax_topx_left.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
	ax_topx_left.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
	ax_topx_left.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
	ax_topx_left.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
	ax_topx_left.legend(fontsize='small')
	ax_topx_left.tick_params(labelbottom=False)

	ax_topx_right.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
	ax_topx_right.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
	ax_topx_right.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
	ax_topx_right.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
	ax_topx_right.legend(fontsize='small')
	ax_topx_right.tick_params(labelbottom=False)
	ax_topx_right.yaxis.set_ticks_position('right')

	ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center_left.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center_left.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
	ax_center_left.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center_left.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
	ax_center_left.legend(fontsize='small')
	ax_center_left.set_ylim(y2ss,y1ss)
	ax_center_left.set_xlim(x1ss,x2ss)
	ax_center_left.set_xlabel(xlabel)
	ax_center_left.set_ylabel(ylabel)

	ax_center_right.scatter(re_extern_big,mue_extern_big,marker='o',edgecolor='black',alpha=0.9,label=label_env_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center_right.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
	ax_center_right.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.9,label=label_env_el,c=item[lim_cd_small],cmap=cmap)
	ax_center_right.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
	ax_center_right.tick_params(labelleft=False)
	ax_center_right.legend(fontsize='small')

	cax = inset_axes(ax_center_left,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center_left.transAxes,borderpad=0)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()
#PARTE DO L07
if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(16,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 4, width_ratios=[1,5,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center_left = fig.add_subplot(gs[1,1])
		ax_center_right = fig.add_subplot(gs[1,2],sharex=ax_center_left,sharey=ax_center_left)
		ax_topx_left = fig.add_subplot(gs[0,1],sharex=ax_center_left)
		ax_topx_right = fig.add_subplot(gs[0,2],sharex=ax_center_left)
		ax_righty = fig.add_subplot(gs[1,3],sharey=ax_center_right)

		ax_topx_left.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx_left.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
		ax_topx_left.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
		ax_topx_left.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
		ax_topx_left.legend(fontsize='small')
		ax_topx_left.tick_params(labelbottom=False)

		ax_topx_right.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx_right.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
		ax_topx_right.axvline(np.average(re_extern_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
		ax_topx_right.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
		ax_topx_right.legend(fontsize='small')
		ax_topx_right.tick_params(labelbottom=False)
		ax_topx_right.yaxis.set_ticks_position('right')

		ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_extern_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center_left.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center_left.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
		ax_center_left.scatter(re_intern_small_cd,mue_intern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center_left.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
		ax_center_left.legend(fontsize='small')
		ax_center_left.set_ylim(y2ss,y1ss)
		ax_center_left.set_xlim(x1ss,x2ss)
		ax_center_left.set_xlabel(xlabel)
		ax_center_left.set_ylabel(ylabel)

		ax_center_right.scatter(re_extern_big_cd,mue_extern_big_cd,marker='o',edgecolor='black',alpha=0.9,label=label_env_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center_right.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
		ax_center_right.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_env_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center_right.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
		ax_center_right.tick_params(labelleft=False)
		ax_center_right.legend(fontsize='small')

		cax = inset_axes(ax_center_left,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center_left.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()

#PARAMETROS REFERENTES AOS COMPONENTES
data_to_look=[box1,n1,np.log10(rff_s),np.log10(rff_ss)]
vmin_to_look=[0,0.5,-2.5,-2.5]
vmax_to_look=[0.6,8,-0.7,-0.7]
labels_to_look=[r'$a_4/a \ 1$',r'$n_1$',r'$RFF_S$',r'$RFF_SS$']
save_names=['box1','n1','rff_s','rff_ss']
for i,item in enumerate(data_to_look):
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
	ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()
####
if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
		ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
		ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)

		ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
		ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()

data_to_look=[box2,n2,rff_s,rff_ss]
vmin_to_look=[0,0.5,0,0]
vmax_to_look=[0.6,8,0.7,0.7]
labels_to_look=[r'$a_4/a \ 2 $',r'$n_2$',r'$RFF_S$',r'$RFF_SS$']
save_names=['box2','n2','rff_s','rff_ss']
for i,item in enumerate(data_to_look):
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_extern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center.scatter(re_extern_big,mue_extern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='--', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
	ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='--', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.15, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0.)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()

if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
		ax_topx.axvline(np.average(re_extern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
		ax_topx.axvline(np.average(re_extern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)

		ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_extern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_extern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='--', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
		ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='--', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()

#################################################
##TRADICIONAL - MUE MÉDIO EM RELAÇÃO AOS MODELOS DE CADA UM DOS COMPONENTES
#################################################
#PARÂMETROS DA RELAÇÃO DE KORMENDY
#GERAL
linspace_re= comp_12_linspace = np.linspace(min(re_s_kpc),max(re_s_kpc),100)

##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & 2C)
re_sersic_cd_kpc,re_sersic_kpc=re_s_kpc[cd_lim],re_s_kpc[elip_lim]
mue_sersic_cd,mue_sersic=mue_med_s[cd_lim],mue_med_s[elip_lim]

[alpha_cd,beta_cd],cov_cd = np.polyfit(re_sersic_cd_kpc,mue_sersic_cd,1,cov=True)
[alpha_s,beta_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

linha_cd = alpha_cd * comp_12_linspace + beta_cd
linha_sersic = alpha_s * comp_12_linspace + beta_s

alpha_cd_label,beta_cd_label = format(alpha_cd,'.3'),format(beta_cd,'.3') 
alpha_ser_label,beta_ser_label = format(alpha_s,'.3'),format(beta_s,'.3')

#COMPONENTES INTERNO-EXTERNO-SÉRSIC - CORTE POR DELTA BIC
re_intern,re_extern,re_sersic_kpc=re_1_kpc[cd_lim],re_2_kpc[cd_lim],re_s_kpc[elip_lim]
mue_intern,mue_extern,mue_sersic=mue_med_comp_1[cd_lim],mue_med_comp_2[cd_lim],mue_med_s[elip_lim]

[alpha_med_1,beta_med_1],cov_1 = np.polyfit(re_intern,mue_intern,1,cov=True)
[alpha_med_2,beta_med_2],cov_2 = np.polyfit(re_extern,mue_extern,1,cov=True)
[alpha_med_s,beta_med_s],cov_s = np.polyfit(re_sersic_kpc,mue_sersic,1,cov=True)

linha_interna = alpha_med_1 * comp_12_linspace + beta_med_1
linha_externa = alpha_med_2 * comp_12_linspace + beta_med_2
linha_sersic = alpha_med_s * comp_12_linspace + beta_med_s

alpha_med_int,beta_med_int = format(alpha_med_1,'.3'),format(beta_med_1,'.3') 
alpha_med_ext,beta_med_ext = format(alpha_med_2,'.3'),format(beta_med_2,'.3')
alpha_med_ser,beta_med_ser = format(alpha_med_s,'.3'),format(beta_med_s,'.3')
#####################################################

#LINHAS SERSIC - E(EL) & cD
re_sersic_cd_kpc_small,re_sersic_cd_kpc_big=re_sersic_cd_kpc[lim_small_center],re_sersic_cd_kpc[lim_big_center]
mue_sersic_cd_small,mue_sersic_cd_big=mue_sersic_cd[lim_small_center],mue_sersic_cd[lim_big_center]

[alpha_cd_small,beta_cd_small],cov_cd_small = np.polyfit(re_sersic_cd_kpc_small,mue_sersic_cd_small,1,cov=True)
[alpha_cd_big,beta_cd_big],cov_cd_big = np.polyfit(re_sersic_cd_kpc_big,mue_sersic_cd_big,1,cov=True)

linha_cd_small = alpha_cd_small * comp_12_linspace + beta_cd_small
linha_cd_big = alpha_cd_big * comp_12_linspace + beta_cd_big

alpha_cd_label_small,beta_cd_label_small = format(alpha_cd_small,'.3'),format(beta_cd_small,'.3') 
alpha_cd_label_big,beta_cd_label_big = format(alpha_cd_big,'.3'),format(beta_cd_big,'.3') 
#####

#LINHAS COMPONENTES INTERNA E EXTERNA - E(EL) & cD
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

#SERSIC SIMPLES -- E(EL),cD,2c
#RAIO EFETIVO
re_kde_cd_kpc_small,re_kde_cd_kpc_big,re_kde_cd_kpc=kde(re_sersic_cd_kpc_small,bw_method=re_factor),kde(re_sersic_cd_kpc_big,bw_method=re_factor),kde(re_sersic_cd_kpc,bw_method=re_factor)
#BRILHO EFETIVO MÉDIO
mue_kde_cd_kpc_small,mue_kde_cd_kpc_big,mue_kde_cd_kpc=kde(mue_sersic_cd_small,bw_method=mue_factor),kde(mue_sersic_cd_big,bw_method=mue_factor),kde(mue_sersic_cd,bw_method=mue_factor)

#GERAL(2C) -- COMPONENTES -- INTERNO,EXTERNO,SERSIC
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

######################################
#GRÁFICO JOINTPLOT DA RELAÇÃO DE KORMENDY
os.makedirs(f'{sample}stats_observation_desi/kormendy_rel_split',exist_ok=True)

##SÉRSIC -- 2C (SMALL E BIG) vs ELIPTICAS
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_2c,color='black')
ax_center.plot(linspace_re, linha_cd, color='black', linestyle='-', label=linha_2c_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc(re_linspace),color='black')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc),ls='--',color='black',label=fr'$\mu={format(np.average(re_sersic_cd_kpc),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='black')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='black',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample_sersic.png')
plt.close()

#SÉRSIC -- E vs E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_small,mue_sersic_cd_small,marker='s',edgecolor='black',alpha=0.3,label=label_elip_el,color='blue')
ax_center.plot(linspace_re, linha_cd_small, color='blue', linestyle='-', label=linha_elip_el_label+'\n'+fr'$\alpha={alpha_cd_label_small}$'+'\n'+fr'$\beta={beta_cd_label_small}$')
ax_center.legend(fontsize='small')
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
ax_righty.plot(mue_kde_cd_kpc_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)
plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample_sersic_small.png')
plt.close()

#SERSIC -- E vs cD
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_sersic_cd_kpc_big,mue_sersic_cd_big,marker='s',edgecolor='black',alpha=0.3,label=label_cd,color='blue')
ax_center.plot(linspace_re, linha_cd_big, color='red', linestyle='-', label=linha_cd_label+' std\n'+fr'$\alpha={alpha_cd_label_big}$'+'\n'+fr'$\beta={beta_cd_label_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
ax_topx.plot(re_linspace,re_kde_cd_kpc_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
ax_topx.axvline(np.average(re_sersic_cd_kpc_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
ax_righty.plot(mue_kde_cd_kpc_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
ax_righty.axhline(np.average(mue_sersic_cd_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_sersic_cd_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample_sersic_big.png')
plt.close()

#######################################
#POR COMPONENTES EM RELAÇÃO A SÉRSIC
##GERAL - 2C(cD & E(EL))
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern,mue_intern,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_2c,color='blue')
ax_center.plot(linspace_re, linha_interna, color='blue', linestyle='--', label=linha_interna_2c_label+'\n'+fr'$\alpha={alpha_med_int}$'+'\n'+fr'$\beta={beta_med_int}$')
ax_center.scatter(re_extern,mue_extern,marker='s',edgecolor='black',alpha=0.3,label=label_env_2c,color='red')
ax_center.plot(linspace_re, linha_externa, color='red', linestyle='-.', label=linha_externa_2c_label+'\n'+fr'$\alpha={alpha_med_ext}$'+'\n'+fr'$\beta={beta_med_ext}$')
ax_center.legend(fontsize='small')
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
plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample.png')
plt.close()

####
#SERSIC vs COMPONENTES E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
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

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_small.png')
plt.close()

####
#SERSIC vs COMPONENTES cD 
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_sersic_kpc,mue_sersic,marker='s',edgecolor='black',alpha=0.3,label=label_elip,color='green')
ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
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

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_big.png')
plt.close()

##COMPONENTES CDS
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='blue')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),color='red')
ax_topx.axvline(np.average(re_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_intern_big),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_big_only.png')
plt.close()

##COMPONENTES E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='red')
ax_center.plot(linspace_re, linha_extern_small, color='red', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='red')
ax_topx.axvline(np.average(re_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='red')
ax_righty.axhline(np.average(mue_intern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_small_only.png')
plt.close()

##COMPONENTE - INTERNO & EXTERNO - CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='red')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comps_small_big.png')
plt.close()

##COMPONENTE INTERNO CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd,color='red')
ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
ax_center.scatter(re_intern_small,mue_intern_small,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el,color='blue')
ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comp_intern_small_big.png')
plt.close()

##COMPONENTE EXTERNO CDS VS E(EL)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

ax_center = fig.add_subplot(gs[1,0])
ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

ax_center.scatter(re_extern_big,mue_extern_big,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd,color='red')
ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
ax_center.legend(fontsize='small')
ax_center.set_ylim(y2ss,y1ss)
ax_center.set_xlim(x1ss,x2ss)
ax_center.set_xlabel(xlabel)
ax_center.set_ylabel(ylabel)

ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
ax_topx.axvline(np.average(re_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
ax_topx.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
ax_topx.legend(fontsize='small')
ax_topx.tick_params(labelbottom=False)

ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
ax_righty.axhline(np.average(mue_extern_big),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
ax_righty.legend(fontsize='small')
ax_righty.tick_params(labelleft=False)

plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comps_extern_small_big.png')
plt.close()

# PARTE DO L07
if sample == 'L07':
	label_cd_cd='cD[cD]'
	label_elip_cd='E[cD]'
	label_elip_el_cd='E(EL)[cD]'
	linha_cd_label_cd='Linha cD[cD]'
	linha_elip_label_cd='Linha E[cD]'
	linha_elip_el_label_cd='Linha E(EL)[cD]'

	label_bojo_cd_cd='Comp 1 cD[cD]'
	label_env_cd_cd='Comp 2 cD[cD]'
	label_bojo_el_cd='Comp 1 E(EL)[cD]'
	label_env_el_cd='Comp 2 E(EL)[cD]'

	linha_interna_2c_label_cd='Linha comp 1 cD[cD]'
	linha_externa_2c_label_cd='Linha comp 2 cD[cD]'

	linha_interna_cd_label_cd='Linha comp 1 cD[cD]'
	linha_externa_cd_label_cd='Linha comp 2 cD[cD]'

	linha_interna_el_label_cd='Linha comp 1 E(EL)[cD]'
	linha_externa_el_label_cd='Linha comp 2 E(EL)[cD]'

	##SOMENTE SÉRSIC (SEPARADO POR CLASSE E & 2C)
	re_sersic_cd_kpc_small_cd,re_sersic_cd_kpc_big_cd,re_sersic_kpc_cd=re_s_kpc[lim_cd_small & cd_cut],re_s_kpc[lim_cd_big & cd_cut],re_s_kpc[elip_lim & cd_cut]
	mue_sersic_cd_small_cd,mue_sersic_cd_big_cd,mue_sersic_cd=mue_med_s[lim_cd_small & cd_cut],mue_med_s[lim_cd_big & cd_cut],mue_med_s[elip_lim & cd_cut]

	[alpha_cd_small_cd,beta_cd_small_cd],cov_cd_small_cd = np.polyfit(re_sersic_cd_kpc_small_cd,mue_sersic_cd_small_cd,1,cov=True)
	[alpha_cd_big_cd,beta_cd_big_cd],cov_cd_big_cd = np.polyfit(re_sersic_cd_kpc_big_cd,mue_sersic_cd_big_cd,1,cov=True)
	[alpha_s_cd,beta_s_cd],cov_s_cd = np.polyfit(re_sersic_kpc_cd,mue_sersic_cd,1,cov=True)

	linha_cd_small_cd = alpha_cd_small_cd * comp_12_linspace + beta_cd_small_cd
	linha_cd_big_cd = alpha_cd_big_cd * comp_12_linspace + beta_cd_big_cd
	linha_sersic_cd = alpha_s_cd * comp_12_linspace + beta_s_cd

	alpha_cd_label_small_cd,beta_cd_label_small_cd = format(alpha_cd_small_cd,'.3'),format(beta_cd_small_cd,'.3') 
	alpha_cd_label_big_cd,beta_cd_label_big_cd = format(alpha_cd_big_cd,'.3'),format(beta_cd_big_cd,'.3') 
	alpha_ser_label_cd,beta_ser_label_cd = format(alpha_s_cd,'.3'),format(beta_s_cd,'.3')

	#####################################################
	#LINHAS COMPONENTES INTERNA E EXTERNA - E(EL) - SUBGRUPO CDS DO ZHAO
	re_intern_small_cd,re_extern_small_cd=re_1_kpc[lim_cd_small & cd_cut],re_2_kpc[lim_cd_small & cd_cut]
	mue_intern_small_cd,mue_extern_small_cd=mue_med_comp_1[lim_cd_small & cd_cut],mue_med_comp_2[lim_cd_small & cd_cut]

	[alpha_med_1_small_cd,beta_med_1_small_cd],cov_1_small_cd = np.polyfit(re_intern_small_cd,mue_intern_small_cd,1,cov=True)
	[alpha_med_2_small_cd,beta_med_2_small_cd],cov_2_small_cd = np.polyfit(re_extern_small_cd,mue_extern_small_cd,1,cov=True)

	linha_intern_small_cd = alpha_med_1_small_cd * comp_12_linspace + beta_med_1_small_cd
	linha_extern_small_cd = alpha_med_2_small_cd * comp_12_linspace + beta_med_2_small_cd

	alpha_med_int_small_cd,beta_med_int_small_cd = format(alpha_med_1_small_cd,'.3'),format(beta_med_1_small_cd,'.3') 
	alpha_med_ext_small_cd,beta_med_ext_small_cd = format(alpha_med_2_small_cd,'.3'),format(beta_med_2_small_cd,'.3')
	##
	#LINHAS COMPONENTES INTERNA E EXTERNA cD - SUBGRUPO CDS DO ZHAO
	re_intern_big_cd,re_extern_big_cd=re_1_kpc[lim_cd_big & cd_cut],re_2_kpc[lim_cd_big & cd_cut]
	mue_intern_big_cd,mue_extern_big_cd=mue_med_comp_1[lim_cd_big & cd_cut],mue_med_comp_2[lim_cd_big & cd_cut]

	[alpha_med_1_big_cd,beta_med_1_big_cd],cov_1_big_cd = np.polyfit(re_intern_big_cd,mue_intern_big_cd,1,cov=True)
	[alpha_med_2_big_cd,beta_med_2_big_cd],cov_2_big_cd = np.polyfit(re_extern_big_cd,mue_extern_big_cd,1,cov=True)

	linha_intern_big_cd = alpha_med_1_big_cd * comp_12_linspace + beta_med_1_big_cd
	linha_extern_big_cd = alpha_med_2_big_cd * comp_12_linspace + beta_med_2_big_cd

	alpha_med_int_big_cd,beta_med_int_big_cd = format(alpha_med_1_big_cd,'.3'),format(beta_med_1_big_cd,'.3') 
	alpha_med_ext_big_cd,beta_med_ext_big_cd = format(alpha_med_2_big_cd,'.3'),format(beta_med_2_big_cd,'.3')

	#RAIO EFETIVO
	re_kde_cd_kpc_small_cd,re_kde_cd_kpc_big_cd,re_kde_sersic_kpc_cd=kde(re_sersic_cd_kpc_small_cd,bw_method=re_factor),kde(re_sersic_cd_kpc_big_cd,bw_method=re_factor),kde(re_sersic_kpc_cd,bw_method=re_factor)
	#BRILHO EFETIVO MÉDIO
	mue_kde_cd_kpc_small_cd,mue_kde_cd_kpc_big_cd,mue_kde_sersic_kpc_cd=kde(mue_sersic_cd_small_cd,bw_method=mue_factor),kde(mue_sersic_cd_big_cd,bw_method=mue_factor),kde(mue_sersic_cd,bw_method=mue_factor)
	##
	#ELIPTICAS (EXTRA-LIGHT) -- COMPONENTES -- INTERNO,EXTERNO
	re_kde_intern_small_cd,re_kde_extern_small_cd=kde(re_intern_small_cd,bw_method=re_factor),kde(re_extern_small_cd,bw_method=re_factor)
	mue_kde_intern_small_cd,mue_kde_extern_small_cd=kde(mue_intern_small_cd,bw_method=mue_factor),kde(mue_extern_small_cd,bw_method=mue_factor)
	##
	#cDs CONVENCIONAIS -- COMPONENTES -- INTERNO,EXTERNO
	re_kde_intern_big_cd,re_kde_extern_big_cd=kde(re_intern_big_cd,bw_method=re_factor),kde(re_extern_big_cd,bw_method=re_factor)
	mue_kde_intern_big_cd,mue_kde_extern_big_cd=kde(mue_intern_big_cd,bw_method=mue_factor),kde(mue_extern_big_cd,bw_method=mue_factor)

	#SÉRSIC -- E vs E(EL) - SUBGRUPO CDS DO ZHAO

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_sersic_kpc_cd,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_cd,color='green')
	ax_center.plot(linspace_re, linha_sersic_cd, color='red', linestyle='-', label=linha_elip_label_cd+'\n'+fr'$\alpha={alpha_ser_label_cd}$'+'\n'+fr'$\beta={beta_ser_label_cd}$')
	ax_center.scatter(re_sersic_cd_kpc_small_cd,mue_sersic_cd_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_cd_small_cd, color='blue', linestyle='-', label=linha_elip_el_label_cd+'\n'+fr'$\alpha={alpha_cd_label_small_cd}$'+'\n'+fr'$\beta={beta_cd_label_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_sersic_kpc_cd(re_linspace),color='green')
	ax_topx.plot(re_linspace,re_kde_cd_kpc_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_sersic_kpc_cd),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc_cd),'.3')}$')
	ax_topx.axvline(np.average(re_sersic_cd_kpc_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_sersic_kpc_cd(mue_linspace),mue_linspace,color='green')
	ax_righty.plot(mue_kde_cd_kpc_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_sersic_cd_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_sersic_cd_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample_sersic_small_cd_zhao.png')
	plt.close()

	#SERSIC -- E vs cD - SUBGRUPO CDS DO ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_sersic_kpc_cd,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_elip_cd,color='green')
	ax_center.plot(linspace_re, linha_sersic_cd, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_ser_label_cd}$'+'\n'+fr'$\beta={beta_ser_label_cd}$')
	ax_center.scatter(re_sersic_cd_kpc_big_cd,mue_sersic_cd_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_cd_cd,color='blue')
	ax_center.plot(linspace_re, linha_cd_big_cd, color='red', linestyle='-', label=linha_cd_label_cd+'\n'+fr'$\alpha={alpha_cd_label_big_cd}$'+'\n'+fr'$\beta={beta_cd_label_big_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_sersic_kpc_cd(re_linspace),color='green')
	ax_topx.plot(re_linspace,re_kde_cd_kpc_big_cd(re_linspace),color='red')
	ax_topx.axvline(np.average(re_sersic_kpc_cd),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc_cd),'.3')}$')
	ax_topx.axvline(np.average(re_sersic_cd_kpc_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(re_sersic_cd_kpc_big_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_sersic_kpc_cd(mue_linspace),mue_linspace,color='green')
	ax_righty.plot(mue_kde_cd_kpc_big_cd(mue_linspace),mue_linspace,color='red')
	ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_sersic_cd_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(mue_sersic_cd_big_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_sample_sersic_big_cds_zhao.png')
	plt.close()

	##COMPONENTE - INTERNO & EXTERNO - CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
	ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
	ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
	ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
	ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comps_small_big_cds_zhao.png')
	plt.close()
 
	##COMPONENTE INTERNO CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
	ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='o',edgecolor='black',alpha=0.3,label=label_bojo_el_cd,color='blue')
	ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comp_intern_small_big_cds_zhao.png')
	plt.close()

	##COMPONENTE EXTERNO CDS VS E(EL) -- SUBGRUPO cDS ZHAO
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)

	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_cd_cd,color='red')
	ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
	ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.3,label=label_env_el,color='blue')
	ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big_cd),ls='-.',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)

	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/rel_kormendy_comps_extern_small_big_cds_zhao.png')
	plt.close()
##RAZÕES ENTRE PARAMETROS
data_to_look=[bt_vec_corr,bt_vec_12,n_ratio_12,re_ratio_12,rff_ratio,chi2_ratio]
vmin_to_look=[0,0,1,1,0,0.9]
vmax_to_look=[1,1,1.5,1.5,0.9,1.5]
labels_to_look=[r'$B/T$',r'$B/T$',r'$n_1/n_2$',r'$Re_1/Re_2 (Kpc)$',r'$RFF_{S+S}/RFF_S$',r'$\chi^2_{S+S}/\chi^2_S$']
save_names=['bt_vec_corr','bt_vec_12','n_ratio_12','re_ratio_12','rff_ratio','chi2_ratio']
for i,item in enumerate(data_to_look):
	fig = plt.figure(figsize=(16,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 4, width_ratios=[1,5,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center_left = fig.add_subplot(gs[1,1])
	ax_center_right = fig.add_subplot(gs[1,2],sharex=ax_center_left,sharey=ax_center_left)
	ax_topx_left = fig.add_subplot(gs[0,1],sharex=ax_center_left)
	ax_topx_right = fig.add_subplot(gs[0,2],sharex=ax_center_left)
	ax_righty = fig.add_subplot(gs[1,3],sharey=ax_center_right)

	ax_topx_left.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
	ax_topx_left.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
	ax_topx_left.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
	ax_topx_left.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
	ax_topx_left.legend(fontsize='small')
	ax_topx_left.tick_params(labelbottom=False)

	ax_topx_right.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
	ax_topx_right.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
	ax_topx_right.axvline(np.average(re_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
	ax_topx_right.axvline(np.average(re_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
	ax_topx_right.legend(fontsize='small')
	ax_topx_right.tick_params(labelbottom=False)
	ax_topx_right.yaxis.set_ticks_position('right')

	ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center_left.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center_left.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
	ax_center_left.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center_left.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
	ax_center_left.legend(fontsize='small')
	ax_center_left.set_ylim(y2ss,y1ss)
	ax_center_left.set_xlim(x1ss,x2ss)
	ax_center_left.set_xlabel(xlabel)
	ax_center_left.set_ylabel(ylabel)

	ax_center_right.scatter(re_extern_big,mue_extern_big,marker='o',edgecolor='black',alpha=0.9,label=label_env_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center_right.plot(linspace_re, linha_extern_big, color='red', linestyle='-.', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
	ax_center_right.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.9,label=label_env_el,c=item[lim_cd_small],cmap=cmap)
	ax_center_right.plot(linspace_re, linha_extern_small, color='blue', linestyle='-.', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
	ax_center_right.tick_params(labelleft=False)
	ax_center_right.legend(fontsize='small')

	cax = inset_axes(ax_center_left,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center_left.transAxes,borderpad=0)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()
#PARTE DO L07
if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(16,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 4, width_ratios=[1,5,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center_left = fig.add_subplot(gs[1,1])
		ax_center_right = fig.add_subplot(gs[1,2],sharex=ax_center_left,sharey=ax_center_left)
		ax_topx_left = fig.add_subplot(gs[0,1],sharex=ax_center_left)
		ax_topx_right = fig.add_subplot(gs[0,2],sharex=ax_center_left)
		ax_righty = fig.add_subplot(gs[1,3],sharey=ax_center_right)

		ax_topx_left.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx_left.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
		ax_topx_left.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
		ax_topx_left.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
		ax_topx_left.legend(fontsize='small')
		ax_topx_left.tick_params(labelbottom=False)

		ax_topx_right.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx_right.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
		ax_topx_right.axvline(np.average(re_extern_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
		ax_topx_right.axvline(np.average(re_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
		ax_topx_right.legend(fontsize='small')
		ax_topx_right.tick_params(labelbottom=False)
		ax_topx_right.yaxis.set_ticks_position('right')

		ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_extern_big_cd),ls='--',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_extern_small_cd),ls='--',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center_left.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center_left.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
		ax_center_left.scatter(re_intern_small_cd,mue_intern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center_left.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
		ax_center_left.legend(fontsize='small')
		ax_center_left.set_ylim(y2ss,y1ss)
		ax_center_left.set_xlim(x1ss,x2ss)
		ax_center_left.set_xlabel(xlabel)
		ax_center_left.set_ylabel(ylabel)

		ax_center_right.scatter(re_extern_big_cd,mue_extern_big_cd,marker='o',edgecolor='black',alpha=0.9,label=label_env_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center_right.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='-.', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
		ax_center_right.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_env_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center_right.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='-.', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
		ax_center_right.tick_params(labelleft=False)
		ax_center_right.legend(fontsize='small')

		cax = inset_axes(ax_center_left,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center_left.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()

#PARAMETROS REFERENTES AOS COMPONENTES
data_to_look=[box1,n1,np.log10(rff_s),np.log10(rff_ss)]
vmin_to_look=[0,0.5,-2.5,-2.5]
vmax_to_look=[0.6,8,-0.7,-0.7]
labels_to_look=[r'$a_4/a \ 1$',r'$n_1$',r'$RFF_S$',r'$RFF_SS$']
save_names=['box1','n1','rff_s','rff_ss']
for i,item in enumerate(data_to_look):
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_topx.plot(re_linspace,re_kde_intern_big(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_intern_small(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big),'.3')}$')
	ax_topx.axvline(np.average(re_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_intern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_intern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_intern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_intern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center.scatter(re_intern_big,mue_intern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center.plot(linspace_re, linha_intern_big, color='red', linestyle='--', label=linha_interna_cd_label+'\n'+fr'$\alpha={alpha_med_int_big}$'+'\n'+fr'$\beta={beta_med_int_big}$')
	ax_center.scatter(re_intern_small,mue_intern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center.plot(linspace_re, linha_intern_small, color='blue', linestyle='--', label=linha_interna_el_label+'\n'+fr'$\alpha={alpha_med_int_small}$'+'\n'+fr'$\beta={beta_med_int_small}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()
####
if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ax_topx.plot(re_linspace,re_kde_intern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx.plot(re_linspace,re_kde_intern_small_cd(re_linspace),color='blue')
		ax_topx.axvline(np.average(re_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_intern_big_cd),'.3')}$')
		ax_topx.axvline(np.average(re_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_intern_small_cd),'.3')}$')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)

		ax_righty.plot(mue_kde_intern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_intern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_intern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_intern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_intern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_intern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center.scatter(re_intern_big_cd,mue_intern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_intern_big_cd, color='red', linestyle='--', label=linha_interna_cd_label_cd+'\n'+fr'$\alpha={alpha_med_int_big_cd}$'+'\n'+fr'$\beta={beta_med_int_big_cd}$')
		ax_center.scatter(re_intern_small_cd,mue_intern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_intern_small_cd, color='blue', linestyle='--', label=linha_interna_el_label_cd+'\n'+fr'$\alpha={alpha_med_int_small_cd}$'+'\n'+fr'$\beta={beta_med_int_small_cd}$')
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()

data_to_look=[box2,n2,rff_s,rff_ss]
vmin_to_look=[0,0.5,0,0]
vmax_to_look=[0.6,8,0.7,0.7]
labels_to_look=[r'$a_4/a \ 2 $',r'$n_2$',r'$RFF_S$',r'$RFF_SS$']
save_names=['box2','n2','rff_s','rff_ss']
for i,item in enumerate(data_to_look):
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0.18)
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ax_topx.plot(re_linspace,re_kde_extern_big(re_linspace),ls='dotted',color='red')
	ax_topx.plot(re_linspace,re_kde_extern_small(re_linspace),color='blue')
	ax_topx.axvline(np.average(re_extern_big),ls='-',color='red',label=fr'$\mu={format(np.average(re_extern_big),'.3')}$')
	ax_topx.axvline(np.average(re_extern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(re_extern_small),'.3')}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	ax_righty.plot(mue_kde_extern_big(mue_linspace),mue_linspace,ls='dotted',color='red')
	ax_righty.plot(mue_kde_extern_small(mue_linspace),mue_linspace,color='blue')
	ax_righty.axhline(np.average(mue_extern_big),ls='-',color='red',label=fr'$\mu={format(np.average(mue_extern_big),'.3')}$')
	ax_righty.axhline(np.average(mue_extern_small),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_extern_small),'.3')}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)

	sc=ax_center.scatter(re_extern_big,mue_extern_big,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd,c=item[lim_cd_big],cmap=cmap)
	ax_center.plot(linspace_re, linha_extern_big, color='red', linestyle='--', label=linha_externa_cd_label+'\n'+fr'$\alpha={alpha_med_ext_big}$'+'\n'+fr'$\beta={beta_med_ext_big}$')
	ax_center.scatter(re_extern_small,mue_extern_small,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el,c=item[lim_cd_small],cmap=cmap)
	ax_center.plot(linspace_re, linha_extern_small, color='blue', linestyle='--', label=linha_externa_el_label+'\n'+fr'$\alpha={alpha_med_ext_small}$'+'\n'+fr'$\beta={beta_med_ext_small}$')
	ax_center.legend(fontsize='small')
	ax_center.set_ylim(y2ss,y1ss)
	ax_center.set_xlim(x1ss,x2ss)
	ax_center.set_xlabel(xlabel)
	ax_center.set_ylabel(ylabel)

	cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.15, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0.)
	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(labels_to_look[i])
	cbar.ax.yaxis.set_ticks_position('left')
	cbar.ax.yaxis.set_label_position('left')
	plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}.png')
	plt.close()

if sample == 'L07':
	for i,item in enumerate(data_to_look):
		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ax_topx.plot(re_linspace,re_kde_extern_big_cd(re_linspace),ls='dotted',color='red')
		ax_topx.plot(re_linspace,re_kde_extern_small_cd(re_linspace),color='blue')
		ax_topx.axvline(np.average(re_extern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(re_extern_big_cd),'.3')}$')
		ax_topx.axvline(np.average(re_extern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(re_extern_small_cd),'.3')}$')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)

		ax_righty.plot(mue_kde_extern_big_cd(mue_linspace),mue_linspace,ls='dotted',color='red')
		ax_righty.plot(mue_kde_extern_small_cd(mue_linspace),mue_linspace,color='blue')
		ax_righty.axhline(np.average(mue_extern_big_cd),ls='-',color='red',label=fr'$\mu={format(np.average(mue_extern_big_cd),'.3')}$')
		ax_righty.axhline(np.average(mue_extern_small_cd),ls='-',color='blue',label=fr'$\mu={format(np.average(mue_extern_small_cd),'.3')}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)

		sc=ax_center.scatter(re_extern_big_cd,mue_extern_big_cd,marker='o',edgecolor='black',alpha=0.9,vmin=vmin_to_look[i],vmax=vmax_to_look[i],label=label_bojo_cd_cd,c=item[lim_cd_big & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_extern_big_cd, color='red', linestyle='--', label=linha_externa_cd_label_cd+'\n'+fr'$\alpha={alpha_med_ext_big_cd}$'+'\n'+fr'$\beta={beta_med_ext_big_cd}$')
		ax_center.scatter(re_extern_small_cd,mue_extern_small_cd,marker='s',edgecolor='black',alpha=0.9,label=label_bojo_el_cd,c=item[lim_cd_small & cd_cut],cmap=cmap)
		ax_center.plot(linspace_re, linha_extern_small_cd, color='blue', linestyle='--', label=linha_externa_el_label_cd+'\n'+fr'$\alpha={alpha_med_ext_small_cd}$'+'\n'+fr'$\beta={beta_med_ext_small_cd}$')
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(sc, cax=cax)
		cbar.set_label(labels_to_look[i])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		plt.savefig(f'{sample}stats_observation_desi/kormendy_rel_split/kormendy_rel_comp_cd_eel_{save_names[i]}_cds_zhao.png')
		plt.close()
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
# save_file=f'{sample}stats_observation_desi/svc_score_el_cd_dupla_{sample}_photutils.dat'

svc_lim=cd_lim & np.isfinite(np.log10(rff_s))
save_file=f'{sample}stats_observation_desi/svc_score_el_cd_dupla_{sample}.dat'

# var_name_vec.extend(vec_espec_casjobs)
# par_vec.extend(par_vec_casjobs)
# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}stats_observation_desi/svc_score_el_cd_dupla_{sample}_casjobs.dat'

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
# save_file=f'{sample}stats_observation_desi/svc_score_el_cd_{sample}_photutils.dat'

# var_name_vec.extend(vec_espec_casjobs)
# par_vec.extend(par_vec_casjobs)
# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
# save_file=f'{sample}stats_observation_desi/svc_score_el_cd_{sample}_casjobs.dat'
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
###################################################################################
###################################################################################
###################################################################################
#HISTOGRAMA DE RFF COM A LINHA DE CORTE


os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta',exist_ok=True)

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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_el.png')
plt.close(fig0)

#HISTOGRAMA DO RFF PARA cD
fig0=plt.figure(figsize=(9,7))
plt.hist(rff_s[lim_cd_big],bins=bins,color='red',alpha=0.4,edgecolor='black',label='cD')
plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlabel('RFF')
plt.ylabel('Objetos')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_true_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_cDs_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_elip_misc.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA cD -- subgrupo cd do Zhao

	fig0=plt.figure(figsize=(9,7))
	plt.hist(rff_s[lim_cd_big & cd_cut],bins=bins,color='red',alpha=0.4,edgecolor='black',label='cD[cD]')
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ ----')
	plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_true_cd_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/histogram_rff_lines_e_el_cd.png')
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
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',label='cD')
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c='blue',edgecolors='black',label='E(EL)')
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_color_coded.png')
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

axs[1].scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_big],y=eta[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(r'$\log\,RFF$')
axs[1].axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs[1].legend()

axs[2].scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_small],y=eta[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs[2].set_xlabel(r'$\log\,RFF$')
axs[2].legend()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_sub_grupos.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7))
axs.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=np.log10(rff_s)[lim_cd_big],y=eta[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
axs.set_xlim(limx)
axs.set_ylim(limy)
axs.set_ylabel(r'$\eta$')
axs.set_xlabel(r'$\log\,RFF$')
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_true_cds.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_extra_light.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_elipticas_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_cds_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_elipticas_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_elipticas_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_elipticas_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7))
	axs.scatter(np.log10(rff_s)[lim_cd_big & cd_cut],eta[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[cD]')
	sns.kdeplot(x=np.log10(rff_s)[lim_cd_big & cd_cut],y=eta[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	axs.set_xlim(limx)
	axs.set_ylim(limy)
	axs.set_ylabel(r'$\eta$')
	axs.set_xlabel(r'$\log\,RFF$')
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_true_cds_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rffxeta_extra_light_cd.png')
	plt.close()

###############################################
#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/bt',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=bt_cd_big,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_true_cd.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,bt_vec_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$B/T$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_bt_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_bt_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt/rffxeta_color_bt_e_el_cd.png')
	plt.close()

##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_true_cd_corr.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_elip_corr.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,bt_vec_corr
vec_label=r'$\log\,RFF$',r'$\eta$',r'$B/T$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_bt_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_bt_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/bt_corr/rffxeta_color_bt_corr_e_el_cd.png')
	plt.close()

###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=rff_ratio_cd_big,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],rff_ratio_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],rff_ratio_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],rff_ratio_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,rff_ratio
vec_label=r'$\log\,RFF$',r'$\eta$',r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=limx,limy,[0,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_rff_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_rff_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/rff_ratio/rffxeta_color_rff_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio',exist_ok=True)
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.9,1.5)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,chi2_ratio
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=limx,limy,[0.9,1.5]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_chi2_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_chi2_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S+S}^2/\chi_{S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/chi2_ratio/rffxeta_color_chi2_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(1000,-3000)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,delta_bic_obs
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\Delta BIC$'
xlim,ylim,zlim=limx,limy,[1000,-3000]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_delta_bic_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_delta_bic_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/delta_bic/rffxeta_color_delta_bic_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,1.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,e_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q$'
xlim,ylim,zlim=limx,limy,[0.5,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_ax_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_ax_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio/rffxeta_color_q_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/q1q2',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.1,2.8)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,axrat_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_1/q_2$'
xlim,ylim,zlim=limx,limy,[0.1,2.8]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_q_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_q_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/q1q2/rffxeta_color_q_ratio_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s),eta,c=box_s,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_s.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-0.5,0.5)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,box_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$a_4/a$'
xlim,ylim,zlim=limx,limy,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_box_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_box_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box/rffxeta_color_box_e_el_cd.png')
	plt.close()

###############################
# MAPA DE COR BOXINESS 1
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/box1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s),eta,c=box1,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box_1.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=box1[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_elip.png')
plt.close()

#MAPA DE COR BOXINESS 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=box1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=box1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],box1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],box1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],box1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$a_4/a \ 1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(box1[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$a_4/a \ 1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(box1[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$a_4/a \ 1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,box1
vec_label=r'$\log\,RFF$',r'$\eta$',r'$a_4/a \ 1$'
xlim,ylim,zlim=limx,limy,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_box1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_box1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=box1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=box1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=box1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=box1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=box1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a\ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=box1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=box1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box1/rffxeta_color_box1_e_el_cd.png')
	plt.close()

###############################
# MAPA DE COR BOXINESS 1
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/box2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s),eta,c=box2,edgecolors='black',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box_1.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[elip_lim],eta[elip_lim],c=box2[elip_lim],edgecolors='black',label='E',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_elip.png')
plt.close()

#MAPA DE COR BOXINESS 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=box2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=box2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],box2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],box2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],box2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$a_4/a \ 2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(box2[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$a_4/a \ 2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(box2[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$a_4/a \ 2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,box2
vec_label=r'$\log\,RFF$',r'$\eta$',r'$a_4/a \ 2$'
xlim,ylim,zlim=limx,limy,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_box2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_box2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=box2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=box2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & e_cut]),eta[elip_lim & e_cut],c=box2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & cd_cut]),eta[elip_lim & cd_cut],c=box2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[elip_lim & (ecd_cut | cde_cut)]),eta[elip_lim & (ecd_cut | cde_cut)],c=box2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a\ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=box2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=box2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/box2/rffxeta_color_box2_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR RAZÃO - re1/re2
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/re1re2',exist_ok=True)
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=np.log10(re_ratio_12[lim_cd_big]),edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=np.log10(re_ratio_12[lim_cd_small]),edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
plt.clim(-2.,0.1)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_el.png')
plt.close()

#ANALISE 3D - RE1/RE2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-2,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[elip_lim]),np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-2,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-2,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - RE1/RE2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(-2,1)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-2,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-2,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,re_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$\log_{10}({R_e}_{1}/{R_e}_{2})$'
xlim,ylim,zlim=limx,limy,[-2,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_re_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_re_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=re_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=re_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$\log_{10}({R_e}_{1}/{R_e}_{2})$', rotation=90)
	plt.clim(-2,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/re1re2/rffxeta_color_re_ratio_e_el_cd.png')
	plt.close()

################################
#MAPA DE COR RAZÃO - n1/n2
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/n1n2',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio.png')
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
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,5.)
ax.legend()
ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n_ratio_12
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_1/{n}_{2})$'
xlim,ylim,zlim=limx,limy,[0,5]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_n_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_n_ratio_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1n2/rffxeta_color_n_ratio_e_el_cd.png')
	plt.close()

################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,12)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,12])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,12])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_3d_projections.png')
plt.close()

vec_data=np.log10(rff_s),eta,n_s
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n$'
xlim,ylim,zlim=limx,limy,[0.5,12]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_n_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_n_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n_sersic/rffxeta_color_n_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/n1',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,15.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,15.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,15.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,15.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,15.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,15.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n1
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_1$'
xlim,ylim,zlim=limx,limy,[0.5,15]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_n1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_n1_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{1}$', rotation=90)
	plt.clim(0.5,15.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n1/rffxeta_color_n1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - INDICE DE SÉRSIC 2 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/n2',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 2 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=n2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=n2[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,4)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n2[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n2[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0.5,10.)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(n2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim(0.5,10.)
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$n_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(n2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$n_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,n2
vec_label=r'$\log\,RFF$',r'$\eta$',r'$n_2$'
xlim,ylim,zlim=limx,limy,[0.5,10]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_n2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_n2_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=n2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=n2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'${n}_{2}$', rotation=90)
	plt.clim(0.5,10.)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/n2/rffxeta_color_n2_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,e1
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_1$'
xlim,ylim,zlim=limx,limy,[0.,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_q1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_q1_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_1/rffxeta_color_q1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2',exist_ok=True)
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2.png')
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

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_big],eta[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(rff_s)[lim_cd_small],eta[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='cD',cmap=cmap)
plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(limx)
plt.ylim(limy)
plt.xlabel(r'$\log\,RFF$')
plt.ylabel(r'$\eta$')

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[elip_lim]),eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],np.log10(rff_s[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],eta[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(r'$\log\,RFF$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(rff_s[lim_cd_small]),eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(rff_s[lim_cd_big]),eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(limx)
axs[0].set_ylim(limy)
axs[0].set_xlabel(r'$\log\,RFF$')
axs[0].set_ylabel(r'$\eta$')
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],np.log10(rff_s[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(rff_s[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(limx)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(r'$\log\,RFF$')
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],eta[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],eta[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(limy)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(r'$\eta$')
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=np.log10(rff_s),eta,e2
vec_label=r'$\log\,RFF$',r'$\eta$',r'$q_2$'
xlim,ylim,zlim=limx,limy,[0.,1]
save_place=f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_q2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_q2_3d_spin.gif'
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_cd_zhao.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_e.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_cd.png')
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
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_big & cd_cut]),eta[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5,0.9]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	plt.scatter(np.log10(rff_s[lim_cd_small & cd_cut]),eta[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(r'$\log\,RFF$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{sample}stats_observation_desi/plano_rff_eta/ax_ratio_2/rffxeta_color_q2_e_el_cd.png')
	plt.close()

###################################################################################
###################################################################################
###################################################################################
#PLANO DE RAZÃO DE RAIOS (RE1/RE2) X N_2


os.makedirs(f'{sample}stats_observation_desi/plano_re_n2',exist_ok=True)

label_x=r'$R_{1}/R_{2}$'
label_y=r'$n_{2}$'
re_rat_lim=[-2,1.5]
n2_ylim=[0,15]

fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
axs[0].scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolor='black',label='E')
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].legend()
axs[1].scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolor='black',label='cD')
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/scatter_re_ratio_n2_geral.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolor='black',label='E')
axs.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolor='black',label='cD')
axs.set_xlim(re_rat_lim)
axs.set_ylim(n2_ylim)
axs.set_xlabel(label_x)
axs.set_ylabel(label_y)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/scatter_re_ratio_n2_geral.png')
plt.close()

#SUBPLOTS

fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,10),constrained_layout=True)
axs[0].scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim],y=n2[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs[0])
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_ylabel(label_y)
axs[0].set_xlabel(label_x)
axs[0].legend()
axs[1].scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big],y=n2[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small],y=n2[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_sub_grupos.png')
plt.close()

#UNITÁRIOS

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim],y=n2[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_xlim(re_rat_lim)
axs.set_ylim(n2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big],y=n2[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_xlim(re_rat_lim)
axs.set_ylim(n2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_true_cds.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small],y=n2[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.legend()
axs.set_xlim(re_rat_lim)
axs.set_ylim(n2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_extra_light.png')
plt.close()

if sample == 'L07':
	#ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[e_cut],n2[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
	sns.kdeplot(x=np.log10(re_ratio_12)[e_cut],y=n2[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_elipticas_zhao.png')
	plt.close()

	#cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[cd_cut],n2[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
	sns.kdeplot(x=np.log10(re_ratio_12)[cd_cut],y=n2[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_cds_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & e_cut],n2[elip_lim & e_cut],c='green',edgecolors='black',alpha=0.2,label='E[E]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & e_cut],y=n2[elip_lim & e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_elipticas_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & cd_cut],n2[elip_lim & cd_cut],c='green',edgecolors='red',alpha=0.2,label='E[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & cd_cut],y=n2[elip_lim & cd_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_elipticas_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],n2[elip_lim & (ecd_cut | cde_cut)],c='green',edgecolors='black',alpha=0.2,label='E[E/cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],y=n2[elip_lim & (ecd_cut | cde_cut)],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_elipticas_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_big & cd_cut],y=n2[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_true_cds_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c='blue',edgecolors='black',alpha=0.2,label='E(EL)[cD]')
	sns.kdeplot(x=np.log10(re_ratio_12)[lim_cd_small & cd_cut],y=n2[lim_cd_small & cd_cut],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.legend()
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rexn2_extra_light_cd.png')
	plt.close()

################################################
#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(np.log10(re_ratio_12),n2,c=bt_vec_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt.png')
plt.close()

#MAPA DE COR BT - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=bt_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=bt_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].set_ylim(n2_ylim)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylim(n2_ylim)
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,bt_vec_12
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)


if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=bt_vec_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=bt_vec_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=bt_vec_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt/rexn2_color_bt_e_el_cd.png')
	plt.close()
##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12),n2,c=bt_vec_corr,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_true_cd_corr.png')
plt.close()

fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=bt_e_corr,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_elip_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,bt_vec_corr
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/bt/re_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)


if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=bt_vec_corr[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=bt_vec_corr[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=bt_vec_corr[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_corr[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/bt_corr/rexn2_color_bt_e_el_cd.png')
	plt.close()
###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=rff_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio.png')
plt.close()

#MAPA DE RAZÃO DE RFF - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=rff_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=rff_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=rff_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],rff_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],rff_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],rff_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,1)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,rff_ratio
vec_label=label_x,label_y,r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/re_n2_rff_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/re_n2_rff_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=rff_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=rff_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=rff_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=rff_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/rff_ratio/rexn2_color_rff_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=chi2_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=chi2_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0.9,1.5)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,chi2_ratio
vec_label=label_x,label_y,r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=None,None,[0.9,1.5]
save_place=f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/re_n2_chi2_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/re_n2_chi2_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=chi2_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=chi2_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=chi2_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=chi2_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/chi2_ratio/rexn2_color_chi2_ratio_e_el_cd.png')
	plt.close()
#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/delta_bic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12),n2,c=delta_bic_obs,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic.png')
plt.close()

#MAPA DE COR DELTA BIC - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=delta_bic_obs[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - cDS

fig1=plt.figure(figsize=(9,7))

plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-1000)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(1000,-3000)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,delta_bic_obs
vec_label=label_x,label_y,r'$\Delta BIC$'
xlim,ylim,zlim=None,None,[1000,-3000]
save_place=f'{sample}stats_observation_desi/plano_re_n2/delta_bic/re_n2_delta_bic_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/delta_bic/re_n2_delta_bic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=delta_bic_obs[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=delta_bic_obs[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=delta_bic_obs[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=delta_bic_obs[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=delta_bic_obs[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/delta_bic/rexn2_color_delta_bic_e_el_cd.png')
	plt.close()
##############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=e_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=axrat_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')

ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0.5,1.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,e_s
vec_label=label_x,label_y,r'$q$'
xlim,ylim,zlim=None,None,[0.5,1.]
save_place=f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/re_n2_ax_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/re_n2_ax_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio/rexn2_color_ax_ratio_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/q1q2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=axrat_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=axrat_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_elip.png')
plt.close()


#MAPA DE COR RAZÃO Q1/Q2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,axrat_ratio_12
vec_label=label_x,label_y,r'$q_1/q_2$'
xlim,ylim,zlim=None,None,[0.1,2.8]
save_place=f'{sample}stats_observation_desi/plano_re_n2/q1q2/re_n2_q_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/q1q2/re_n2_q_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=axrat_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=axrat_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=axrat_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=axrat_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=axrat_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/q1q2/rexn2_color_q_ratio_e_el_cd.png')
	plt.close()
##############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=box_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=box_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,box_s
vec_label=label_x,label_y,r'$a_4/a$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_re_n2/box/re_n2_box_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/box/re_n2_box_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=box_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=box_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=box_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=box_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=box_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box/rexn2_color_box_e_el_cd.png')
	plt.close()
##############################
#MAPA DE COR BOXINESS 1
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/box1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=box1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=box1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=box1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=box1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],box1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],box1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],box1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a \ 1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box1[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$a_4/a \ 1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box1[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$a_4/a \ 1$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,box1
vec_label=label_x,label_y,r'$a_4/a \ 1$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_re_n2/box1/re_n2_box1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/box1/re_n2_box1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=box1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=box1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=box1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=box1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=box1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=box1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=box1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box1/rexn2_color_box1_e_el_cd.png')
	plt.close()

##############################
#MAPA DE COR BOXINESS 2
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/box2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=box2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=box2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=box2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=box2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],box2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],box2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],box2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a \ 2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box2[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box2[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box2[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$a_4/a \ 2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box2[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box2[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box2[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$a_4/a \ 2$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,box2
vec_label=label_x,label_y,r'$a_4/a \ 2$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_re_n2/box2/re_n2_box2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/box2/re_n2_box2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=box2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=box2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=box2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=box2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=box2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=box2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=box2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/box2/rexn2_color_box2_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR RAZÃO - n1/n2
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/n1n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=n_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n_ratio_12[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO n1/n2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n_ratio_12[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${n}_{1}/{n}_{2}$', rotation=90)
plt.clim(0,2.)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,5.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')

ax.set_zlim(0,5.)
ax.legend()

ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1/n_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()

plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,5.])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$n_1/n_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_ratio_12[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_ratio_12[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,5])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$n_1/n_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,n_ratio_12
vec_label=label_x,label_y,r'$n_1/n_2$'
xlim,ylim,zlim=None,None,[0.,5]
save_place=f'{sample}stats_observation_desi/plano_re_n2/n1n2/re_n2_n_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/n1n2/re_n2_n_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1/n_2$', rotation=90)
	plt.clim(0,5)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1n2/rexn2_color_n_ratio_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/n_sersic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=n_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,12)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,12])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,12])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_3d_projections.png')
plt.close()

vec_data=re_ratio_12,n2,n_s
vec_label=label_x,label_y,r'$n$'
xlim,ylim,zlim=None,None,[0.5,12]
save_place=f'{sample}stats_observation_desi/plano_re_n2/n_sersic/re_n2_n_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/n_sersic/re_n2_n_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n_sersic/rexn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/n1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=n1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=n1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,10.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,10.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_xlim(re_rat_lim)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_xlim(re_rat_lim)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,n1
vec_label=label_x,label_y,r'$n_1$'
xlim,ylim,zlim=None,None,[0.5,10]
save_place=f'{sample}stats_observation_desi/plano_re_n2/n1/re_n2_n1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/n1/re_n2_n1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=n1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=n1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=n1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_colotio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=n1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_colorio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=n1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_o_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/n1/rexn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=e1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=e1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,e1
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/re_n2_q1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/re_n2_q1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_1/rexn2_color_q1_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12),n2,c=e2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[elip_lim],n2[elip_lim],c=e2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_big],n2[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(np.log10(re_ratio_12)[lim_cd_small],n2[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(re_rat_lim)
plt.ylim(n2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[elip_lim]),n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],n2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(np.log10(re_ratio_12[lim_cd_small]),n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(np.log10(re_ratio_12[lim_cd_big]),n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(re_rat_lim)
axs[0].set_ylim(n2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(re_rat_lim)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],n2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],n2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(n2_ylim)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexeta_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=re_ratio_12,n2,e2
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/re_n2_q2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/re_n2_q2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=e2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=e2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & e_cut]),n2[elip_lim & e_cut],c=e2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & cd_cut]),n2[elip_lim & cd_cut],c=e2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[elip_lim & (ecd_cut | cde_cut)]),n2[elip_lim & (ecd_cut | cde_cut)],c=e2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_big & cd_cut]),n2[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(np.log10(re_ratio_12[lim_cd_small & cd_cut]),n2[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(re_rat_lim)
	plt.ylim(n2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_re_n2/ax_ratio_2/rexn2_color_q2_e_el_cd.png')
	plt.close()
###################################################################################
###################################################################################
###################################################################################
#PLANO DE N1 X NS
#PARA DIVIDIR AS cDS DAS E(EL)


os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns',exist_ok=True)

n1_xlim=[-0.5,10.5]
ns_ylim=[-0.5,12.5]

label_x=r'$n_1$'
label_y=r'$n_s$'


par_temp = np.column_stack([n1[cd_lim], n_s[cd_lim]])
coefs, inter, acc = svc_calc(par_temp, lim_cd_small[cd_lim].astype(int))

a, c = coefs
b = inter
# Plot
plt.figure()
plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='cD')
xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
yplot = -(a*xplot + b) / c
lim_y=yplot<np.max(n_s[cd_lim])
m = -(a/c)
b_plot = -(b/c)
label_line = rf'$F_s={acc:.3f}$'+'\n'+rf'$\alpha={m:.3f}$'+'\n'+rf'$\beta={b_plot:.3f}$'
plt.plot(xplot[lim_y], yplot[lim_y], color='black', lw=3, label=label_line)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n1_ns_split.png')
plt.close()

plt.figure(figsize=(8, 7))
plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.1, c='blue', edgecolor='black', label='E(EL)')
plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='cD')
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
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n1_ns_contorno.png')
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
axs[0].set_xlim(n1_xlim)
axs[0].set_ylim(ns_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)

axs[1].scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='cD')
sns.kdeplot(x=n1[lim_cd_big],y=n_s[lim_cd_big],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,ax=axs[1])#thresh=0)
axs[1].plot([],[],color='black',label=r'$\rho$')
axs[1].plot(xplot[lim_y], yplot[lim_y], color='black', lw=3)
axs[1].legend()
axs[1].set_xlim(n1_xlim)
axs[1].set_ylim(ns_ylim)
axs[1].set_xlabel(r'$n_1$')
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n1_ns_contorno.png')
plt.close()
##################################################
#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=bt_vec_12[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_2C.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=bt_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt/n1xns_color_bt_e_el_cd.png')
	plt.close()
#########################################################
#MAPA DE COR - RAZÃO BT - COM CORREÇÃO
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=bt_vec_corr[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_2C.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/bt_corr/n1xns_color_bt_e_el_cd.png')
	plt.close()
###################################################################################
#MAPA DE COR - RFF RATIO
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=rff_ratio[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,0.9)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_2C.png')
plt.close()

#MAPA DE COR DA RFF RATIO - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=rff_ratio_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,0.9)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE COR DA RFF RATIO - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=rff_ratio_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
plt.clim(0,0.9)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,0.9)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,0.9)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_S$', rotation=90)
	plt.clim(0,0.9)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/rff_ratio/n1xns_color_rff_ratio_e_el_cd.png')
	plt.close()
############
#MAPA DE COR - N2

os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/n2',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=n2[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,8)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_2C.png')
plt.close()

#MAPA DE COR N2 - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=n2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,8)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_true_cd.png')
plt.close()

#MAPA DE COR N2 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=n2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_2$', rotation=90)
plt.clim(0.5,8)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=n2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0.5,8)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=n2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0.5,8)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=n2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_2$', rotation=90)
	plt.clim(0.5,8)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/n2/n1xns_color_n2_e_el_cd.png')
	plt.close()
#############################
#MAPA DE COR - CHI2 RATIO
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=chi2_ratio[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0.9,1.1)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_2C.png')
plt.close()

#MAPA DE COR CHI2 RATIO - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR CHI2 RATIO - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0.9,1.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0.9,1.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi^{2}_{S}/\chi^{2}_{S+S}$', rotation=90)
	plt.clim(0.9,1.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/chi2_ratio/n1xns_color_chi2_ratio_e_el_cd.png')
	plt.close()

#MAPA DE COR - BOX 1
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/box1',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=box1[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_2C.png')
plt.close()

#MAPA DE COR DA BOX 1 - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=box1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 1 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=box1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=box1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=box1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=box1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box1/n1xns_color_box1_e_el_cd.png')
	plt.close()

#MAPA DE COR - BOX 2
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/box2',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=box2[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_2C.png')
plt.close()

#MAPA DE COR DA BOX 2 - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=box2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 2 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=box2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(0,0.6)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=box2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=box2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=box2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(0,0.6)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/box2/n1xns_color_box2_e_el_cd.png')
	plt.close()

#MAPA DE COR - ETA
os.makedirs(f'{sample}stats_observation_desi/plano_n1_ns/eta',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n1[cd_lim],n_s[cd_lim],c=eta[cd_lim],edgecolors='black',cmap=cmap,label='2C')
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,0.1)
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_2C.png')
plt.close()

#MAPA DE COR DA BOX 1 - cDS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=eta[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,0.1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_true_cd.png')
plt.close()

#MAPA DE COR DA BOX 1 - EXTRA LIGHT 
fig1=plt.figure(figsize=(9,7))
plt.scatter(n1[lim_cd_small],n2[lim_cd_small],c=eta[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\eta$', rotation=90)
plt.clim(0,0.1)
plt.legend()
plt.xlim(n1_xlim)
plt.ylim(ns_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_e_el.png')
plt.close()

if sample=='L07':
	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[cd_cut],n2[cd_cut],c=eta[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,0.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_cd_zhao.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_big & cd_cut],n2[lim_cd_big & cd_cut],c=eta[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,0.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n1[lim_cd_small & cd_cut],n2[lim_cd_small & cd_cut],c=eta[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\eta$', rotation=90)
	plt.clim(0,0.1)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n1_ns/eta/n1xns_color_eta_e_el_cd.png')
	plt.close()

###################################################################################
###################################################################################
###################################################################################
#PLANO DE RAZÃO DE RAIOS (N1/N2) x N2
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2',exist_ok=True)

label_x=r'$n_{1}/n_{2}$'
label_y=r'$log_{10} n_{2}$'
n_rat_lim=[-0.5,18]
logn2_ylim=[-0.5,1.3]

fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolor='black',label='E')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
axs[1].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolor='black',label='cD')
axs[2].set_xlabel(label_x)
axs[2].legend()

plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/scatter_n_ratio_n2_geral.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
axs.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolor='black',label='E')
axs.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolor='black',label='E(EL)')
axs.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolor='black',label='cD')
axs.set_xlim(n_rat_lim)
axs.set_ylim(logn2_ylim)
axs.set_xlabel(label_x)
axs.set_ylabel(label_y)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/scatter_n_ratio_n2_geral.png')
plt.close()
#SUBPLOTS

fig,axs=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,10),constrained_layout=True)
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=n_ratio_12[elip_lim],y=np.log10(n2)[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs[0])
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_ylabel(label_y)
axs[0].set_xlabel(label_x)
axs[0].legend()
axs[1].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=n_ratio_12[lim_cd_big],y=np.log10(n2)[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs[1])
axs[1].set_xlabel(label_x)
axs[1].legend()
axs[2].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=n_ratio_12[lim_cd_small],y=np.log10(n2)[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs[2])
axs[2].set_xlabel(label_x)
axs[2].legend()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_sub_grupos.png')
plt.close()

#UNITÁRIOS

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c='green',edgecolors='black',alpha=0.2,label='E')
sns.kdeplot(x=n_ratio_12[elip_lim],y=np.log10(n2)[elip_lim],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_xlim(n_rat_lim)
axs.set_ylim(logn2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c='red',edgecolors='black',alpha=0.2,label='cD')
sns.kdeplot(x=n_ratio_12[lim_cd_big],y=np.log10(n2)[lim_cd_big],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.set_xlim(n_rat_lim)
axs.set_ylim(logn2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
axs.legend()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_true_cds.png')
plt.close()

fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
axs.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c='blue',edgecolors='black',alpha=0.2,label='E(EL)')
sns.kdeplot(x=n_ratio_12[lim_cd_small],y=np.log10(n2)[lim_cd_small],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
axs.legend()
axs.set_xlim(n_rat_lim)
axs.set_ylim(logn2_ylim)
axs.set_ylabel(label_y)
axs.set_xlabel(label_x)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_extra_light.png')
plt.close()

if sample == 'L07':
	#ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
	sns.kdeplot(x=n_ratio_12[e_cut],y=np.log10(n2)[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_zhao.png')
	plt.close()

	#cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
	sns.kdeplot(x=n_ratio_12[cd_cut],y=np.log10(n2)[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_cds_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c='green',edgecolors='black',alpha=0.2,label='E[E]')
	sns.kdeplot(x=n_ratio_12[elip_lim & e_cut],y=np.log10(n2)[elip_lim & e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c='green',edgecolors='red',alpha=0.2,label='E[cD]')
	sns.kdeplot(x=n_ratio_12[elip_lim & cd_cut],y=np.log10(n2)[elip_lim & cd_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c='green',edgecolors='black',alpha=0.2,label='E[E/cD]')
	sns.kdeplot(x=n_ratio_12[elip_lim & (ecd_cut | cde_cut)],y=np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_elipticas_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[cD]')
	sns.kdeplot(x=n_ratio_12[lim_cd_big & cd_cut],y=np.log10(n2)[lim_cd_big & cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	axs.legend()
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_true_cds_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
	axs.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c='blue',edgecolors='black',alpha=0.2,label='E(EL)[cD]')
	sns.kdeplot(x=n_ratio_12[lim_cd_small & cd_cut],y=np.log10(n2)[lim_cd_small & cd_cut],fill=False,levels=30,color="blue",linewidths=1,alpha=0.8,thresh=0,ax=axs)
	axs.legend()
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_ylabel(label_y)
	axs.set_xlabel(label_x)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/nxn2_extra_light_cd.png')
	plt.close()
################################################

#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt',exist_ok=True)
fig1=plt.figure(figsize=(9,7),constrained_layout=True)
plt.scatter(n_ratio_12,np.log10(n2),c=bt_vec_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt.png')
plt.close()

#MAPA DE COR BT - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=bt_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_elip.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=bt_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_true_cd.png')
plt.close()

#MAPA DE COR DA RAZÃO BT - EXTRA LIGHT 

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=bt_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],bt_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),bt_vec_12
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/n_ratio_n2_bt_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/n_ratio_n2_bt_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=bt_vec_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=bt_vec_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=bt_vec_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=bt_vec_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=bt_vec_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=bt_vec_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt/nxn2_color_bt_e_el_cd.png')
	plt.close()

##################################
#MAPA DE COR - RAZÃO BT - CORRIGIDO
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12,np.log10(n2),c=bt_vec_corr,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=bt_cd_big_corr,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_true_cd_corr.png')
plt.close()

fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=bt_e_corr,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_elip_corr.png')
plt.close()


fig1=plt.figure(figsize=(9,7))
plt.suptitle('BT CORRIGIDO')
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=bt_cd_small_corr,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$B/T$', rotation=90)
plt.clim(0,1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_e_el_corr.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],bt_e_corr,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_e_corr,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(bt_cd_small_corr,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_e_corr,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(bt_cd_small_corr,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_projections.png')
plt.close()

#ANALISE 3D - RAZÃO BT 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],bt_cd_small_corr,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],bt_cd_big_corr,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$B/T$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(bt_cd_small_corr,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(bt_cd_big_corr,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$B/T$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(bt_cd_small_corr,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(bt_cd_big_corr,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$B/T$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),bt_vec_corr
vec_label=label_x,label_y,r'$B/T$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/n_ratio_n2_bt_corr_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/n_ratio_n2_bt_corr_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=bt_vec_corr[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=bt_vec_corr[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=bt_vec_corr[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=bt_vec_corr[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=bt_vec_corr[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=bt_vec_corr[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=bt_vec_corr[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$B/T$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/bt_corr/nxn2_color_bt_corr_e_el_cd.png')
	plt.close()

###################################
#MAPA DE RAZÃO DE RFF
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=rff_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio.png')
plt.close()

#MAPA DE RAZÃO DE RFF - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=rff_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_elip.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=rff_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_true_cd.png')
plt.close()

#MAPA DE RAZÃO DE RFF - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=rff_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
plt.clim(0.,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE RFF 

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],rff_ratio_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],rff_ratio_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],rff_ratio_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$RFF_{S+S}/RFF_{S}$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(rff_ratio_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(rff_ratio_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(rff_ratio_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlim([0,1])
axs[1].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(rff_ratio_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(rff_ratio_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(rff_ratio_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$RFF_{S+S}/RFF_{S}$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),rff_ratio
vec_label=label_x,label_y,r'$RFF_{S+S}/RFF_{S}$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/n_ratio_n2_rff_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/n_ratio_n2_rff_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=rff_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=rff_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=rff_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=rff_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=rff_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=rff_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=rff_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$RFF_{S+S}/RFF_{S}$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/rff_ratio/nxn2_color_rff_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR RAZÃO DE CHI2
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=chi2_ratio,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=chi2_ratio[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=chi2_ratio[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO DE CHI2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=chi2_ratio[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
plt.clim(0.9,1.1)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO DE CHI2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],chi2_ratio[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],chi2_ratio[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],chi2_ratio[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.9,1.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(chi2_ratio[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(chi2_ratio[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(chi2_ratio[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.9,1.5])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(chi2_ratio[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(chi2_ratio[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(chi2_ratio[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.9,1.5])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$\chi_{S}^2/\chi_{S+S}^2$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),chi2_ratio
vec_label=label_x,label_y,r'$\chi_{S}^2/\chi_{S+S}^2$'
xlim,ylim,zlim=None,None,[0.9,1.5]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/n_ratio_n2_chi2_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/n_ratio_n2_chi2_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=chi2_ratio[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=chi2_ratio[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=chi2_ratio[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=chi2_ratio[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=chi2_ratio[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=chi2_ratio[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=chi2_ratio[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\chi_{S}^2/\chi_{S+S}^2$', rotation=90)
	plt.clim(0.9,1.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/chi2_ratio/nxn2_color_chi2_ratio_e_el_cd.png')
	plt.close()

#################################
#MAPA DE COR DELTA BIC
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=delta_bic_obs,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-3000)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic.png')
plt.close()

#MAPA DE COR DELTA BIC - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=delta_bic_obs[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-3000)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_elip.png')
plt.close()

#MAPA DE COR DELTA BIC - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=delta_bic_obs[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-3000)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_true_cd.png')
plt.close()

#MAPA DE COR DELTA BIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=delta_bic_obs[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$\Delta BIC$', rotation=90)
plt.clim(1000.,-3000)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_el.png')
plt.close()

#ANALISE 3D - DELTA BIC

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],delta_bic_obs[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],delta_bic_obs[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],delta_bic_obs[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(1000,-3000)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$\Delta BIC$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(delta_bic_obs[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(delta_bic_obs[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(delta_bic_obs[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([1000,-3000])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$\Delta BIC$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(delta_bic_obs[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(delta_bic_obs[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(delta_bic_obs[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([1000,-3000])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$\Delta BIC$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),delta_bic_obs
vec_label=label_x,label_y,r'$\Delta BIC$'
xlim,ylim,zlim=None,None,[1000,-3000]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/n_ratio_n2_delta_bic_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/n_ratio_n2_delta_bic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=delta_bic_obs[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=delta_bic_obs[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=delta_bic_obs[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=delta_bic_obs[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=delta_bic_obs[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=delta_bic_obs[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=delta_bic_obs[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$\Delta BIC$', rotation=90)
	plt.clim(1000,-3000)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/delta_bic/nxn2_color_delta_bic_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR RAZÃO AXIAL
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=e_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=axrat_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=axrat_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=axrat_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q$', rotation=90)
plt.clim(0.5,1.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_q_e_el.png')
plt.close()

#ANALISE 3D - RAZÃO AXIAL

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],axrat_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,1.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,1.])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,1.])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q$')
axs[2].set_ylabel(label_y)
axs[2].legend()
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e_s
vec_label=label_x,label_y,r'$q$'
xlim,ylim,zlim=None,None,[0.5,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/n_ratio_n2_ax_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/n_ratio_n2_ax_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q$', rotation=90)
	plt.clim(0.5,1.)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio/nxn2_color_ax_ratio_e_el_cd.png')
	plt.close()

###############################
#MAPA DE COR RAZÃO - Q1/Q2
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=axrat_ratio_12,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=axrat_ratio_12[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=axrat_ratio_12[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO Q1/Q2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=axrat_ratio_12[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1/q_2$', rotation=90)
plt.clim(0.7,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_e_el.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],axrat_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(axrat_ratio_12[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(axrat_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - Q1/Q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],axrat_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],axrat_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.1,2.8)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1/q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(axrat_ratio_12[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(axrat_ratio_12[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.1,2.8])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_1/q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(axrat_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(axrat_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.1,2.8])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_1/q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_q_ratio_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),axrat_ratio_12
vec_label=label_x,label_y,r'$q_1/q_2$'
xlim,ylim,zlim=None,None,[0.1,2.8]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/n_ratio_n2_q_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/n_ratio_n2_q_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=axrat_ratio_12[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=axrat_ratio_12[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=axrat_ratio_12[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=axrat_ratio_12[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=axrat_ratio_12[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=axrat_ratio_12[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=axrat_ratio_12[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1/q_2$', rotation=90)
	plt.clim(0.1,2.8)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/q1q2/nxn2_color_axrat_ratio_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR BOXINESS
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/box',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=box_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_s.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=box_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=box_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=box_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],box_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],box_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],box_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$a_4/a$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$a_4/a$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),box_s
vec_label=label_x,label_y,r'$a_4/a$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/box/n_ratio_n2_box_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/box/n_ratio_n2_box_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=box_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=box_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=box_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=box_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=box_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=box_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=box_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box/nxn2_color_box_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR BOXINESS 1
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=box1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=box1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=box1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=box1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 1$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],box1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],box1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],box1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a \ 1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box1[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$a_4/a \ 1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box1[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$a_4/a \ 1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),box1
vec_label=label_x,label_y,r'$a_4/a \ 1$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/n_ratio_n2_box1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/n_ratio_n2_box1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=box1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=box1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=box1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=box1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=box1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=box1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=box1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 1$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box1/nxn2_color_box1_e_el_cd.png')
	plt.close()
###############################
#MAPA DE COR BOXINESS 2
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=box2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2.png')
plt.close()

#########################
#MAPA DE COR BOXINESS - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=box2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_elip.png')
plt.close()

#MAPA DE COR BOXINESS - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=box2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_true_cd.png')
plt.close()

#MAPA DE COR BOXINESS - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=box2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$a_4/a \ 2$', rotation=90)
plt.clim(-0.5,0.5)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_el.png')
plt.close()

#ANALISE 3D - BOXINESS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],box2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],box2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],box2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(-0.5,0.5)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$a_4/a \ 2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(box2[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(box2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(box2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-0.5,0.5])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$a_4/a \ 2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(box2[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(box2[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(box2[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-0.5,0.5])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$a_4/a \ 2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),box2
vec_label=label_x,label_y,r'$a_4/a \ 2$'
xlim,ylim,zlim=None,None,[-0.5,0.5]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/n_ratio_n2_box2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/n_ratio_n2_box2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=box2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=box2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=box2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=box2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=box2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=box2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=box2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$a_4/a \ 2$', rotation=90)
	plt.clim(-0.5,0.5)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/box2/nxn2_color_box2_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR RAZÃO - re1/re2
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=np.log10(re_ratio_12),edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - ELIPTICAS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=np.log10(re_ratio_12[elip_lim]),edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_elip.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=np.log10(re_ratio_12[lim_cd_big]),edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_true_cd.png')
plt.close()

#MAPA DE COR RAZÃO re1/re2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=np.log10(re_ratio_12[lim_cd_small]),edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'${R_e}_{1}/{R_e}_{2}$', rotation=90)
plt.clim(-2,2.)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_el.png')
plt.close()

#ANALISE 3D - re1/re2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],np.log10(re_ratio_12[elip_lim]),alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],np.log10(re_ratio_12[lim_cd_small]),alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],np.log10(re_ratio_12[lim_cd_big]),alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,5.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'${R_e}_1/{R_e}_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[elip_lim]),n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-2,2.])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[elip_lim]),np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-2,2])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_projections.png')
plt.close()

#ANALISE 3D - re1/re2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],re_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],re_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,5.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'${R_e}_1/{R_e}_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(np.log10(re_ratio_12[lim_cd_small]),n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(np.log10(re_ratio_12[lim_cd_big]),n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([-2,2.])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(np.log10(re_ratio_12[lim_cd_small]),np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(np.log10(re_ratio_12[lim_cd_big]),np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([-2,2])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'${R_e}_1/{R_e}_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),re_ratio_12
vec_label=label_x,label_y,r'${R_e}_1/{R_e}_2$'
xlim,ylim,zlim=None,None,[-2,2]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/n_ratio_n2_re_ratio_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/n_ratio_n2_re_ratio_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=np.log10(re_ratio_12)[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=np.log10(re_ratio_12)[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=np.log10(re_ratio_12)[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=np.log10(re_ratio_12)[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=np.log10(re_ratio_12)[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=np.log10(re_ratio_12)[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=np.log10(re_ratio_12)[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'${R_e}_1/{R_e}_2$', rotation=90)
	plt.clim(-2,2)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/re1re2/nxn2_color_re_ratio_e_el_cd.png')
	plt.close()
################################
#MAPA DE COR - INDICE DE SÉRSIC
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=n_s,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=n_sersic_e,edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=n_sersic_cd_big,edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=n_sersic_cd_small,edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n$', rotation=90)
plt.clim(3,8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_el.png')
plt.close()

#ANALISE 3D - n1/n2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],n_sersic_e,alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n_sersic_cd_small,alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n_sersic_cd_big,alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,12)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n_sersic_e,n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n_sersic_cd_small,n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n_sersic_cd_big,n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,12])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$n$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n_sersic_e,np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n_sersic_cd_small,np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n_sersic_cd_big,np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,12])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$n$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_3d_projections.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),n_s
vec_label=label_x,label_y,r'$n$'
xlim,ylim,zlim=None,None,[0.5,12]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/n_ratio_n2_n_sersic_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/n_ratio_n2_n_sersic_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=n_s[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=n_s[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=n_s[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=n_s[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=n_s[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=n_s[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=n_s[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n$', rotation=90)
	plt.clim(0.5,12)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n_sersic/nxn2_color_n_e_el_cd.png')
	plt.close()
##########################
#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=n1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=n1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_elip.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=n1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=n1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$n_1$', rotation=90)
plt.clim(2,6)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_el.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],n1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,10.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(n1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(n1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_projections.png')
plt.close()

#ANALISE 3D - n1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],n1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],n1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0.5,10.)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$n_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(n1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(n1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0.5,10.])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$n_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(n1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(n1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0.5,10.])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$n_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),n1
vec_label=label_x,label_y,r'$n_1$'
xlim,ylim,zlim=None,None,[0.5,10]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/n_ratio_n2_n1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/n_ratio_n2_n1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=n1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=n1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=n1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=n1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=n1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=n1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=n1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$n_1$', rotation=90)
	plt.clim(0.5,10)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/n1/nxn2_color_n1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 1 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=e1,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=e1[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 1 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=e1[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_true_cd.png')
plt.close()

#MAPA DE COR INDICE DE SÉRSIC 1 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=e1[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_1$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_el.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],e1[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_projections.png')
plt.close()

#ANALISE 3D - q1

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e1[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e1[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_1$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e1[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e1[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_1$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e1[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e1[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_1$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e1
vec_label=label_x,label_y,r'$q_1$'
xlim,ylim,zlim=None,None,[0,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/n_ratio_n2_q1_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/n_ratio_n2_q1_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e1[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e1[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e1[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e1[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e1[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e1[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e1[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_1$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_1/nxn2_color_q1_e_el_cd.png')
	plt.close()

##########################
#MAPA DE COR - RAZÃO AXIAL 2 - SS
os.makedirs(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2',exist_ok=True)
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12,np.log10(n2),c=e2,edgecolors='black',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2- ELIPTICAS
fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],c=e2[elip_lim],edgecolors='black',label='E',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_elip.png')
plt.close()

#MAPA DE COR RAZÃO AXIAL 2 - cDS

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],c=e2[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_true_cd.png')
plt.close()
#MAPA DE COR RAZÃO AXIAL 2 - EXTRA LIGHT

fig1=plt.figure(figsize=(9,7))
plt.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],c=e2[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
cbar=plt.colorbar()
cbar.set_label(r'$q_2$', rotation=90)
plt.clim(0.5,0.8)
plt.legend()
plt.xlim(n_rat_lim)
plt.ylim(logn2_ylim)
plt.xlabel(label_x)
plt.ylabel(label_y)
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_el.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],e2[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[elip_lim],n_ratio_12[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[1].scatter(e2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[elip_lim],np.log10(n2)[elip_lim],alpha=0.4,c='green',edgecolor='black',label='E')
axs[2].scatter(e2[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_projections.png')
plt.close()

#ANALISE 3D - q2

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],e2[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
ax.scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],e2[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
ax.set_zlim(0,1)
ax.legend()
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(r'$q_2$')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_only2c.png')
plt.close()

fig,axs = plt.subplots(1,3,figsize=(15,5))
#XY
axs[0].scatter(n_ratio_12[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[0].scatter(n_ratio_12[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[0].set_xlim(n_rat_lim)
axs[0].set_ylim(logn2_ylim)
axs[0].set_xlabel(label_x)
axs[0].set_ylabel(label_y)
axs[0].legend()
#ZX
axs[1].scatter(e2[lim_cd_small],n_ratio_12[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[1].scatter(e2[lim_cd_big],n_ratio_12[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[1].set_xlim([0,1])
axs[1].set_ylim(n_rat_lim)
axs[1].set_xlabel(r'$q_2$')
axs[1].set_ylabel(label_x)
axs[1].legend()
#ZY
axs[2].scatter(e2[lim_cd_small],np.log10(n2)[lim_cd_small],alpha=0.5,c='blue',edgecolor='black',label='E(EL)')
axs[2].scatter(e2[lim_cd_big],np.log10(n2)[lim_cd_big],alpha=0.6,c='red',edgecolor='black',label='cD')
axs[2].set_xlim([0,1])
axs[2].set_ylim(logn2_ylim)
axs[2].set_xlabel(r'$q_2$')
axs[2].set_ylabel(label_y)
axs[2].legend()

plt.tight_layout()
plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_3d_projections_only2c.png')
plt.close()

vec_data=n_ratio_12,np.log10(n2),e2
vec_label=label_x,label_y,r'$q_2$'
xlim,ylim,zlim=None,None,[0.,1]
save_place=f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/n_ratio_n2_q2_3d_spin_only2c.gif',f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/n_ratio_n2_q2_3d_spin.gif'
make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place)

if sample=='L07':
	#ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c=e2[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_zhao.png')
	plt.close()

	#cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c=e2[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_cd_zhao.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & e_cut],np.log10(n2)[elip_lim & e_cut],c=e2[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_e.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & cd_cut],np.log10(n2)[elip_lim & cd_cut],c=e2[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_cd.png')
	plt.close()

	#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[elip_lim & (ecd_cut | cde_cut)],np.log10(n2)[elip_lim & (ecd_cut | cde_cut)],c=e2[elip_lim & (ecd_cut | cde_cut)],edgecolors='black',label='E[E/cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_misc.png')
	plt.close()

	#cDs -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_big & cd_cut],np.log10(n2)[lim_cd_big & cd_cut],c=e2[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_cd_cd.png')
	plt.close()

	#EXTRA LIGHT -- SUBGRUPO cD ZHAO
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(n_ratio_12[lim_cd_small & cd_cut],np.log10(n2)[lim_cd_small & cd_cut],c=e2[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
	cbar=plt.colorbar()
	cbar.set_label(r'$q_2$', rotation=90)
	plt.clim(0,1)
	plt.legend()
	plt.xlim(n_rat_lim)
	plt.ylim(logn2_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{sample}stats_observation_desi/plano_n_ratio_n2/ax_ratio_2/nxn2_color_q2_e_el_cd.png')
	plt.close()

