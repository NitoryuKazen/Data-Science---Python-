import os 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp,nan
import numpy as np
import numpy.ma as ma
import scipy.special as scs
import scipy.optimize as scp
import scipy.stats as sst
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import matplotlib.pyplot as plt
from subprocess import call

warnings.filterwarnings("ignore")
import multiprocess as mp

#############################
def linfunc(x,a,b):
	return a*x+b
############################
def sersic(x,ie,re,n):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.)))+22.5
	return i
##############################
def sersicexp(x,ie,re,n,i0,rd):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))+abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
#################################
def doublesersic(x,ie,re,n,i0,rd,nd):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	bnd=2.*abs(nd)-1./3+4./405/abs(nd)+46./25515/abs(nd)**2+131./1148175/abs(nd)**3-2194697./30690717750/abs(nd)**4
	s1=abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))
	s2=abs(i0)*np.exp(-bnd*((np.divide(x,abs(rd)))**(1./abs(nd))-1.))
	i=-2.5*np.log10(s1+s2)+22.5
	return i
###############################
def envel(x,i0,rd):
	i=-2.5*np.log10(abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
####################################
def modelagem(x,y,sig):
	bounds_s=((-np.inf,-np.inf,0.5),(np.inf,np.inf,15.))	
	bounds_se=((-np.inf,-np.inf,0.5,-np.inf,-np.inf),(np.inf,np.inf,10.,np.inf,np.inf))	
	bounds_ss=((-np.inf,-np.inf,0.5,-np.inf,-np.inf,0.5),(np.inf,np.inf,10.,np.inf,np.inf,15.))	
	
	spopt,spcov=scp.curve_fit(sersic,x,y,sigma=sig,p0=[0.3,30.,4.],bounds=bounds_s,maxfev=1000000)
	chisq_s=np.sum(np.power((y-(sersic(x,*spopt)))/(sig),2))/(len(y)-3.)
	#
	try:
		sepopt,sepcov = scp.curve_fit(sersicexp,x,y,sigma=sig,p0=[spopt[0],spopt[1]/2.,spopt[2],spopt[0]/3.,1.75*spopt[1]],bounds=bounds_se,maxfev=1000000)
	except:
		sepopt=[spopt[0],spopt[1]/2.,4.,spopt[0]/3.,1.75*spopt[1]]
	chisq_se=np.sum(np.power((y-(sersicexp(x,*sepopt)))/(sig),2))/(len(y)-5.)
	#
	try:
		sspopt,sspcov = scp.curve_fit(doublesersic,x,y,sigma=sig,p0=[spopt[0],spopt[1]/2.,spopt[2],spopt[0]/3.,1.75*spopt[1],2.],bounds=bounds_ss,maxfev=1000000)
	except:
		sspopt=[spopt[0],spopt[1]/2.,spopt[2],spopt[0]/3.,1.75*spopt[1],2.]
	chisq_ss=np.sum(np.power((y-(doublesersic(x,*sspopt)))/(sig),2))/(len(y)-6.)
	#
	for i in range(20):
		try:
			nsepopt,nsepcov = scp.curve_fit(sersicexp,x,y,sigma=sig,p0=[np.random.uniform(0.01,0.3),np.random.uniform(5.,30.), np.random.uniform(1.,10.),np.random.uniform(0.,0.1),np.random.uniform(1.,50.)],bounds=bounds_se,maxfev=10000000)
		except:
			nsepopt=[np.random.uniform(0.01,0.3),np.random.uniform(5.,30.),np.random.uniform(0.5,10.), np.random.uniform(0.01,0.3),np.random.uniform(1.,50.)]
		chisq_se_temp=np.sum(np.power((y-(sersicexp(x,*nsepopt)))/(sig),2))/(len(y)-5.)
		if chisq_se_temp < chisq_se:
			sepopt=1.*nsepopt
			sepcov=1.*nsepcov
			chisq_se=chisq_se_temp

	chirat_1=chisq_s/chisq_se
	chisq_s_1=chisq_s
	#
	for i in range(20):
		try:
			nsspopt,nsspcov = scp.curve_fit(doublesersic,x,y, sigma=sig,p0=[np.random.uniform(0.01,0.3),np.random.uniform(5.,30.),np.random.uniform(0.5,10.),np.random.uniform(0.01,0.2),np.random.uniform(1.,50.), np.random.uniform(0.5,15.)],bounds=bounds_ss,maxfev=10000000)
		except:
			nsspopt=[np.random.uniform(0.01,0.3),np.random.uniform(1.,30.),np.random.uniform(0.5,10.), np.random.uniform(0.01,0.2),np.random.uniform(1.,50.),np.random.uniform(0.5,15.)]
		chisq_ss_temp=np.sum(np.power((y-(doublesersic(x,*nsspopt)))/(sig),2))/(len(y)-6.)

		if chisq_ss_temp<chisq_ss:
			sspopt=1.*nsspopt
			sspcov=1.*nsspcov
			chisq_ss=chisq_ss_temp

	chirat_2=chisq_s/chisq_ss
	
	return chirat_1,chirat_2,chisq_s_1,chisq_se,chisq_ss,spopt,sepopt,sspopt,spcov,sepcov,sspcov
###################################
def total_flux(ie,re,n):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	fac=2*n-1
	if fac > 0:
		i = ie*2*np.pi*np.power(re,2)*n*(np.divide(np.exp(bn),np.power(bn,2*n)))*scs.factorial(fac)
	elif fac <= 0:
		nfac=fac+1
		i = ie*2*np.pi*np.power(re,2)*n*(np.divide(np.exp(bn),np.power(bn,2*n)))*scs.gamma(nfac)
	return i
################################################################################################################################################
def param_intern_extern(slice_interno_start,slice_interno_stop,slice_externo_start,slice_externo_stop,sma,a3,b3,a4,b4,a3_err,b3_err,a4_err,b4_err,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g,sigmaskyg):
	###################################################
	#LIMITES
	slice_interno=slice(slice_interno_start,slice_interno_stop)
	slice_externo=slice(slice_externo_start,slice_externo_stop)
	###
	#Interno
	intens_g_interno=intens_fix_g[slice_interno]
	intens_r_interno=intens_fix_r[slice_interno]
	int_err_g_interno=int_fix_err_g[slice_interno]
	int_err_r_interno=int_fix_err_r[slice_interno]
	#Externo
	intens_g_externo=intens_fix_g[slice_externo]
	intens_r_externo=intens_fix_r[slice_externo]		
	int_err_g_externo=int_fix_err_g[slice_externo]
	int_err_r_externo=int_fix_err_r[slice_externo]
	###
	cor_lim_interno=np.where(intens_g_interno > sigmaskyg)
	cor_lim_externo=np.where(intens_g_externo > sigmaskyg)
	###################################################
	#REGIÃO INTERNA
	
	intens_g_interno = intens_g_interno[cor_lim_interno]
	intens_r_interno = intens_r_interno[cor_lim_interno]

	int_err_g_interno = int_err_g_interno[cor_lim_interno]
	int_err_r_interno = int_err_r_interno[cor_lim_interno]

	mag_g_interno = 22.5 -2.5*np.log10(intens_g_interno) + 2.5*(log10(0.1569166))
	mag_r_interno = 22.5 -2.5*np.log10(intens_r_interno) + 2.5*(log10(0.1569166))

	mag_err_g_interno = 2*2.5*np.log10(np.exp(1.))*int_err_g_interno/intens_g_interno 
	mag_err_r_interno = 2*2.5*np.log10(np.exp(1.))*int_err_r_interno/intens_r_interno

	smagr_interno=sma[cor_lim_interno]
	test_gr_interno = mag_g_interno-mag_r_interno
	test_gr_err_interno = np.sqrt(mag_err_g_interno**2 + mag_err_r_interno**2)
	
	test_gr_interno = test_gr_interno[np.isfinite(test_gr_interno)]
	test_gr_err_interno = test_gr_err_interno[np.isfinite(test_gr_interno)]
	smagr_interno=smagr_interno[np.isfinite(test_gr_interno)]

	###################################################
	#REGIÃO EXTERNA
	
	intens_g_externo = intens_g_externo[cor_lim_externo]
	intens_r_externo = intens_r_externo[cor_lim_externo]

	int_err_g_externo = int_err_g_externo[cor_lim_externo]
	int_err_r_externo = int_err_r_externo[cor_lim_externo]

	mag_g_externo = 22.5 -2.5*np.log10(intens_g_externo) + 2.5*(log10(0.1569166))
	mag_r_externo = 22.5 -2.5*np.log10(intens_r_externo) + 2.5*(log10(0.1569166))

	mag_err_g_externo = 2*2.5*np.log10(np.exp(1.))*int_err_g_externo/intens_g_externo 
	mag_err_r_externo = 2*2.5*np.log10(np.exp(1.))*int_err_r_externo/intens_r_externo

	smagr_externo=sma[cor_lim_externo]
	test_gr_externo = mag_g_externo-mag_r_externo
	test_gr_err_externo = np.sqrt(mag_err_g_externo**2 + mag_err_r_externo**2)
	
	test_gr_externo = test_gr_externo[np.isfinite(test_gr_externo)]
	test_gr_err_externo = test_gr_err_externo[np.isfinite(test_gr_externo)]
	smagr_externo=smagr_externo[np.isfinite(test_gr_externo)]
	
	if len(test_gr_interno) >=3 and len(test_gr_externo) >=3:
		fix_log_interno_popt,fix_log_interno_pcov=scp.curve_fit(linfunc,np.log10(smagr_interno),test_gr_interno,sigma=test_gr_err_interno)
		fix_interno_popt,fix_interno_pcov=scp.curve_fit(linfunc,np.power(smagr_interno,0.25),test_gr_interno,sigma=test_gr_err_interno)
		alpha_log_interno=fix_log_interno_popt[0]
		alpha_interno=fix_interno_popt[0]
		
		fix_log_externo_popt,fix_log_externo_pcov=scp.curve_fit(linfunc,np.log10(smagr_externo),test_gr_externo,sigma=test_gr_err_externo)
		fix_externo_popt,fix_externo_pcov=scp.curve_fit(linfunc,np.power(smagr_externo,0.25),test_gr_externo,sigma=test_gr_err_externo)
		alpha_log_externo=fix_log_externo_popt[0]
		alpha_externo=fix_externo_popt[0]

	else:

		alpha_log_interno,alpha_log_externo,alpha_interno,alpha_externo = -1000,-1000,-1000,-1000

	######################################################################################################
	#A3
	a3_interno=a3[slice_interno]
	a3_externo=a3[slice_externo]
	#
	a3_err_interno=a3_err[slice_interno]
	a3_err_externo=a3_err[slice_externo]
	#
	a3_med_interno=np.average(a3_interno,weights=1/np.power(a3_err_interno,2))
	a3_med_externo=np.average(a3_externo,weights=1/np.power(a3_err_externo,2))
	#A4
	a4_interno=a4[slice_interno]
	a4_externo=a4[slice_externo]
	#
	a4_err_interno=a4_err[slice_interno]
	a4_err_externo=a4_err[slice_externo]
	#
	a4_med_interno=np.average(a4_interno,weights=1/np.power(a4_err_interno,2))
	a4_med_externo=np.average(a4_externo,weights=1/np.power(a4_err_externo,2))
	#B3
	b3_interno=b3[slice_interno]
	b3_externo=b3[slice_externo]
	#
	b3_err_interno=b3_err[slice_interno]
	b3_err_externo=b3_err[slice_externo]
	#
	b3_med_interno=np.average(b3_interno,weights=1/np.power(b3_err_interno,2))
	b3_med_externo=np.average(b3_externo,weights=1/np.power(b3_err_externo,2))
	#B4
	b4_interno=b4[slice_interno]
	b4_externo=b4[slice_externo]
	#
	b4_err_interno=b4_err[slice_interno]
	b4_err_externo=b4_err[slice_externo]
	#
	b4_med_interno=np.average(b4_interno,weights=1/np.power(b4_err_interno,2))
	b4_med_externo=np.average(b4_externo,weights=1/np.power(b4_err_externo,2))
	#DISKNESS
	diskness_interno=a4[slice_interno]/sma[slice_interno]
	diskness_externo=a4[slice_externo]/sma[slice_externo]
	#
	diskness_err_interno=a4_err[slice_interno]/sma[slice_interno]
	diskness_err_externo=a4_err[slice_externo]/sma[slice_externo]
	#
	diskness_med_interno=np.average(diskness_interno,weights=1/np.power(diskness_err_interno,2))
	diskness_med_externo=np.average(diskness_externo,weights=1/np.power(diskness_err_externo,2))
	#	
	vec_intern_extern=[alpha_log_interno,alpha_log_externo,alpha_interno,alpha_externo,a3_med_interno,a3_med_externo,a4_med_interno,a4_med_externo,b3_med_interno,b3_med_externo,b4_med_interno,b4_med_externo,diskness_med_interno,diskness_med_externo]

	return vec_intern_extern
####################################
def modelflux(vec_se,vec_ss):
	flux_sersic_se=total_flux(*vec_se[:3])
	flux_env_se=total_flux(*vec_se[3:5],1.)
	flux_se=flux_env_se+flux_sersic_se


	frac_sersic_se=flux_sersic_se/flux_se
	frac_env_se=flux_env_se/flux_se

	flux_sersic_1_ss=total_flux(*vec_ss[:3])
	flux_sersic_2_ss=total_flux(*vec_ss[3:6])
	flux_ss=flux_sersic_1_ss+flux_sersic_2_ss
	frac_sersic_1_ss=flux_sersic_1_ss/flux_ss
	frac_sersic_2_ss=flux_sersic_2_ss/flux_ss

	return flux_sersic_se,flux_env_se,flux_se,frac_sersic_se,frac_env_se,flux_sersic_1_ss,flux_sersic_2_ss,flux_ss,frac_sersic_1_ss,frac_sersic_2_ss
###################################################
def model_sep(vec_se,vec_ss,sma,a3,b3,a4,b4,a3_err,b3_err,a4_err,b4_err,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g,sigmaskyg):
	vec_diff_se=sersic(sma,*vec_se[:3])-envel(sma,*vec_se[3:5])#
	
	xx_se=ma.masked_where(sersic(sma,*vec_se[:3])-envel(sma,*vec_se[3:5]) < 0,vec_diff_se)
	
	env_slice=ma.notmasked_contiguous(xx_se)
	bojo_slice=ma.clump_masked(xx_se)
	
	if len(env_slice) == 1 and len(bojo_slice) == 1:
		if env_slice[0].start == 0:
			if 5 < env_slice[0].stop < len(sma):
				mod_which_se='mod_split_se'
				func_intern_se='exp_intern'
				se_slices=[env_slice[0].start,env_slice[0].stop,bojo_slice[0].start,bojo_slice[0].stop]
			else:
				mod_which_se='mod_split_se_small'
				func_intern_se='exp_intern'		
				se_slices=[env_slice[0].start,env_slice[0].stop,bojo_slice[0].start,bojo_slice[0].stop]
		else:
			if env_slice[0].start > 5:
				if env_slice[0].stop - env_slice[0].start > 10:
					mod_which_se='mod_split_se'
					func_intern_se='sersic_intern'
					se_slices=[bojo_slice[0].start,bojo_slice[0].stop,env_slice[0].start,env_slice[0].stop]
				elif env_slice[0].stop - env_slice[0].start <= 10:
					mod_which_se='mod_split_se_small'
					func_intern_se='sersic_intern'
					se_slices=[bojo_slice[0].start,bojo_slice[0].stop,env_slice[0].start,env_slice[0].stop]
			else:	
				mod_which_se='mod_split_se_small'
				func_intern_se='sersic_intern'
				se_slices=[bojo_slice[0].start,bojo_slice[0].stop,env_slice[0].start,env_slice[0].stop]
	elif len(env_slice) == 1 and len(bojo_slice) == 2:
		##cd 3 componentes
		mod_which_se='mod_split_se_double'
		func_intern_se='sersic_intern'
		se_slices=[-1,-1,-1,-1]
	else:
		##cd splitada
		mod_which_se='mod_part_ss'
		if bojo_slice != []:
			func_intern_se='only_sersic'
			se_slices=[bojo_slice[0].start,bojo_slice[0].stop,-1,-1]
		if env_slice != []:
			func_intern_se='only_exp'
			se_slices=[-1,-1,env_slice[0].start,env_slice[0].stop]

	if -1 not in se_slices:
		param_se_intern_extern=param_intern_extern(*se_slices,sma,a3,b3,a4,b4,a3_err,b3_err,a4_err,b4_err,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g,sigmaskyg)
	else:
		param_se_intern_extern=[-1000 for _ in range(14)]

	#######################################################################################

	vec_diff_ss=sersic(sma,*vec_ss[:3])-sersic(sma,*vec_ss[3:])#
	
	xx_ss=ma.masked_where(sersic(sma,*vec_ss[:3])-sersic(sma,*vec_ss[3:]) < 0,vec_diff_ss)
	sersic_1_slice=ma.notmasked_contiguous(xx_ss)
	sersic_2_slice=ma.clump_masked(xx_ss)
	if len(sersic_2_slice) == 1 and len(sersic_1_slice) == 1:
		if sersic_1_slice[0].start == 0:
			if 5 < sersic_1_slice[0].stop < len(sma):
				mod_which_ss='mod_split_ss'
				func_intern_ss='sersic_1_intern'
				ss_slices=[sersic_1_slice[0].start,sersic_1_slice[0].stop,sersic_2_slice[0].start,sersic_2_slice[0].stop]
			else:
				mod_which_ss='mod_split_ss_small'
				func_intern_ss='sersic_1_intern'
				ss_slices=[sersic_1_slice[0].start,sersic_1_slice[0].stop,sersic_2_slice[0].start,sersic_2_slice[0].stop]
		else:
			if sersic_1_slice[0].start > 5:
				if sersic_1_slice[0].stop - sersic_1_slice[0].start > 10:
					mod_which_ss='mod_split_ss'
					func_intern_ss='sersic_2_intern'
					ss_slices=[sersic_2_slice[0].start,sersic_2_slice[0].stop,sersic_1_slice[0].start,sersic_1_slice[0].stop]
				elif sersic_1_slice[0].stop - sersic_1_slice[0].start <= 10:
					mod_which_ss='mod_split_ss_small'
					func_intern_ss='sersic_2_intern'
					ss_slices=[sersic_2_slice[0].start,sersic_2_slice[0].stop,sersic_1_slice[0].start,sersic_1_slice[0].stop]
			else:	
				mod_which_ss='mod_split_ss_small'
				func_intern_ss='sersic_2_intern'
				ss_slices=[sersic_2_slice[0].start,sersic_2_slice[0].stop,sersic_1_slice[0].start,sersic_1_slice[0].stop]
	elif len(sersic_1_slice) == 1 and len(sersic_2_slice) == 2:
		##cd 3 componentes
		mod_which_ss='mod_split_ss_double'
		func_intern_ss='sersic_intern'
		ss_slices=-1,-1,-1,-1
	else:
		##cd splitada
		mod_which_ss='mod_part_ss'
		if sersic_1_slice != []:
			func_intern_ss='only_sersic_1'
			ss_slices=[sersic_1_slice[0].start,sersic_1_slice[0].stop,-1,-1]
		
		elif sersic_2_slice != []:
			ss_slices=[-1,-1,sersic_2_slice[0].start,sersic_2_slice[0].stop]
			func_intern_ss='only_sersic_2'
	if -1 not in ss_slices:
		param_ss_intern_extern=param_intern_extern(*ss_slices,sma,a3,b3,a4,b4,a3_err,b3_err,a4_err,b4_err,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g,sigmaskyg)
	else:
		param_ss_intern_extern=[-1000 for _ in range(14)]
	return mod_which_se,func_intern_se,mod_which_ss,func_intern_ss,*param_se_intern_extern,*param_ss_intern_extern
################################################################################################################################################
###IMAGEM E GRÁFICOS
################################################################################################################################################
def coefF(cluster,sma,a3,a3_err,a4,a4_err,b3,b3_err,b4,b4_err):
	a3_iqr,a4_iqr,b3_iqr,b4_iqr=[(sst.iqr(a3),np.percentile(a3,[25,75])),(sst.iqr(a4),np.percentile(a4,[25,75])),(sst.iqr(b3),np.percentile(b3,[25,75])),(sst.iqr(b4),np.percentile(b4,[25,75]))]
	#
	a3_lim,a4_lim,b3_lim,b4_lim=(a3_iqr[1][0]-1.5*a3_iqr[0] < a3) & (a3 < a3_iqr[1][1]+1.5*a3_iqr[0]),(a4_iqr[1][0]-1.5*a4_iqr[0] < a4) & (a4 < a4_iqr[1][1]+1.5*a4_iqr[0]),(b3_iqr[1][0]-1.5*b3_iqr[0] < b3) & (b3 < b3_iqr[1][1]+1.5*b3_iqr[0]),(b4_iqr[1][0]-1.5*b4_iqr[0] < b4) & (b4 < b4_iqr[1][1]+1.5*b4_iqr[0])

	a3,a4,b3,b4=a3[a3_lim],a4[a4_lim],b3[b3_lim],b4[b4_lim]
	
	a3_err,a4_err,b3_err,b4_err=a3_err[a3_lim],a4_err[a4_lim],b3_err[b3_lim],b4_err[b4_lim]
		
	sma_a3,sma_a4,sma_b3,sma_b4=sma[a3_lim],sma[a4_lim],sma[b3_lim],sma[b4_lim]
	
	vec_med=[]
	slowvec=[]
	vec_erro=[]
	fraction=0.9
	coefs=[(a3,a3_err,r'$a_3$',0,'a3',sma_a3),(a4,a4_err,r'$a_4$',1,'a4',sma_a4),(b3,b3_err,r'$b_3$',2,'b3',sma_b3),(b4,b4_err,r'$b_4$',3,'b4',sma_b4),(a4/sma_a4,a4_err/sma_a4,r'$a_4/sma$',4,'diskness',sma_a4)]
	for coef,coef_err,nomey,j,savename,radius in coefs:
		plt.figure()
		plt.suptitle(cluster+' '+savename,fontsize=10)
		med_1=np.average(coef[np.where(coef_err != 0.0)],weights=1/np.power(coef_err[np.where(coef_err != 0.0)],2))
		plt.errorbar(np.power(radius,0.25),coef,yerr=coef_err,fmt='o',markersize=2,color='k',ecolor='0.8',label='1.0 '+format(med_1,'.3E'))
		slow=lowess(coef,np.power(radius,0.25),frac=fraction)
		slowvec.append(slow)
		med_slow=np.average(slow[:,1])		
		plt.axhline(y=0.,xmin=0,xmax=1,c='k')
		plt.plot(slow[:,0],slow[:,1],linewidth=2,ls='--',c='red',label='0.9 '+format(med_slow,'.3E'))
		plt.ylabel(nomey)
		plt.xlabel(r'$R^{1/4}$ (arcsec)')
		plt.legend(title=r'$F-\mu$',fontsize='x-small')
		plt.ylim(np.min(coef)-abs(np.max(coef)-np.min(coef))/5.,np.max(coef)+abs(np.max(coef)-np.min(coef))/5.)

		plt.savefig(f'WHL_sky/{cluster}/{cluster}_{savename}.png')
		plt.savefig(f'WHL_sky/{savename}/{cluster}.png')
		plt.close()
				
		slope_slow,slope_erro=np.polyfit(np.power(slow[:,0],4.),slow[:,1],1,cov=True)
		vec_med.extend([med_1,med_slow,slope_slow[0]])
		vec_erro.append(slope_erro[0][0])
	return slowvec,vec_med,sma_a3,sma_a4,sma_b3,sma_b4,a3,a4,b3,b4,a3_err,a4_err,b3_err,b4_err,vec_erro
##############################################################################################################################
############################################################################################################################
def modelplots(cluster,sma,mag_iso,mag_err,spopt,sepopt,sspopt,chisq_s,chisq_se,chisq_ss):
	ylim=[np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.]
	fig,axs=plt.subplots(1,3,figsize=(15, 5))#,sharey=True)
	plt.subplots_adjust(hspace=0.35, wspace=0.35)

	axs[0].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=4,zorder=0, color='k')
	axs[0].plot(np.power(sma,0.25),sersic(sma,*spopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_s ,color='blue',linewidth=5)
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylim(ylim)
	axs[0].legend()

	axs[1].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=4,zorder=0, color='k')
	axs[1].plot(np.power(sma,0.25),sersicexp(sma,*sepopt),label=r'S+E Fit, $\chi^2_\nu=$%.2f' % chisq_se,color='orange',linewidth=3)
	axs[1].plot(np.power(sma,0.25),sersic(sma,*sepopt[:3]),label='S',color='red',linewidth=2)
	axs[1].plot(np.power(sma,0.25),envel(sma,*sepopt[3:5]),label='E',color='yellow',linewidth=2)
	axs[1].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylim(ylim)
	axs[1].legend()
	
	axs[2].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=4,zorder=0, color='k')
	axs[2].plot(np.power(sma,0.25),doublesersic(sma,*sspopt),label=r'S+S Fit, $\chi^2_\nu=$%.2f' % chisq_ss,color='orange',linewidth=3)
	axs[2].plot(np.power(sma,0.25),sersic(sma,*sspopt[:3]),label='S_b',color='red',linewidth=2)
	axs[2].plot(np.power(sma,0.25),sersic(sma,*sspopt[3:6]),label='S_e',color='yellow',linewidth=2)
	axs[2].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[2].set_ylim(ylim)
	axs[2].legend()

	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_models.png')
	plt.savefig(f'WHL_sky/sersic_fits/{cluster}_models.png')
	plt.close()

	fig,axs=plt.subplots(1,2,figsize=(10, 5))#,sharey=True)
	plt.subplots_adjust(hspace=0.35, wspace=0.35)

	axs[0].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=4,zorder=0, color='k')
	axs[0].plot(np.power(sma,0.25),sersic(sma,*spopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_s ,color='blue',linewidth=5)
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylim(ylim)
	axs[0].legend()
	
	axs[1].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=4,zorder=0, color='k')
	axs[1].plot(np.power(sma,0.25),doublesersic(sma,*sspopt),label=r'S+S Fit, $\chi^2_\nu=$%.2f' % chisq_ss,color='orange',linewidth=3)
	axs[1].plot(np.power(sma,0.25),sersic(sma,*sspopt[:3]),label='S_b',color='red',linewidth=2)
	axs[1].plot(np.power(sma,0.25),sersic(sma,*sspopt[3:6]),label='S_e',color='yellow',linewidth=2)
	axs[1].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylim(ylim)
	axs[1].legend()

	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_models_s_ss.png')
	plt.savefig(f'WHL_sky/sersic_fits/{cluster}_models_s_ss.png')
	plt.close()
	
	################################################################################################

	plt.figure()
	plt.plot(np.power(sma,0.25),mag_iso,linewidth=2,color='k')
	plt.plot(np.power(sma,0.25),sersic(sma,*spopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_s ,color='blue',linewidth=2)
	plt.fill_between(np.power(sma,0.25),mag_iso,sersic(sma,*spopt),color='m')
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylim(ylim)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_rff_linear.png')	
	plt.savefig(f'WHL_sky/rff_linear/{cluster}.png')		
	plt.close()

	return
####################################################################################################################################################
####################################################################################################################################################
def testegr(cluster,sma,sigmaskyg,sigmaskyr,intens_free_g,int_free_err_g,intens_free_r,int_free_err_r, intens_fix_g,int_fix_err_g,intens_fix_r,int_fix_err_r):
	if np.isnan(sigmaskyg) == True:
		sigmasky=sigmaskyr
	else:
		sigmasky=sigmaskyg
	intens_g_free = intens_free_g[np.where(intens_free_g > sigmasky)]
	intens_r_free = intens_free_r[np.where(intens_free_g > sigmasky)]

	intens_g_fix = intens_fix_g[np.where(intens_fix_g > sigmasky)]
	intens_r_fix = intens_fix_r[np.where(intens_fix_g > sigmasky)]

	int_err_g_free = int_free_err_g[np.where(intens_free_g > sigmasky)]
	int_err_r_free = int_free_err_r[np.where(intens_free_g > sigmasky)]

	int_err_g_fix = int_fix_err_g[np.where(intens_fix_g > sigmasky)]
	int_err_r_fix = int_fix_err_r[np.where(intens_fix_g > sigmasky)]

	mag_g_free = 22.5 -2.5*np.log10(intens_g_free) + 2.5*(log10(0.1569166))
	mag_r_free = 22.5 -2.5*np.log10(intens_r_free) + 2.5*(log10(0.1569166))

	mag_g_fix = 22.5 -2.5*np.log10(intens_g_fix) + 2.5*(log10(0.1569166))
	mag_r_fix = 22.5 -2.5*np.log10(intens_r_fix) + 2.5*(log10(0.1569166))

	mag_err_g_free = 2*2.5*np.log10(np.exp(1.))*int_err_g_free/intens_g_free 
	mag_err_r_free = 2*2.5*np.log10(np.exp(1.))*int_err_r_free/intens_r_free 

	mag_err_g_fix = 2*2.5*np.log10(np.exp(1.))*int_err_g_fix/intens_g_fix 
	mag_err_r_fix = 2*2.5*np.log10(np.exp(1.))*int_err_r_fix/intens_r_fix 

	smagr_free_0=sma[np.where(intens_free_g>sigmasky)]
	smagr_fix_0=sma[np.where(intens_fix_g>sigmasky)]
	
	test_gr_free_0 = mag_g_free-mag_r_free
	test_gr_fix_0 = mag_g_fix-mag_r_fix
	
	test_gr_err_free_0 = np.sqrt(mag_err_g_free**2 + mag_err_r_free**2)
	test_gr_err_fix_0 = np.sqrt(mag_err_g_fix**2 + mag_err_r_fix**2)
	
	test_gr_free = test_gr_free_0[np.isfinite(test_gr_free_0)]
	test_gr_fix = test_gr_fix_0[np.isfinite(test_gr_fix_0)]
	
	test_gr_err_free = test_gr_err_free_0[np.isfinite(test_gr_free_0)]
	test_gr_err_fix = test_gr_err_fix_0[np.isfinite(test_gr_fix_0)]

	smagr_free=smagr_free_0[np.isfinite(test_gr_free_0)]
	smagr_fix=smagr_fix_0[np.isfinite(test_gr_fix_0)]
	
	try:
		fix_log_popt,fix_log_cov=np.polyfit(np.log10(smagr_fix),test_gr_fix,deg=1,w=test_gr_err_fix,cov=True)
		fix_popt,fix_cov=np.polyfit(np.power(smagr_fix,0.25),test_gr_fix,deg=1,w=test_gr_err_fix,cov=True)
		free_log_popt,free_log_cov=np.polyfit(np.log10(smagr_free),test_gr_free,deg=1,w=test_gr_err_free,cov=True)
		free_popt,free_cov=np.polyfit(np.power(smagr_free,0.25),test_gr_free,deg=1,w=test_gr_err_free,cov=True)

		plt.figure()
		plt.errorbar(np.log10(smagr_free),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
		plt.plot(np.log10(smagr_free),linfunc(np.log10(smagr_free),*free_log_popt),c='r',label='Linear Fit')
		plt.xlabel(r'$R$ log(arcsec)')
		plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		plt.ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
		plt.legend(fontsize='x-small')
		plt.tight_layout()
		plt.savefig(f'WHL_sky/{cluster}/{cluster}_gr_free_log.png')
		plt.savefig(f'WHL_sky/cor_log_free/{cluster}_gr_free_log.png')
		plt.close()

		plt.figure()
		plt.errorbar(np.log10(smagr_fix),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
		plt.plot(np.log10(smagr_fix),linfunc(np.log10(smagr_fix),*fix_log_popt),c='r',label='Linear Fit')
		plt.xlabel(r'$R$ log(arcsec)')
		plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		plt.ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
		plt.legend(fontsize='x-small')
		plt.tight_layout()
		plt.savefig(f'WHL_sky/{cluster}/{cluster}_gr_fixx_log.png')
		plt.savefig(f'WHL_sky/cor_log_fix/{cluster}_gr_fixx_log.png')
		plt.close()
		#######################################################################################
			
		plt.figure()
		plt.errorbar(np.power(smagr_free,0.25),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
		plt.plot(np.power(smagr_free,0.25),linfunc(np.power(smagr_free,0.25),*free_popt),c='r',label='Linear Fit')
		plt.xlabel(r'$R^{1/4}$ (arcsec)')
		plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		plt.ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
		plt.legend(fontsize='x-small')
		plt.tight_layout()
		plt.savefig(f'WHL_sky/{cluster}/{cluster}_gr_free.png')
		plt.savefig(f'WHL_sky/cor_free/{cluster}_gr_free.png')
		plt.close()
		
		plt.figure()
		plt.errorbar(np.power(smagr_fix,0.25),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
		plt.plot(np.power(smagr_fix,0.25),linfunc(np.power(smagr_fix,0.25),*fix_popt),c='r',label='Linear Fit')
		plt.xlabel(r'$R^{1/4}$ (arcsec)')
		plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		plt.ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
		plt.legend(fontsize='x-small')
		plt.tight_layout()
		plt.savefig(f'WHL_sky/{cluster}/{cluster}_gr_fixx.png')
		plt.savefig(f'WHL_sky/cor_fix/{cluster}_gr_fixx.png')
		plt.close()
		cor_erro=[fix_log_cov[0][0],fix_cov[0][0],free_log_cov[0][0],free_cov[0][0]]
	
	except:
		fix_log_popt=fix_log_alpha,fix_log_beta=[np.nan,np.nan]
		fix_popt=fix_alpha,fix_beta=[np.nan,np.nan]
		free_log_popt=free_log_alpha,free_log_beta=[np.nan,np.nan]
		free_popt=free_alpha,free_beta=[np.nan,np.nan]

		fix_log_cov=[np.nan,np.nan]
		fix_cov=[np.nan,np.nan]
		free_log_cov=[np.nan,np.nan]
		free_cov=[np.nan,np.nan]
		cor_erro=[fix_log_cov[0],fix_cov[0],free_log_cov[0],free_cov[0]]

	corvec=[fix_log_popt[0],fix_popt[0],free_log_popt[0],free_popt[0]]
	return smagr_free,test_gr_free,test_gr_err_free,smagr_fix,test_gr_fix,test_gr_err_fix,free_popt,fix_popt,corvec,cor_erro
######################################################################################################################################################
###############################################################################################################################################
def analise(cluster,sma,mag_iso,mag_err,eps,ellip_err,pa,pa_err,sma_a3,a3,a3_err,sma_a4,a4,a4_err,sma_b3,b3,b3_err,sma_b4,b4,b4_err,slowvec, smagr_free,test_gr_free,test_gr_err_free,smagr_fix,test_gr_fix,test_gr_err_fix,free_popt,fix_popt):
	fig, axs = plt.subplots(3,3,figsize=(27,21))
	plt.subplots_adjust(hspace=0.35, wspace=0.35)
	plt.suptitle(cluster,fontsize=10)

	axs[0,0].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[0,0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0,0].set_ylim([np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.,np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.])
	axs[0,0].invert_yaxis()

	axs[0,1].errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=2, color='k',ecolor='0.8')
	axs[0,1].set_ylim([0,np.max(eps)+np.max(eps)*0.1])
	axs[0,1].set_ylabel(r'$e$')
	
	axs[0,2].errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180.,fmt='o',markersize=2, color='k',ecolor='0.8')
	axs[0,2].set_ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	axs[0,2].set_ylabel('PA (deg)')
	
	axs[1,0].errorbar(np.power(sma_a3,0.25),a3,yerr=a3_err,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[1,0].axhline(y=0.,xmin=0,xmax=1,c='k')
	axs[1,0].plot(slowvec[0][:,0],slowvec[0][:,1],linewidth=2,ls='--',c='red')
	axs[1,0].set_ylabel(r'$a_3$')

	axs[1,1].errorbar(np.power(sma_a4,0.25),a4,yerr=a4_err,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[1,1].axhline(y=0.,xmin=0,xmax=1,c='k')
	axs[1,1].plot(slowvec[1][:,0],slowvec[1][:,1],linewidth=2,ls='--',c='red')
	axs[1,1].set_ylabel(r'$a_4$')
	
	axs[1,2].errorbar(np.power(sma_b3,0.25),b3,yerr=b3_err,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[1,2].axhline(y=0.,xmin=0,xmax=1,c='k')
	axs[1,2].plot(slowvec[2][:,0],slowvec[2][:,1],linewidth=2,ls='--',c='red')
	axs[1,2].set_ylabel(r'$b_3$')
	
	axs[2,0].errorbar(np.power(sma_b4,0.25),b4,yerr=b4_err,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[2,0].axhline(y=0.,xmin=0,xmax=1,c='k')
	axs[2,0].plot(slowvec[3][:,0],slowvec[3][:,1],linewidth=2,ls='--',c='red')
	axs[2,0].set_ylabel(r'$b_4$')
	axs[2,0].set_xlabel(r'$R^{1/4}$ (arcsec)')

	try:
		axs[2,1].errorbar(np.power(smagr_free,0.25),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
		axs[2,1].plot(np.power(smagr_free,0.25),linfunc(np.power(smagr_free,0.25),*free_popt),c='r')
		axs[2,1].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		axs[2,1].set_ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
		axs[2,1].set_xlabel(r'$R^{1/4}$ (arcsec)')

		axs[2,2].errorbar(np.power(smagr_fix,0.25),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
		axs[2,2].plot(np.power(smagr_fix,0.25),linfunc(np.power(smagr_fix,0.25),*fix_popt),c='r')
		axs[2,2].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		axs[2,2].set_xlabel(r'$R^{1/4}$ (arcsec)')
		axs[2,2].set_ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
	except:
		axs[2,1].errorbar([],[])
		axs[2,1].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		axs[2,1].set_xlabel(r'$R^{1/4}$ (arcsec)')

		axs[2,2].errorbar([],[])
		axs[2,2].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
		axs[2,2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	plt.savefig(f'WHL_sky/analise/{cluster}.png')
	plt.close()
	return
################################################################################################################################################
##############################################################################################################################################
def rawplots(cluster,sma,mag_iso,mag_err,eps,ellip_err,pa,pa_err):
	fig,axs=plt.subplots(1,3,figsize=(9, 3))
	plt.subplots_adjust(hspace=0.35, wspace=0.35)
	limy=[np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.,np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.]
	axs[0].errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=3,color='k',ecolor='0.8')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_ylim(limy)
	axs[0].yaxis.set_inverted(True)

	axs[1].errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=3, color='k',ecolor='0.8')
	axs[1].set_ylim([0,np.max(eps)+np.max(eps)*0.1])
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylabel(r'$e$')
	
	axs[2].errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=3, color='k',ecolor='0.8')
	axs[2].set_ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	axs[2].set_xlabel(r'$R$ (arcsec)')
	axs[2].set_ylabel('PA (deg)')

	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_iso_info.png')	
	plt.savefig(f'WHL_sky/mag_e_pa/{cluster}_iso_info.png')	
	plt.close()

	plt.figure()
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=3,color='k',ecolor='0.8')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.ylim(limy)
	plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_mu.png')
	plt.close()

	plt.figure()
	plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=3, color='k',ecolor='0.8')
	plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$e$')
	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_elip.png')	
	plt.close()

	plt.figure()
	plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=3, color='k',ecolor='0.8')
	plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	plt.xlabel(r'$R$ (arcsec)')
	plt.ylabel('PA (deg)')
	plt.tight_layout()
	plt.savefig(f'WHL_sky/{cluster}/{cluster}_pa.png')	
	plt.close()	
	
	return
###############################################################################################################################################
####################################################################################################################################################
def bcgfigs(cluster,n_controle,ok,okk,info,output_geral,output_modelos,output_erros):
	if os.path.isfile(f'WHL_gr/{cluster}/iso_table_gr.dat') == False:
		with open(output_geral,'a') as graph_data:
			vec=['-1000' for i in range(42)]
			graph_data.write(f'{cluster} {' '.join(vec)}\n')
		with open(output_erros,'a') as graph_erros:
			vec=['-1000' for i in range(11)]
			graph_erros.write(f'{cluster} {' '.join(vec)}\n')
		# with open(output_modelos,'a') as output:
		# 	vet=['-1000' for i in range(21)]	
		# 	output.write(f'{cluster} {' '.join(vet)}\n')
		return
	else:
		#call(f'mkdir WHL_gr/{tipo}/{cluster}',shell=True)
		path_save=f'WHL_sky/{cluster}'
		print(cluster)
		os.makedirs(path_save,exist_ok=True)
		#####################################################################################################################	
		temp=[[] for i in range(25)]
		iso_table=open(f'WHL_gr/{cluster}/iso_table_gr.dat','r')
		for item in iso_table.readlines():
			if len(item.split()) == 5:
				extval_ellip=float(item.split()[0])
				extval_pa=float(item.split()[1])
				maxrad=float(item.split()[2])
				sigmaskyg=float(item.split()[3])
				sigmaskyr=float(item.split()[4])
			else:
				for i in range(25):
					temp[i].append(float(item.split()[i]))			
		vec_dirty=[]
		for i in range(25):
			vec_dirty.append(np.asarray(temp[i]))
			
		ext_cut=ma.masked_where(vec_dirty[3] == vec_dirty[3][-1],vec_dirty[3])
		ext_slice=ma.clump_masked(ext_cut)
		for piece in ext_slice:
			if piece.stop == len(vec_dirty[3]):
				ext_cleaner=piece.start
		#
		vec=[]
		for conj in vec_dirty:
			vec.append(conj[vec_dirty[2]<vec_dirty[2][ext_cleaner]])
		x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err,intens_free_g,int_free_err_g, intens_free_r,int_free_err_r,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g=vec

		#
		mag_iso = 22.5-2.5*np.log10(intens) + 2.5*(log10(0.1569166))
		mag_err = 2*2.5*np.log10(np.exp(1.))*int_err/intens
		#
		inf_cut=(ellip_err!=0) & (pa_err!=0)
		epopt, epcov = scp.curve_fit(linfunc,np.log10(sma[inf_cut]), eps[inf_cut], sigma=ellip_err[inf_cut])
		ellipgrad=epopt[0]
		ppopt, pcov = scp.curve_fit(linfunc,np.log10(sma[inf_cut]), pa[inf_cut], sigma=pa_err[inf_cut])
		pagrad=ppopt[0]
		# print('Ellipticity gradient =',format(epopt[0], '.3E'))
		# print('PA gradient =',format(ppopt[0], '.3E'))

		if cluster in ok:
			vec=info[ok.index(cluster)]
			
			spopt=np.abs([float(vec[7]),float(vec[8]),float(vec[9])])#7,8,9]]
			sepopt=np.abs([float(vec[10]),float(vec[11]),float(vec[12]),float(vec[13]),float(vec[14])])#10,11,12,13,14
			sspopt=np.abs([float(vec[15]),float(vec[16]),float(vec[17]), float(vec[18]),float(vec[19]),float(vec[20])])#15,16,17,18,19,20
			
			chirat_1,chirat_2,chisq_s,chisq_se,chisq_ss= float(vec[2]),float(vec[3]),float(vec[4]),float(vec[5]),float(vec[6])

		elif cluster not in ok:
			vec=modelagem(sma,mag_iso,mag_err/2)
			chirat_1=vec[0]
			chirat_2=vec[1]
			chisq_s=vec[2]
			chisq_se=vec[3]
			chisq_ss=vec[4]
			spopt=np.abs(vec[5])
			sepopt=np.abs(vec[6])
			sspopt=np.abs(vec[7])

			with open(output_modelos,'a') as output:
				output.write('%s %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E  \n'%(cluster,ellipgrad,pagrad,chirat_1,chirat_2,chisq_s,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0], sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],sspopt[2],sspopt[3],sspopt[4],sspopt[5]))

		'''
		print('Sersic fit results:\n    Ie =',format(abs(spopt[0]), '.3E'),'\n    Re =',format(spopt[1], '.3E'),'\n    n =',format(abs(spopt[2]), '.3E'))
		print('S+E fit results:\n    Ie =',format(abs(sepopt[0]), '.3E'),'\n    Re =',format(abs(sepopt[1]), '.3E'),'\n    n =',format(abs(sepopt[2]), '.3E'),'\n    I0 =',format(abs(sepopt[3]), '.3E'),'\n    Rd =',format(abs(sepopt[4]), '.3E'))
		print('S+S fit results:\n    Ie =',format(abs(sspopt[0]), '.3E'),'\n    Re =',format(abs(sspopt[1]), '.3E'),'\n    n =',format(abs(sspopt[2]), '.3E'),'\n    I0 =',format(abs(sspopt[3]), '.3E'),'\n    Rd =',format(abs(sspopt[4]), '.3E'),'\n    nd =',format(abs(sspopt[5]), '.3E'))
		'''
		
		###############
		#PONTO DE SEPARAÇÃO DE MODELOS S+E
		#################
		mod_which = model_sep(sepopt,sspopt,sma,a3,b3,a4,b4,a3_err,b3_err,a4_err,b4_err,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g,sigmaskyg)
		flux_parts = modelflux(sepopt,sspopt)
		######################################################################################################
		#RFF CIRCULAR
		############# raio linear, 2pi para cada flux
		
		flux_resid_circ=np.trapz(np.multiply(sma,np.abs(np.power(10,-mag_iso)-np.power(10,-sersic(sma,*spopt)))),x=np.power(sma,0.25))
		
		flux_obs_circ=np.trapz(np.multiply(sma,np.abs(np.power(10,-mag_iso))),x=np.power(sma,0.25))
		
		rff_lin_circ=np.divide(flux_resid_circ,flux_obs_circ)

		######
		#RFF ELIPSE 
		###########
		
		flux_resid_elip=np.trapz(np.multiply(sma*(sma*np.sqrt(1-np.power(eps,2))),np.abs(np.power(10,-mag_iso)-np.power(10,-sersic(sma,*spopt)))),x=np.power(sma,0.25))
		
		flux_obs_elip=np.trapz(np.multiply(sma*(sma*np.sqrt(1-np.power(eps,2))),np.abs(np.power(10,-mag_iso))),x=np.power(sma,0.25))
		
		rff_lin_elip=np.divide(flux_resid_elip,flux_obs_elip)	

		# PLOT DE PARAMETROS GERAIS RAIOX(MAGNITUDE,ELIPTICIDADE,PA)
		rawplots(cluster,sma,mag_iso,mag_err,eps,ellip_err,pa,pa_err)
		###

		# FOURIER COEFFICIENTS
		slowvec,vec_med,sma_a3,sma_a4,sma_b3,sma_b4,a3,a4,b3,b4,a3_err,a4_err,b3_err,b4_err,vec_erro=coefF(cluster,sma,a3,a3_err,a4,a4_err,b3,b3_err,b4,b4_err)
		###
		
		# PLOT DO PERFIL DE SERSIC E SERIC + EXPONENCIAL RESPECTIVAMENTE	
		modelplots(cluster,sma,mag_iso,mag_err,spopt,sepopt,sspopt,chisq_s,chisq_se,chisq_ss)
		###

		#TESTE DE COR G-R
		smagr_free,test_gr_free,test_gr_err_free,smagr_fix,test_gr_fix,test_gr_err_fix,free_popt,fix_popt,corvec,cor_erro=testegr(cluster,sma,sigmaskyg, sigmaskyr,intens_free_g,int_free_err_g,intens_free_r,int_free_err_r, intens_fix_g,int_fix_err_g,intens_fix_r,int_fix_err_r)
		###
		
		#ANALISE GERAL
		analise(cluster,sma,mag_iso,mag_err,eps,ellip_err,pa,pa_err,sma_a3,a3,a3_err,sma_a4,a4,a4_err,sma_b3,b3,b3_err,sma_b4,b4,b4_err,slowvec,smagr_free,test_gr_free,test_gr_err_free,smagr_fix,test_gr_fix,test_gr_err_fix,free_popt,fix_popt)
		###
			
		outvec=[cluster,ellipgrad,pagrad,chirat_1,chisq_s,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0], sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],sspopt[2],sspopt[3],sspopt[4],sspopt[5]]
		outvec_error=[cluster,str(epcov[0][0]),str(pcov[0][0])]
		#mod_which=' '.join(list(map(str,mod_which)))
		
		print(cluster,n_controle,'agora foi')

		with open(output_geral,'a') as graph_data:
			graph_data.write(f'{' '.join(map(str,outvec))} {' '.join(map(str,vec_med))} {' '.join(map(str,corvec))} {str(rff_lin_circ)} {str(rff_lin_elip)} {' '.join(map(str,flux_parts))} {' '.join(map(str,mod_which))} \n')
		with open(output_erros,'a') as graph_erros:
			graph_erros.write(f"{' '.join(map(str, outvec_error))} {' '.join(map(str, vec_erro))} {' '.join(map(str, cor_erro))}\n")
		return
def run_bcgfigs(args):
	cluster,n_controle,ok,okk,info,output_geral,output_modelos,output_erros = args
	return bcgfigs(cluster,n_controle,ok,okk,info,output_geral,output_modelos,output_erros)
if __name__ == "__main__":

	output_geral='graph_stats_WHL_sky_v2.dat'
	output_modelos='iso_geral_values_coef_WHL_sky.dat'
	output_erros='graph_errors_WHL_sky.dat'

	outg=open(output_geral,'a')
	outm=open(output_modelos,'a')
	oute=open(output_erros,'a')
	file_list=['a3','a4','b3','b4','analise','cor_fix','cor_free','cor_log_fix','cor_log_free','diskness','mag_e_pa','rff_linear','sersic_fits']
	for file in file_list:
		os.makedirs(f'WHL_sky/{file}',exist_ok=True)
	ok=[]
	info=[]
	with open('iso_geral_values_coef_WHL_sky.dat','r+') as output:
		for item in output.readlines():
			linha=item.split()
			ok.append(linha[0])
			info.append(linha[1:])
	okk=[]
	infoall=[]
	with open('graph_stats_WHL_sky_v2.dat','r+') as graph_data:
		for bcg in graph_data.readlines():
			linha=bcg.split()
			okk.append(linha[0])
			infoall.append(linha[1:])

	galaxias=[]
	with open('checkiso_WHL.dat','r') as inp1:
		ninp1=inp1.readlines()
		for bcg in ninp1:
			ll1=bcg.split()
			galaxias.append(ll1[0])

	# bcgfigs('000002',1,ok,okk,info,output_geral,output_modelos,output_erros)
	# with mp.Pool(processes=19) as pool:
	# 	chunksize=1
	# 	pool.starmap(bcgfigs,[(obj,galaxias.index(obj),ok,okk,info,output_geral,output_modelos) for obj in galaxias],chunksize=chunksize)

	new_galaxias=[item for item in galaxias if item not in okk]
	obs_list = [(cl,galaxias.index(cl),ok,okk,info,output_geral,output_modelos,output_erros) for cl in new_galaxias]

	with mp.Pool(processes=17) as pool:
		chunksize=1
		for _ in pool.imap_unordered(run_bcgfigs, obs_list,chunksize=chunksize):
			pass
