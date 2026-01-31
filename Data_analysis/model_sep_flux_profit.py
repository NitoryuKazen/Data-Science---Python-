from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp
import numpy as np
from subprocess import call
import scipy.special as scs
import scipy.optimize as scp 
import scipy.integrate as sci
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
############################################################################################

def sersic(x,ie,re,n):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.)))+22.5
	return i
def sersicexp(x,ie,re,n,i0,rd):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	b1=2.*abs(1.)-1./3+4./405/abs(1.)+46./25515/abs(1.)**2+131./1148175/abs(1.)**3-2194697./30690717750/abs(1.)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))+abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
def doublesersic(x,ie,re,n,i0,rd,nd):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	bnd=2.*abs(nd)-1./3+4./405/abs(nd)+46./25515/abs(nd)**2+131./1148175/abs(nd)**3-2194697./30690717750/abs(nd)**4
	s1=abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))
	s2=abs(i0)*np.exp(-bnd*((np.divide(x,abs(rd)))**(1./abs(nd))-1.))
	i=-2.5*np.log10(s1+s2)+22.5
	return i
def envel(x,i0,rd):
	i=-2.5*np.log10(abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
def total_flux(ie,re,n):
	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	fac=2*n-1
	if fac > 0:
		i = ie*2*np.pi*np.power(re,2)*n*(np.divide(np.exp(bn),np.power(bn,2*n)))*scs.factorial(fac)
	elif fac < 0:
		nfac=fac+1
		i = ie*2*np.pi*np.power(re,2)*n*(np.divide(np.exp(bn),np.power(bn,2*n)))*scs.gamma(nfac)
	return i

def musersic(r,re,n,mtot):
	bn = 2.*n-1/3.+4./(405.*n)+46./(25515*n**2)+131./(1148175*n**3)-2194697./(30690717750*n**4)
	mue = mtot + 5*np.log10(re) + 2.5*np.log10(2*pi*n*np.exp(bn)*scs.gamma(2*n)/np.power(bn,2*n))
	return mue+2.5*bn*(np.power(r/re,1./n)-1.)/np.log(10.)
def mudouble(r,re,n,mtot,re2,n2,mtot2):
	bn = 2.*n-1/3.+4./(405.*n)+46./(25515*n**2)+131./(1148175*n**3)-2194697./(30690717750*n**4)
	mue = mtot + 5*np.log10(re) + 2.5*np.log10(2*pi*n*np.exp(bn)*scs.gamma(2*n)/np.power(bn,2*n))
	bn2 = 2*n2-1/3.+4./(405.*n2)+46./(25515*n2**2)+131./(1148175*n2**3)-2194697./(30690717750*n2**4)
	mue2 = mtot2 + 5*np.log10(re2) + 2.5*np.log10(2*pi*n2*np.exp(bn2)*scs.gamma(2*n2)/np.power(bn2,2*n2))
	return -2.5*np.log10(np.power(10,-0.4*(mue+2.5*bn*(np.power(r/re,1./n)-1.)/np.log(10.)))+np.power(10,-0.4*(mue2+2.5*bn2*(np.power(r/re2,1./n2)-1.)/np.log(10.))))
def muonly(r,re,n,mtot):
	bn = 2.*n-1/3.+4./(405.*n)+46./(25515*n**2)+131./(1148175*n**3)-2194697./(30690717750*n**4)
	mue = mtot + 5*np.log10(re) + 2.5*np.log10(2*pi*n*np.exp(bn)*scs.gamma(2*n)/np.power(bn,2*n))
	i= -2.5*np.log10(np.power(10,-0.4*(mue+2.5*bn*(np.power(r/re,1./n)-1.)/np.log(10.))))
	return i

def mulinear(r,re,n,mtot):
	bn = 2.*n-1/3.+4./(405.*n)+46./(25515*n**2)+131./(1148175*n**3)-2194697./(30690717750*n**4)
	mue = mtot + 5*np.log10(re) + 2.5*np.log10(2*pi*n*np.exp(bn)*scs.gamma(2*n)/np.power(bn,2*n))
	i= np.power(10,-0.4*(mue+2.5*bn*(np.power(r/re,1./n)-1.)/np.log(10.)))
	return i

def btcorrection(cluster,r_kron,vec_double_model):
	
	x0=lambda r:mulinear(r,*vec_double_model[:3])*2*np.pi
	x1=lambda r:mulinear(r,*vec_double_model[3:])*2*np.pi
	
	flux_c1_inf=sci.quad(x0,0,np.inf)
	flux_c2_inf=sci.quad(x1,0,np.inf)

	flux_c1_kron=sci.quad(x0,0,r_kron)
	flux_c2_kron=sci.quad(x1,0,r_kron)
	
	corr_1=flux_c1_kron[0]/flux_c1_inf[0]
	corr_2=flux_c2_kron[0]/flux_c2_inf[0]
	vec=[cluster,corr_1,corr_2]
	return vec

###################################################################################
def model_sep(sma,vec_se,vec_ss):

	vec_diff_se=muonly(sma,*vec_se[:3])-muonly(sma,*vec_se[3:])#
	
	xx_se=ma.masked_where(muonly(sma,*vec_se[:3])-muonly(sma,*vec_se[3:]) < 0,vec_diff_se)
	
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
		mod_which_se='mod_part_se'
		if bojo_slice != []:
			func_intern_se='only_sersic'
			se_slices=[bojo_slice[0].start,bojo_slice[0].stop,-1,-1]
		if env_slice != []:
			func_intern_se='only_exp'
			se_slices=[-1,-1,env_slice[0].start,env_slice[0].stop]

	#######################################################################################
	vec_diff_ss=muonly(sma,*vec_ss[:3])-muonly(sma,*vec_ss[3:])#
	
	xx_ss=ma.masked_where(muonly(sma,*vec_ss[:3])-muonly(sma,*vec_ss[3:]) < 0,vec_diff_ss)
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
	return mod_which_se,func_intern_se,mod_which_ss,func_intern_ss
###################################################################################
def modelplots(cluster,sma,vec_s_galfit,vec_se_galfit,vec_ss_galfit,vec_s_profit,vec_se_profit,vec_ss_profit,chisq_s_galfit,chisq_se_galfit,chisq_ss_galfit,best_model):
	if best_model == 'S':
		model_s='S (best) profit'
		model_se='S+E profit'
		model_ss='S+S profit'
	elif best_model == 'S+E':
		model_s='S profit'
		model_se='S+E (best) profit'
		model_ss='S+S profit'
	elif best_model == 'S+S':
		model_s='S profit'
		model_se='S+E profit'
		model_ss='S+S (best) profit'

	if np.sum(vec_se_galfit) == 1.0 or np.sum(vec_ss_galfit) == 0.0:
		check_finite=np.isfinite([musersic(sma,*vec_s_galfit)[0],musersic(sma,*vec_s_profit)[0],mudouble(sma,*vec_se_profit)[0],mudouble(sma,*vec_ss_profit)[0]])
		ymin=min(musersic(sma,*vec_s_galfit)[0][check_finite[0]],musersic(sma,*vec_s_profit)[0][check_finite[1]],mudouble(sma,*vec_se_profit)[0][check_finite[2]],mudouble(sma,*vec_ss_profit)[0][check_finite[3]])
		ymax=max(musersic(sma,*vec_s_galfit)[-1][check_finite[0]],musersic(sma,*vec_s_profit)[-1][check_finite[1]],mudouble(sma,*vec_se_profit)[-1][check_finite[2]],mudouble(sma,*vec_ss_profit)[-1][check_finite[3]])
	else:
		print(vec_s_galfit,vec_se_galfit,vec_ss_galfit,vec_s_profit,vec_se_profit,vec_ss_profit)
		check_finite=np.isfinite([musersic(sma,*vec_s_galfit)[0],musersic(sma,*vec_s_profit)[0],mudouble(sma,*vec_se_galfit)[0],mudouble(sma,*vec_se_profit)[0],mudouble(sma,*vec_ss_profit)[0],mudouble(sma,*vec_ss_galfit)[0]])
		ymin=min(musersic(sma,*vec_s_galfit)[0][check_finite[0]],musersic(sma,*vec_s_profit)[0][check_finite[1]],mudouble(sma,*vec_se_galfit)[0][check_finite[2]],mudouble(sma,*vec_se_profit)[0][check_finite[3]],mudouble(sma,*vec_ss_profit)[0][check_finite[4]],mudouble(sma,*vec_ss_galfit)[0][check_finite[5]])
		ymax=max(musersic(sma,*vec_s_galfit)[-1][check_finite[0]],musersic(sma,*vec_s_profit)[-1][check_finite[1]],mudouble(sma,*vec_se_galfit)[-1][check_finite[2]],mudouble(sma,*vec_se_profit)[-1][check_finite[3]],mudouble(sma,*vec_ss_profit)[-1][check_finite[4]],mudouble(sma,*vec_ss_galfit)[-1][check_finite[5]])
	y_vec=[ymax + 0.5,ymin-0.5]

	fig,axs=plt.subplots(1,3,figsize=(15, 5))#,sharey=True)
	plt.suptitle(cluster)

	axs[0].plot(np.power(sma,0.25),musersic(sma,*vec_s_galfit),label=r'S GALFIT Fit, $\chi^2_\nu=$%.2f' % chisq_s_galfit ,color='black')
	axs[0].plot(np.power(sma,0.25),musersic(sma,*vec_s_profit),label=model_s ,color='red')
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylim(y_vec)

	if np.sum(vec_se_galfit) == 1.0:
		axs[1].plot([],label=r'S+E GALFIT (NÃO CONVERGIU)',color='black',linewidth=1.5)
	else:
		axs[1].plot(np.power(sma,0.25),mudouble(sma,*vec_se_galfit),label=r'S+E GALFIT, $\chi^2_\nu=$%.2f' % chisq_se_galfit,color='black',linewidth=1.5)
		axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_galfit[:3]),label='S',color='black',ls='dotted',linewidth=1)
		axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_galfit[3:]),label='E',color='black',ls='dashed',linewidth=1)

	axs[1].plot(np.power(sma,0.25),mudouble(sma,*vec_se_profit),label=model_se,color='red',linewidth=1.5)
	axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_profit[:3]),label='S',color='red',ls='dotted',linewidth=1)
	axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_profit[3:]),label='E',color='red',ls='dashed',linewidth=1)

	axs[1].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylim(y_vec)
	
	
	if np.sum(vec_ss_galfit) == 0.0:
		axs[2].plot([],label=r'S+S GALFIT (NÃO CONVERGIU)',color='black',linewidth=1.5)
	else:
		axs[2].plot(np.power(sma,0.25),mudouble(sma,*vec_ss_galfit),label=r'S+S GALFIT, $\chi^2_\nu=$%.2f' % chisq_ss_galfit,color='black',linewidth=1.5)
		axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_galfit[:3]),label='S1',color='black',ls='dotted',linewidth=1)
		axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_galfit[3:]),label='S2',color='black',ls='dashed',linewidth=1)

	axs[2].plot(np.power(sma,0.25),mudouble(sma,*vec_ss_profit),label=model_ss,color='red',linewidth=1.5)
	axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_profit[:3]),label='S1',color='red',ls='dotted',linewidth=1)
	axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_profit[3:]),label='S2',color='red',ls='dashed',linewidth=1)

	axs[2].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[2].set_ylim(y_vec)
	
	fig.legend(loc=('outside upper right'))
	fig.savefig('graficos_1D/'+cluster+'_models_all.png')	
	plt.close(fig)
	#########################################################################################
	#GALFIT

	fig,axs=plt.subplots(1,3,figsize=(15, 5))#,sharey=True)
	plt.suptitle(cluster+' GALFIT')

	axs[0].plot(np.power(sma,0.25),musersic(sma,*vec_s_galfit),label=r'S GALFIT Fit, $\chi^2_\nu=$%.2f' % chisq_s_galfit ,color='black')
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylim(y_vec)

	if np.sum(vec_se_galfit) == 1.0:
		axs[1].plot([],label=r'S+E GALFIT (NÃO CONVERGIU)',color='black',linewidth=1.5)
	else:
		axs[1].plot(np.power(sma,0.25),mudouble(sma,*vec_se_galfit),label=r'S+E GALFIT, $\chi^2_\nu=$%.2f' % chisq_se_galfit,color='black',linewidth=1.5)
		axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_galfit[:3]),label='S',color='black',ls='dotted',linewidth=1)
		axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_galfit[3:]),label='E',color='black',ls='dashed',linewidth=1)

	axs[1].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylim(y_vec)
	
	if np.sum(vec_ss_galfit) == 0.0:
		axs[2].plot([],label=r'S+S GALFIT (NÃO CONVERGIU)',color='black',linewidth=1.5)
	else:
		axs[2].plot(np.power(sma,0.25),mudouble(sma,*vec_ss_galfit),label=r'S+S GALFIT, $\chi^2_\nu=$%.2f' % chisq_ss_galfit,color='black',linewidth=1.5)
		axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_galfit[:3]),label='S1',color='black',ls='dotted',linewidth=1)
		axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_galfit[3:]),label='S2',color='black',ls='dashed',linewidth=1)

	axs[2].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[2].set_ylim(y_vec)
	
	fig.legend(loc=('outside upper right'))
	fig.savefig('graficos_1D/'+cluster+'_models_galfit.png')	
	plt.close(fig)
	#########################################################################################
	#PROFIT

	fig,axs=plt.subplots(1,3,figsize=(15, 5))#,sharey=True)
	plt.suptitle(cluster+ ' PROFIT')

	axs[0].plot(np.power(sma,0.25),musersic(sma,*vec_s_profit),label=model_s ,color='red')
	axs[0].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[0].set_ylim(y_vec)


	axs[1].plot(np.power(sma,0.25),mudouble(sma,*vec_se_profit),label=model_se,color='red',linewidth=1.5)
	axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_profit[:3]),label='S',color='red',ls='dotted',linewidth=1)
	axs[1].plot(np.power(sma,0.25),muonly(sma,*vec_se_profit[3:]),label='E',color='red',ls='dashed',linewidth=1)

	axs[1].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[1].set_ylim(y_vec)
	
	
	axs[2].plot(np.power(sma,0.25),mudouble(sma,*vec_ss_profit),label=model_ss,color='red',linewidth=1.5)
	axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_profit[:3]),label='S1',color='red',ls='dotted',linewidth=1)
	axs[2].plot(np.power(sma,0.25),muonly(sma,*vec_ss_profit[3:]),label='S2',color='red',ls='dashed',linewidth=1)

	axs[2].set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	axs[2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[2].set_ylim(y_vec)
	
	fig.legend(loc=('outside upper right'))
	fig.savefig('graficos_1D/'+cluster+'_models_profit.png')	
	plt.close(fig)

	return
###################################################################################
def bcgfigs(cluster,xc,yc,re_s_galfit,mag_s_galfit,n_s_galfit,reb_se_galfit,magb_se_galfit,nb_se_galfit,red_se_galfit,magd_se_galfit,re1_ss_galfit,mag1_ss_galfit,n1_ss_galfit,re2_ss_galfit,mag2_ss_galfit,n2_ss_galfit,re_s_profit,mag_s_profit,n_s_profit,reb_se_profit,magb_se_profit,nb_se_profit,red_se_profit,magd_se_profit,re1_ss_profit,mag1_ss_profit,n1_ss_profit,re2_ss_profit,mag2_ss_profit,n2_ss_profit,chisq_s_galfit,chisq_se_galfit,chisq_ss_galfit,best_model,r_kron):
	sma = np.arange(1,max(xc,yc))
	vec_ss_profit=[re1_ss_profit,n1_ss_profit,mag1_ss_profit,re2_ss_profit,n2_ss_profit,mag2_ss_profit]

	corr_se_profit=btcorrection(r_kron,vec_se_profit)

	return
#################################################################################

sample='L07'

header_data=np.loadtxt(f'profit_observation.header',dtype=str)
data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky.dat',dtype=str).T
data_obs=dict(zip(header_data,data_obs_temp))

header_kron=['cluster','kron_r']
data_kron_temp=np.loadtxt(f'{sample}_files/kron_radius_L07.dat',dtype=str).T
data_kron=dict(zip(header_kron,data_kron_temp))

cluster=data_obs['cluster']

xc,yc=data_obs['XCEN'].astype(float),data_obs['YCEN'].astype(float)
mag1,mag2=data_obs['MAG_1'].astype(float),data_obs['MAG_2'].astype(float)
re1,re2=data_obs['RE_1'].astype(float),data_obs['RE_2'].astype(float)
n1,n2=data_obs['NSER_1'].astype(float),data_obs['NSER_2'].astype(float)

r_kron=data_kron['kron_r'].astype(float)

r_max=np.asarray([xc,yc]).T
# sma=[np.arange(1,np.max(raio)) for raio in r_max]
sma=np.arange(1,np.max(r_max))
vec_ss=np.asarray([re1,n1,mag1,re2,n2,mag2])

corr_bt_ss=[]

vec1=np.asarray([re1,n1,mag1]).T
vec2=np.asarray([re2,n2,mag2]).T

# musersic(sma,*vec1[0])
for i,item in enumerate(r_kron):
	corr_bt_ss.append(btcorrection(cluster[i],item,vec_ss.T[i]))

corr_bt_ss=np.asarray(corr_bt_ss)
np.savetxt(f'{sample}_files/{sample}_corr_bt.dat',corr_bt_ss,fmt='%s',newline='\n')

###########################################################################################