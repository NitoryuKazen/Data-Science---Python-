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
import scipy.stats as sst
import scipy.signal as sss
import photutils.isophote as phi
import photutils.aperture as php
import numpy.ma as ma
from astropy.io import fits
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings("ignore")
#############################

def linfunc(x,a,b):
	return a*x+b

def skyradfunc(x,a,b,c):
	return a*np.exp(-x/b)+c

def calc_sky(image,mask,maskb,xcenter,ycenter,cluster):
	vsky=[]
	dsky=[]
	for j in range(image.shape[0]):
		for i in range(image.shape[1]):
			if mask[j,i]==0 and maskb[j,i]==0:
				vsky.append(image[j,i])
				dsky.append(((j-ycenter)**2+(i-xcenter)**2)**0.5)
	if len(dsky) <= 3:
		skyvalue=+0
	else:
		popt,pcov=scp.curve_fit(skyradfunc,dsky,vsky,p0=[200,100,100])
		skyvalue =+ skyradfunc(np.max(dsky),*popt)
	return skyvalue
	
def devauc(x,ie,re):
	n=4.
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	i=-2.5*np.log10(ie*np.exp(-bn*((np.divide(x,re))**(1./n)-1.)))+22.5
	return i

def sersic(x,ie,re,n):
	n=abs(n)
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	i=-2.5*np.log10(ie*np.exp(-bn*((np.divide(x,re))**(1./n)-1.)))+22.5
	return i

def sersictest(x,ie,re,n):
	n=abs(n)
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	i=-2.5*log10(ie*exp(-(2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4)*((x/re)**(1./n)-1.)))+22.5
	return i


def sersicexp(x,ie,re,n,i0,rd):
	n=abs(n)
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	i0=abs(i0)
	rd=abs(rd)
	i=-2.5*np.log10(ie*np.exp(-bn*((np.divide(x,re))**(1./n)-1.))+i0*np.exp(-np.divide(x,rd)))+22.5
	return i
	
def doublesersic(x,ie,re,n,i0,rd,nd):
	n=abs(n)
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	bnd=2.*nd-1./3+4./405/nd+46./25515/nd**2+131./1148175/nd**3-2194697./30690717750/nd**4
	i0=abs(i0)
	rd=abs(rd)
	nd=abs(nd)
	s1=ie*np.exp(-bn*((np.divide(x,re))**(1./n)-1.))
	s2=i0*np.exp(-bnd*((np.divide(x,rd))**(1./nd)-1.))
	i=-2.5*np.log10(s1+s2)+22.5
	return i

def envel(x,i0,rd):
	i0=abs(i0)
	rd=abs(rd)
	i=-2.5*np.log10(i0*np.exp(-np.divide(x,rd)))+22.5
	return i


# Now create a bootstrap confidence interval around the a LOWESS fit

def bcgfigs(cluster,bigtemp,tipo):
	print(cluster,tipo)
	call('mkdir L07_coef_gain/'+tipo+'/'+cluster,shell=True)
	#####################################################################################################################	
	temp=[[] for i in range(25)]
	iso_table=open('L07_coef_gain/'+cluster+'/iso_table_gr.dat','r')
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
	vec=[]
	for i in range(25):
		vec.append(np.asarray(temp[i]))
	x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err,intens_free_g,int_free_err_g, intens_free_r,int_free_err_r,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g=vec
	#
	mag_iso = 22.5-2.5*np.log10(intens) + 2.5*(log10(0.1569166))
	mag_err = 2*2.5*np.log10(np.exp(1.))*int_err/intens
	#
	# PLOT DE PARAMETROS GERAIS RAIOX(MAGNITUDE,ELIPTICIDADE,PA)

	plt.figure(figsize=(9, 3))
	plt.subplots_adjust(hspace=0.35, wspace=0.35)
	plt.subplot(1, 3, 1)
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,fmt='o',markersize=3,color='k',ecolor='0.8')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.,np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.])
	plt.gca().invert_yaxis()

	plt.subplot(1, 3, 2)
	plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=3, color='k',ecolor='0.8')
	plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$e$')
	
	plt.subplot(1, 3, 3)
	plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=3, color='k',ecolor='0.8')
	plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
	plt.xlabel(r'$R$ (arcsec)')
	plt.ylabel('PA (deg)')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_iso_info.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/mag_e_pa/'+cluster+'_iso_info.png')	
	plt.close()	

	######################################################################################################
	# FOURIER COEFFICIENTS

	a3_iqr,a4_iqr,b3_iqr,b4_iqr=iqrvec=[(sst.iqr(a3),np.percentile(a3,[25,75])),(sst.iqr(a4),np.percentile(a4,[25,75])),(sst.iqr(b3),np.percentile(b3,[25,75])),(sst.iqr(b4),np.percentile(b4,[25,75]))]
	#
	a3_lim,a4_lim,b3_lim,b4_lim=(a3_iqr[1][0]-1.5*a3_iqr[0] < a3) & (a3 < a3_iqr[1][1]+1.5*a3_iqr[0]),(a4_iqr[1][0]-1.5*a4_iqr[0] < a4) & (a4 < a4_iqr[1][1]+1.5*a4_iqr[0]),(b3_iqr[1][0]-1.5*b3_iqr[0] < b3) & (b3 < b3_iqr[1][1]+1.5*b3_iqr[0]),(b4_iqr[1][0]-1.5*b4_iqr[0] < b4) & (b4 < b4_iqr[1][1]+1.5*b4_iqr[0])

	a3,a4,b3,b4=a3[a3_lim],a4[a4_lim],b3[b3_lim],b4[b4_lim]
	
	a3_err,a4_err,b3_err,b4_err=a3_err[a3_lim],a4_err[a4_lim],b3_err[b3_lim],b4_err[b4_lim]
		
	sma_a3,sma_a4,sma_b3,sma_b4=sma[a3_lim],sma[a4_lim],sma[b3_lim],sma[b4_lim]
	
	vec_med=[]
	breaktest=[]
	fourier_all=[[],[],[],[],[]]
	raio_all=[[],[],[],[],[]]
	slowvec=[]
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

		plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_'+savename+'.png')
		plt.savefig('L07_coef_gain/'+tipo+'/'+savename+'/'+cluster+'.png')
		plt.close()
		
		plusflor=ma.masked_where(np.floor(slow[:,1])!=0.0,slow[:,1])
		pluslice=np.ma.notmasked_contiguous(plusflor)

		minusflor=ma.masked_where(np.floor(slow[:,1])==0.0,slow[:,1])
		minuslice=np.ma.notmasked_contiguous(minusflor)
		
		slope_slow=np.polyfit(slow[:,0],slow[:,1],1)				
		breaktest.append((len(minuslice),len(pluslice)))
		vec_med.append((med_1,med_slow,slope_slow[0]))
		
#		for i in range(len(slow[:,0])):
#			raio_all[j].append(slow[:,0][i])
#			fourier_all[j].append(slow[:,1][i])
	#	print(raio_all)		
		#print(sma/sma[-1],sma)
		
		'''
		tentar fazer um teste se o slice é do tamanho do len slow, se for ele tem ue falar que não tem teste, 
		pode ser um np.nan caso não tenha, se tiver, preciso saber o indice e o ponto raio de corte, 
		o corte vai ter que ocorrer em um valor maior do que 25/75 por cento da amostra (usar o iqr) seria 
		interessante então ir por esse caminho
							
		6 - averiguar a possibilidade de analisar de maneiras diferentes de acordo com a literatura
		'''
##############################################################################################################################
	#PLOT DO PERFIL DE SERSIC E SERIC + EXPONENCIAL RESPECTIVAMENTE
	
	fig = plt.subplots(figsize=(15, 5),sharey=True)
	plt.subplots_adjust(hspace=0.35, wspace=0.35)

	plt.subplot(1, 3, 1)
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,label='Obs. profile',fmt='o',markersize=4,zorder=0, color='k')
	plt.plot(np.power(sma,0.25),sersic(sma,*bigtemp[0][1:]),label=r'S Fit, $\chi^2_\nu=$%.2f' % bigtemp[0][0] ,color='blue',linewidth=5)
	plt.gca().invert_yaxis()
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylim([np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.])
	plt.legend()
	plt.tight_layout()

	plt.subplot(1, 3, 2)
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,label='Obs. profile',fmt='o',markersize=4,zorder=0, color='k')
	plt.plot(np.power(sma,0.25),sersicexp(sma,*bigtemp[1][1:]),label=r'S+E Fit, $\chi^2_\nu=$%.2f' % bigtemp[1][0],color='orange',linewidth=3)
	plt.plot(np.power(sma,0.25),sersic(sma,*bigtemp[1][1:4]),label='S',color='red',linewidth=2)
	plt.plot(np.power(sma,0.25),envel(sma,*bigtemp[1][4:6]),label='E',color='yellow',linewidth=2)
	plt.gca().invert_yaxis()
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylim([np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.])
	plt.legend()
	plt.tight_layout()

	plt.subplot(1, 3, 3)
	plt.errorbar(np.power(sma,0.25),mag_iso,yerr=mag_err,label='Obs. profile',fmt='o',markersize=4,zorder=0, color='k')
	plt.plot(np.power(sma,0.25),doublesersic(sma,*bigtemp[2][1:]),label=r'S+S Fit, $\chi^2_\nu=$%.2f' % bigtemp[2][0],color='orange',linewidth=3)
	plt.plot(np.power(sma,0.25),sersic(sma,*bigtemp[2][1:4]),label='S_b',color='red',linewidth=2)
	plt.plot(np.power(sma,0.25),sersic(sma,*bigtemp[2][4:7]),label='S_e',color='yellow',linewidth=2)
	plt.gca().invert_yaxis()
	plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylim([np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.])
	plt.legend()
	plt.tight_layout()

	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_models.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/sersic_fits/'+cluster+'_models.png')	
	plt.close()
	
	####################################################################
	#TESTE DE COR G-R
	
	intens_g_free = intens_free_g[np.where(intens_free_g > sigmaskyg)]
	intens_r_free = intens_free_r[np.where(intens_free_g > sigmaskyg)]

	intens_g_fix = intens_fix_g[np.where(intens_fix_g > sigmaskyg)]
	intens_r_fix = intens_fix_r[np.where(intens_fix_g > sigmaskyg)]

	int_err_g_free = int_free_err_g[np.where(intens_free_g > sigmaskyg)]
	int_err_r_free = int_free_err_r[np.where(intens_free_g > sigmaskyg)]

	int_err_g_fix = int_fix_err_g[np.where(intens_fix_g > sigmaskyg)]
	int_err_r_fix = int_fix_err_r[np.where(intens_fix_g > sigmaskyg)]

	mag_g_free = 22.5 -2.5*np.log10(intens_g_free) + 2.5*(log10(0.1569166))
	mag_r_free = 22.5 -2.5*np.log10(intens_r_free) + 2.5*(log10(0.1569166))

	mag_g_fix = 22.5 -2.5*np.log10(intens_g_fix) + 2.5*(log10(0.1569166))
	mag_r_fix = 22.5 -2.5*np.log10(intens_r_fix) + 2.5*(log10(0.1569166))

	mag_err_g_free = 2*2.5*np.log10(np.exp(1.))*int_err_g_free/intens_g_free 
	mag_err_r_free = 2*2.5*np.log10(np.exp(1.))*int_err_r_free/intens_r_free 

	mag_err_g_fix = 2*2.5*np.log10(np.exp(1.))*int_err_g_fix/intens_g_fix 
	mag_err_r_fix = 2*2.5*np.log10(np.exp(1.))*int_err_r_fix/intens_r_fix 

	smagr_free_0=sma[np.where(intens_free_g>sigmaskyg)]
	smagr_fix_0=sma[np.where(intens_fix_g>sigmaskyg)]
	
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

	polyfit_fix_log_1=np.polyfit(np.log10(smagr_fix),test_gr_fix,1,w=1/np.power(test_gr_err_fix,2))
	polyfit_free_log_1=np.polyfit(np.log10(smagr_free),test_gr_free,1,w=1/np.power(test_gr_err_free,2))

	polyfit_fix_log_2=np.polyfit(np.log10(smagr_fix),test_gr_fix,2,w=1/np.power(test_gr_err_fix,2))
	polyfit_free_log_2=np.polyfit(np.log10(smagr_free),test_gr_free,2,w=1/np.power(test_gr_err_free,2))
	
	pfixlog1=np.poly1d(polyfit_fix_log_1)
	pfreelog1=np.poly1d(polyfit_free_log_1)

	pfixlog2=np.poly1d(polyfit_fix_log_2)
	pfreelog2=np.poly1d(polyfit_free_log_2)

	polyfit_fix_1=np.polyfit(np.power(smagr_fix,0.25),test_gr_fix,1,w=1/np.power(test_gr_err_fix,2))
	polyfit_free_1=np.polyfit(np.power(smagr_free,0.25),test_gr_free,1,w=1/np.power(test_gr_err_free,2))

	polyfit_fix_2=np.polyfit(np.power(smagr_fix,0.25),test_gr_fix,2,w=1/np.power(test_gr_err_fix,2))
	polyfit_free_2=np.polyfit(np.power(smagr_free,0.25),test_gr_free,2,w=1/np.power(test_gr_err_free,2))

	pfix1=np.poly1d(polyfit_fix_1)
	pfree1=np.poly1d(polyfit_free_1)

	pfix2=np.poly1d(polyfit_fix_2)
	pfree2=np.poly1d(polyfit_free_2)

	lowcor_log_fix=lowess(test_gr_fix,np.log10(smagr_fix),frac=0.9)
	lowcor_log_free=lowess(test_gr_free,np.log10(smagr_free),frac=0.9)
	lowcor_fix=lowess(test_gr_fix,np.power(smagr_fix,0.25),frac=0.9)
	lowcor_free=lowess(test_gr_free,np.power(smagr_free,0.25),frac=0.9)

	plowfit_fix_log=np.polyfit(lowcor_log_fix[:,0],lowcor_log_fix[:,1],1)
	plowfit_free_log=np.polyfit(lowcor_log_free[:,0],lowcor_log_free[:,1],1)
	plowfit_fix=np.polyfit(lowcor_fix[:,0],lowcor_fix[:,1],1)
	plowfit_free=np.polyfit(lowcor_free[:,0],lowcor_free[:,1],1)

	plowlogfix=np.poly1d(plowfit_fix_log)
	plowlogfree=np.poly1d(plowfit_free_log)
	plowfix=np.poly1d(plowfit_fix)
	plowfree=np.poly1d(plowfit_free)

	polyfit_vec=[polyfit_fix_log_1,polyfit_fix_log_2,polyfit_fix_1,polyfit_fix_2,polyfit_free_1, polyfit_free_2,polyfit_free_log_1,polyfit_free_log_2]
	
	plowfit_vec=[plowfit_fix_log,plowfit_fix,plowfit_free_log,plowfit_free]

	plt.figure()
	plt.errorbar(np.log10(smagr_free),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
	plt.plot(np.log10(smagr_free),pfreelog1(np.log10(smagr_free)),c='r',label='Linear Fit')
	plt.plot(np.log10(smagr_free),pfreelog2(np.log10(smagr_free)),c='r',linestyle='dashed',label='Quadratic Fit')
	plt.plot(lowcor_log_free[:,0],lowcor_log_free[:,1],c='b',label='Lowess 0.9')
	plt.plot(lowcor_log_free[:,0],plowlogfree(lowcor_log_free[:,0]),label='Lowess Linear fit')
	plt.xlabel(r'$R$ log(arcsec)')
	plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_gr_free_log_polyfit.png')
	plt.savefig('L07_coef_gain/'+tipo+'/cor_log_free/'+cluster+'_gr_free_log_polyfit.png')
	plt.close()
	
	plt.figure()
	plt.errorbar(np.log10(smagr_fix),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
	plt.plot(np.log10(smagr_fix),pfixlog1(np.log10(smagr_fix)),c='r',label='Linear Fit')
	plt.plot(np.log10(smagr_fix),pfixlog2(np.log10(smagr_fix)),c='r',linestyle='dashed',label='Quadratic Fit')
	plt.plot(lowcor_log_fix[:,0],lowcor_log_fix[:,1],c='b',label='Lowess 0.9')
	plt.plot(lowcor_log_fix[:,0],plowlogfix(lowcor_log_fix[:,0]),label='Lowess Linear fit')
	plt.xlabel(r'$R$ log(arcsec)')
	plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_gr_fixx_log_polyfit.png')
	plt.savefig('L07_coef_gain/'+tipo+'/cor_log_fix/'+cluster+'_gr_fixx_log_polyfit.png')
	plt.close()
	#######################################################################################
	#np.power(sma[lim_bojo],0.25)
	
	plt.figure()
	plt.errorbar(np.power(smagr_free,0.25),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
	plt.plot(np.power(smagr_free,0.25),pfree1(np.power(smagr_free,0.25)),c='r',label='Linear Fit')
	plt.plot(np.power(smagr_free,0.25),pfree2(np.power(smagr_free,0.25)),c='r',linestyle='dashed',label='Quadratic Fit')
	plt.plot(lowcor_free[:,0],lowcor_free[:,1],c='b',label='Lowess 0.9')
	plt.plot(lowcor_free[:,0],plowfree(lowcor_free[:,0]),label='Lowess Linear fit')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_gr_free.png')
	plt.savefig('L07_coef_gain/'+tipo+'/cor_free/'+cluster+'_gr_free.png')
	plt.close()
	
	plt.figure()
	plt.errorbar(np.power(smagr_fix,0.25),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
	plt.plot(np.power(smagr_fix,0.25),pfix1(np.power(smagr_fix,0.25)),c='r',label='Linear Fit')
	plt.plot(np.power(smagr_fix,0.25),pfix2(np.power(smagr_fix,0.25)),c='r',linestyle='dashed',label='Quadratic Fit')
	plt.plot(lowcor_fix[:,0],lowcor_fix[:,1],c='b',label='Lowess 0.9')
	plt.plot(lowcor_fix[:,0],plowfix(lowcor_fix[:,0]),label='Lowess Linear fit')
	plt.xlabel(r'$R^{1/4}$ (arcsec)')
	plt.ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	plt.ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
	plt.legend(fontsize='x-small')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_gr_fixx.png')
	plt.savefig('L07_coef_gain/'+tipo+'/cor_fix/'+cluster+'_gr_fixx.png')
	plt.close()
	#call('rm L07_coef_gain/'+cluster+'/test*',shell=True)
		
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
	
	axs[2,1].errorbar(np.power(smagr_free,0.25),test_gr_free, yerr=test_gr_err_free,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[2,1].plot(np.power(smagr_free,0.25),pfree1(np.power(smagr_free,0.25)),c='r')
	axs[2,1].plot(lowcor_free[:,0],lowcor_free[:,1],c='b',label='Lowess 0.9')
	axs[2,1].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	axs[2,1].set_ylim([np.min(test_gr_free)-(np.max(test_gr_free)-np.min(test_gr_free))/10.,np.max(test_gr_free)+(np.max(test_gr_free)-np.min(test_gr_free))/10.])
	axs[2,1].set_xlabel(r'$R^{1/4}$ (arcsec)')

	axs[2,2].errorbar(np.power(smagr_fix,0.25),test_gr_fix, yerr=test_gr_err_fix,fmt='o',markersize=2,color='k',ecolor='0.8')
	axs[2,2].plot(np.power(smagr_fix,0.25),pfix1(np.power(smagr_fix,0.25)),c='r')
	axs[2,2].plot(lowcor_fix[:,0],lowcor_fix[:,1],c='b')
	axs[2,2].set_ylabel(r'$g-r$ (mag arcsec$^{-2}$)')
	axs[2,2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[2,2].set_ylim([np.min(test_gr_fix)-(np.max(test_gr_fix)-np.min(test_gr_fix))/10.,np.max(test_gr_fix)+(np.max(test_gr_fix)-np.min(test_gr_fix))/10.])
	#plt.show()
	plt.savefig('L07_coef_gain/'+tipo+'/analise_'+tipo+'/'+cluster+'.png')
	plt.close()

	
	return vec_med,polyfit_vec,plowfit_vec,breaktest	
graph_data=open('graph_stats.dat','w')

#fourier_lowess=a3_all,a4_all,b3_all,b4_all,disk_all=[[[],[]] for i in range(5)]
with open('/home/andre/Documents/Projetos/isophotest/iso_geral_values_coef_astro_tipo.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('/home/andre/Documents/Projetos/isophotest/iso_geral_values_coef_astro_tipo.dat','r')
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	coef_comp=bcgfigs(ll1[0],[[float(ll1[11]),float(ll1[14]),float(ll1[15]),float(ll1[16])],[float(ll1[12]),float(ll1[17]),float(ll1[18]),float(ll1[19]),float(ll1[20]),float(ll1[21])],[float(ll1[13]),float(ll1[22]),float(ll1[23]),float(ll1[24]),float(ll1[25]),float(ll1[26]),float(ll1[27])]],ll1[-1])

	#x=[ll1[0],float(ll1[11]),float(ll1[14]),float(ll1[15]),float(ll1[16]),float(ll1[12]),float(ll1[17]), float(ll1[18]),float(ll1[19]),float(ll1[20]),float(ll1[21]),float(ll1[13]),float(ll1[22]),float(ll1[23]),float(ll1[24]),float(ll1[25]),float(ll1[26]),float(ll1[27]), int(ll1[30]),coef_comp[0][0],coef_comp[0][1],coef_comp[0][2],coef_comp[0][3],coef_comp[0][4],coef_comp[1][0][0],coef_comp[1][0][1],coef_comp[1][1][0],coef_comp[1][1][1],coef_comp[1][1][2],coef_comp[1][2][0],coef_comp[1][2][1],coef_comp[1][3][0],coef_comp[1][3][1],coef_comp[1][3][2],coef_comp[1][4][0],coef_comp[1][4][1],coef_comp[1][5][0],coef_comp[1][5][1],coef_comp[1][5][2],coef_comp[1][6][0],coef_comp[1][6][1],coef_comp[1][7][0],coef_comp[1][7][1],coef_comp[1][7][2]]
#	print(len(x))
	
	
	graph_data.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %i %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %i %i %i %i %i %i %i %i %i %i \n' %(ll1[0],float(ll1[5]),float(ll1[6]),float(ll1[7]),float(ll1[8]),float(ll1[9]),float(ll1[11]),float(ll1[14]),float(ll1[15]),float(ll1[16]),float(ll1[12]), float(ll1[17]),float(ll1[18]),float(ll1[19]),float(ll1[20]),float(ll1[21]),float(ll1[13]),float(ll1[22]),float(ll1[23]),float(ll1[24]),float(ll1[25]),float(ll1[26]),float(ll1[27]), int(ll1[30]),coef_comp[0][0][0],coef_comp[0][0][1],coef_comp[0][0][2],coef_comp[0][1][0],coef_comp[0][1][1],coef_comp[0][1][2],coef_comp[0][2][0],coef_comp[0][2][1],coef_comp[0][2][2],coef_comp[0][3][0],coef_comp[0][3][1],coef_comp[0][3][2],coef_comp[0][4][0],coef_comp[0][4][1],coef_comp[0][4][2],coef_comp[1][0][0],coef_comp[1][0][1],coef_comp[1][1][0],coef_comp[1][1][1],coef_comp[1][1][2],coef_comp[1][2][0],coef_comp[1][2][1],coef_comp[1][3][0],coef_comp[1][3][1],coef_comp[1][3][2],coef_comp[1][4][0],coef_comp[1][4][1],coef_comp[1][5][0],coef_comp[1][5][1],coef_comp[1][5][2],coef_comp[1][6][0],coef_comp[1][6][1],coef_comp[1][7][0],coef_comp[1][7][1],coef_comp[1][7][2],coef_comp[2][0][0],coef_comp[2][0][1],coef_comp[2][1][0],coef_comp[2][1][1],coef_comp[2][2][0],coef_comp[2][2][1],coef_comp[2][3][0],coef_comp[2][3][1],coef_comp[3][0][0],coef_comp[3][0][1],coef_comp[3][1][0],coef_comp[3][1][1],coef_comp[3][2][0],coef_comp[3][2][1],coef_comp[3][3][0],coef_comp[3][3][1],coef_comp[3][4][0],coef_comp[3][4][1]))

############################################################################################

'''
param_name=[('a3',0),('a4',1),('b3',2),('b4',3),('a4/sma',4)]
grupos=[('eliptica',0),('envelope',1),('assimetrica',2)]
#for tip,idt in grupos:
#	for nome,idn in param_name:
print(a3_all)
plt.figure()
#plt.suptitle(tip+' '+nome,fontsize=10)
plt.axhline(y=0.,xmin=0,xmax=1,c='k')
for dupla in a3_all:
	upy=[]
	lowy=[]
	#print(dupla)#,idt,idn)
	for item in dupla:
		#print(item)			
		for i in range(len(item[0])):
			upy.append(np.max(item[1]))
			lowy.append(np.min(item[1]))
			plt.plot(item[0],item[1],linewidth=1,ls='--',alpha=0.8)
plt.ylim(np.min(lowy)-abs(np.max(upy)-np.min(lowy))/5.,np.max(upy)+abs(np.max(upy)-np.min(lowy))/5.)
#plt.ylabel(nome)
plt.xlabel(r'$R/R_{max}$')
#plt.savefig('L07_coef_gain/'+tipo+'/'+nome+'.png')
plt.show()
plt.close()
'''		

'''

	fraction=[0.9]#0.6,0.7,0.8,0.9]
	linefrac=['--']#,'--','-.',':']
	fraction_cores=np.linspace(0,1,len(fraction))
	coefs=[(a3,a3_err,r'$a_3$',0,'a3',sma_a3)]#,(a4,a4_err,r'$a_4$',1,'a4',sma_a4),(b3,b3_err,r'$b_3$',2,'b3',sma_b3),(b4,b4_err,r'$b_4$',3,'b4',sma_b4),(a4/sma_a4,a4_err/sma_a4,r'$a_4/sma$',4,'diskness',sma_a4)]
	fration_tuple=[[(0.9,1,1)]]#(0.6,0,0),(0.7,0,1),(0.8,1,0),(0.9,1,1)]
	for coef,coef_err,nomey,j,savename,radius in coefs:
		plt.figure()
		plt.suptitle(cluster+' '+savename,fontsize=10)
		med_1=np.average(coef[np.where(coef_err != 0.0)],weights=1/np.power(coef_err[np.where(coef_err != 0.0)],2))
		vec_med.append(med_1)
		plt.errorbar(np.power(radius,0.25),coef,yerr=coef_err,fmt='o',markersize=2,color='k',ecolor='0.8',label='1.0 '+format(vec_med[j],'.3E'))

		for i in range(len(fraction)):
			slow=lowess(coef,np.power(radius,0.25),frac=fraction[i])
			sflor=ma.masked_where(np.floor(slow[:,1])==0.0,slow[:,1])
			tflor=np.ma.notmasked_contiguous(sflor)
			print(sflor[0])
			print(slow[:,1][ttt[0]])			
			tentar fazer um teste se o slice é do tamanho do len slow, se for ele tem ue falar que não tem teste, 
			pode ser um np.nan caso não tenha, se tiver, preciso saber o indice e o ponto raio de corte, 
			o corte vai ter que ocorrer em um valor maior do que 25/75 por cento da amostra (usar o iqr) seria 
			interessante então ir por esse caminho

#			print((np.where(slow[:,1]< 0.),np.where(slow[:,1]> 0.)))
			med_slow=np.average(slow[:,1])
			vec_med.append(med_slow)
			plt.plot(slow[:,0],slow[:,1],linewidth=2,ls=linefrac[i],c=plt.cm.viridis(fraction_cores[i]),label=str(fraction[i])+' '+format(med_slow,'.3E'))
			plt.axhline(y=0.,xmin=0,xmax=1,c='k')
		plt.ylabel(nomey)
		plt.xlabel(r'$R^{1/4}$ (arcsec)')
		plt.legend(title=r'$F-\mu$',fontsize='x-small')
		plt.ylim(np.min(coef)-abs(np.max(coef)-np.min(coef))/5.,np.max(coef)+abs(np.max(coef)-np.min(coef))/5.)

		plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_'+savename+'.png')
		plt.savefig('L07_coef_gain/'+tipo+'/'+savename+'/'+cluster+'.png')
		plt.close()
		


		fig, axs = plt.subplots(2, 2,sharex=True)
		plt.subplots_adjust(hspace=0.35, wspace=0.35)
		plt.suptitle(cluster+' '+savename,fontsize=10)
		
		for i in range(len(fration_tuple)):
			slow=lowess(coef,np.power(radius,0.25),frac=fraction[i])
			axs[fration_tuple[i][1],fration_tuple[i][2]].errorbar(np.power(radius,0.25),coef,yerr=coef_err,fmt='o',markersize=1,color='k',ecolor='0.9',label='1.0')			

			axs[fration_tuple[i][1],fration_tuple[i][2]].plot(slow[:,0],slow[:,1],linewidth=2,c=plt.cm.viridis(fraction_cores[i]),label=str(fraction[i]))
		
			if i == 0 or i == 2:
				axs[fration_tuple[i][1],fration_tuple[i][2]].set_ylabel(nomey)
			if i == 2 or i == 3:
				axs[fration_tuple[i][1],fration_tuple[i][2]].set_xlabel(r'$R^{1/4}$ (arcsec)')
			
			axs[fration_tuple[i][1],fration_tuple[i][2]].set_ylim(np.min(coef)-abs(np.max(coef)-np.min(coef))/5.,np.max(coef)+abs(np.max(coef)-np.min(coef))/5.)
					
			axs[fration_tuple[i][1],fration_tuple[i][2]].legend(fontsize='xx-small')
		plt.tight_layout()
		plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_sub.png')
		plt.close()

'''
