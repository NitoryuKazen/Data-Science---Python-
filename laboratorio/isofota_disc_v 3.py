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

def sersicexp(x,ie,re,n,i0,rd):
	n=abs(n)
	ie=abs(ie)
	bn=2.*n-1./3+4./405/n+46./25515/n**2+131./1148175/n**3-2194697./30690717750/n**4
	i0=abs(i0)
	rd=abs(rd)
	i=-2.5*np.log10(ie*np.exp(-bn*((np.divide(x,re))**(1./n)-1.))+i0*np.exp(-np.divide(x,rd)))+22.5
	return i

def envel(x,i0,rd):
	i0=abs(i0)
	rd=abs(rd)
	i=-2.5*np.log10(i0*np.exp(-np.divide(x,rd)))+22.5
	return i


with open('pargal_L07.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pre_parser_info_clean_r4.dat','r')
ok=[]
output=open('iso_geral_values_maskcor_3.dat','r+')
for item in output.readlines():
	ok.append(item.split()[0])

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()	
	cluster=ll1[0]
	tipo=ll1[1]
	flagmask=int(ll1[2])
	flagn=int(ll1[3])
	flagmag=int(ll1[4])
	flagre=int(ll1[5])
	flagrmax=int(ll1[6])
	flagdelta=int(ll1[21])
###################################################################3
	
	if [flagmask,flagdelta] == [0,0]:

		if int(cluster) in ok :#or cluster == '1417':
			pass
		else:
			print(cluster,ik)
			
			call('mkdir L07_maskcor/'+tipo+'/'+cluster,shell=True)	
			ajust1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[1].data
			ajust2 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[2].data
			header = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[2].header
			header1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[1].header
			mask = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask.fits', memmap=True)[0].data
			maskb = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask_b.fits', memmap=True)[0].data

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
########################################################################
		# sky calculation
			skyvalue=calc_sky(ajust1,mask,maskb,xc,yc,cluster)
			image = ajust1 - skyvalue
			negpixs = image[np.where((image<skyvalue) & (image>-10000.))]
			sigmasky = np.std(negpixs)/np.sqrt(1.-2./np.pi)
			print('Sigma sky =',sigmasky)
			sigmasky/=4.
##########################################################################3
		#AJUSTE DAS ISOFOTAS
			isoimage=ma.masked_where(mask==1,image)
			isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=pa0)
			isomodel=phi.Ellipse(isoimage,isogal)
			isolist=isomodel.fit_image(minsma=5,maxsma=np.max(image.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)
#########################################################################
		#VERIFICACAO DE CONVERGENCIA
			
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow(isoimage,vmin=0,vmax=1000,origin='lower')
			paircont=0
			for sma in isolist.sma:
				if isolist.intens[np.where(isolist.sma==sma)][0]>sigmasky:
					iso = isolist.get_closest(sma)
					x, y, = iso.sampled_coordinates()
					if paircont%2==0:
						plt.plot(x, y, color='white',linewidth=1)
					paircont+=1
			plt.xlabel('X (pix)')
			plt.ylabel('Y (pix)')
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_iso_free.png')
			plt.close(fig)

			while len(isolist.sma)==0: # caso nao haja convergencia, repetir
				isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=np.random.uniform(-180.,180.))
				isomodel=phi.Ellipse(isoimage,isogal)
				isolist=isomodel.fit_image(minsma=5,maxsma=np.max(image.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)

				fig, ax = plt.subplots(figsize=(6, 6))
				ax.imshow(isoimage,vmin=0,vmax=1000,origin='lower')

				paircont=0
				for sma in isolist.sma:
					if isolist.intens[np.where(isolist.sma==sma)][0]>sigmasky:
						iso = isolist.get_closest(sma)
						x, y, = iso.sampled_coordinates()
						if paircont%2==0:
							plt.plot(x, y, color='white',linewidth=1)
						paircont+=1
				plt.xlabel('X (pix)')
				plt.ylabel('Y (pix)')
				plt.tight_layout()
				plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_iso_free.png')
				plt.close(fig)
###########################################################################################
		#VALORES DE INTERESSE
						
			x0=isolist.x0[np.where(isolist.intens>sigmasky)]
			y0=isolist.y0[np.where(isolist.intens>sigmasky)]
			sma=isolist.sma[np.where(isolist.intens>sigmasky)]*0.396
			pa=isolist.pa[np.where(isolist.intens>sigmasky)]
			eps=isolist.eps[np.where(isolist.intens>sigmasky)]
			intens=isolist.intens[np.where(isolist.intens>sigmasky)]
			ellip_err=isolist.ellip_err[np.where(isolist.intens>sigmasky)]
			pa_err=isolist.pa_err[np.where(isolist.intens>sigmasky)]
			int_err=isolist.int_err[np.where(isolist.intens>sigmasky)]
			
			extval_ellip=np.average(eps,weights=np.power(sma,2))
			extval_pa=np.average(pa,weights=np.power(sma,2))
			maxrad=np.max(sma) # raio da ultima elipse ajustada
			print(sma)
			epopt, epcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(eps), sigma=ellip_err)
			ellipgrad=epopt[0]
			print('Ellipticity gradient =',format(epopt[0], '.3E'))
			ppopt, pcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(pa), sigma=pa_err)
			pagrad=ppopt[0]
			print('PA gradient =',format(ppopt[0], '.3E'))

			inptrue=open('L07_maskcor/'+tipo+'/'+cluster+'/iso_table.dat','w')
			inptrue.write('%f \t %f \t %f \n'%(extval_ellip,extval_pa,maxrad))		
			for r in range(len(sma)):
				inptrue.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],ellip_err[r],pa_err[r],int_err[r]))
			inptrue.close()
			
###############################################################################################
		# AJUSTES DAS LEIS DE DE VAUCOULERS, SERSIC E SERSIC + EXPONENCIAL RESPECTIVAMENTE

			mag_iso = 22.5+(2.5*np.log10(1./NMGY)) -2.5*np.log10(intens) + 2.5*(log10(0.1569166))

			vpopt,vpcov = scp.curve_fit(devauc,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[1.e3,10],maxfev=10000000)
			#
			spopt,spcov=scp.curve_fit(sersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[1.e3,10,4],maxfev=10000000)

			chisq_s=np.sum(np.power((mag_iso-(sersic(sma,*spopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-3.)
			#
			sepopt,sepcov = scp.curve_fit(sersicexp,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[spopt[0],spopt[1],spopt[2],spopt[0]/1000.,2.*spopt[0]],maxfev=10000000)

			chisq_se=np.sum(np.power((mag_iso-(sersicexp(sma,*sepopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)

			for i in range(20):

				nsepopt,nsepcov = scp.curve_fit(sersicexp,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[np.random.uniform(1,1.e5),np.random.uniform(1,100),np.random.uniform(0.5,8),np.random.uniform(1,1.e5),np.random.uniform(1,200)],maxfev=10000000)

				chisq_se_temp=np.sum(np.power((mag_iso-(sersicexp(sma,*nsepopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)

				if chisq_se_temp<chisq_se:
					sepopt=1.*nsepopt
					sepcov=1.*nsepcov
					chisq_se=chisq_se_temp

			chirat_1=chisq_s/chisq_se
			chisq_s_1=chisq_s
			#
			print('de Vauc fit results:\n    Ie =',format(abs(vpopt[0]), '.3E'),'\n    Re =',format(vpopt[1], '.3E'))
			print('Sersic fit results:\n    Ie =',format(abs(spopt[0]), '.3E'),'\n    Re =',format(spopt[1], '.3E'),'\n    n =',format(abs(spopt[2]), '.3E'))
			print('S+E fit results:\n    Ie =',format(abs(sepopt[0]), '.3E'),'\n    Re =',format(sepopt[1], '.3E'),'\n    n =',format(abs(sepopt[2]), '.3E'),'\n    I0 =',format(abs(sepopt[3]), '.3E'),'\n    Rd =',format(sepopt[4], '.3E'))
###################################################################################
		# PLOT DE PARAMETROS GERAIS RAIOX(MAGNITUDE,ELIPTICIDADE,PA)

			plt.figure(figsize=(9, 3))
			plt.subplots_adjust(hspace=0.35, wspace=0.35)
			plt.subplot(1, 3, 1)
			plt.errorbar(np.power(sma,0.25),mag_iso,yerr=2*2.5*np.log10(np.exp(1.))*int_err/intens,fmt='o',markersize=4,color='k',ecolor='0.5')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.ylim([np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.,np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.])
			plt.gca().invert_yaxis()
			plt.subplot(1, 3, 2)
			plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=4, color='k',ecolor='0.5')
			plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylabel(r'$e$')
			plt.subplot(1, 3, 3)
			plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=4, color='k',ecolor='0.5')
			plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
			plt.xlabel(r'$R$ (arcsec)')
			plt.ylabel('PA (deg)')
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_iso_info.png')	
			plt.close()	
			
			plt.figure()
			plt.errorbar(np.power(sma,0.25), eps, yerr=2*ellip_err, fmt='o', markersize=4, color='k',ecolor='0.5')
			plt.ylim([0,np.max(eps)+np.max(eps)*0.1])
			plt.xlabel(r'$R$ (arcsec)')
			plt.ylabel(r'$e$')
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'/elipticidade.png')
			
			plt.figure()
			plt.errorbar(np.power(sma,0.25), pa/np.pi*180., yerr=2*pa_err/np.pi*180., fmt='o', markersize=4, color='k',ecolor='0.5')
			plt.ylim([np.min(pa/np.pi*180.)-(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))*0.1,np.max(pa/np.pi*180.)+(np.max(pa/np.pi*180.)-np.min(pa/np.pi*180.))/10.])
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylabel('PA (deg)')
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'/position_angle.png')

######################################################################################################

			#PLOT DO PERFIL DE SERSIC E SERIC + EXPONENCIAL RESPECTIVAMENTE

			fig = plt.subplots(figsize=(12, 6),sharey=True)
			plt.subplots_adjust(hspace=0.35, wspace=0.35)
			plt.suptitle('Isofotas livres')

			plt.subplot(1, 2, 1)
			plt.errorbar(np.power(sma,0.25),mag_iso,yerr=2*2.5*np.log10(np.exp(1.))*int_err/intens,label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(sma,0.25),sersic(sma,*spopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_s ,color='blue',linewidth=5)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.])
			plt.legend()
			plt.tight_layout()

			plt.subplot(1, 2, 2)
			plt.errorbar(np.power(sma,0.25),mag_iso,yerr=2*2.5*np.log10(np.exp(1.))*int_err/intens,label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(sma,0.25),sersicexp(sma,*sepopt),label=r'S+E Fit, $\chi^2_\nu=$%.2f' % chisq_se,color='orange',linewidth=3)
			plt.plot(np.power(sma,0.25),sersic(sma,*sepopt[0:3]),label='S',color='red',linewidth=2)
			plt.plot(np.power(sma,0.25),envel(sma,*sepopt[3:5]),label='E',color='yellow',linewidth=2)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso)+(np.max(mag_iso)-np.min(mag_iso))/10.,np.min(mag_iso)-(np.max(mag_iso)-np.min(mag_iso))/10.])
			plt.legend()
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_s_se_models.png')	
			plt.close()	
###############################################################################################
###############################################################################################
###############################################################################################
########ELIPSES FIXAS COM OS VALORES OBTIDOS PARA A ULTIMA ELIPSE OBTIDA PELO GALFIT

			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow(isoimage,vmin=0,vmax=1000,origin='lower')
			gsma=5
			smaintens=[[],[],[]]
			while gsma<=maxrad/0.396:
				isolist = phi.EllipseSample(isoimage,sma=gsma,astep=0.02,x0=xc, y0=yc, eps=ell0, position_angle=pa0, sclip=3.0, nclip=5)
				intenslist=isolist.extract()[2]
				if len(intenslist)>1:
					smaintens[0].append(gsma*0.396)
					smaintens[1].append(np.average(intenslist))
					smaintens[2].append(np.std(intenslist)/np.sqrt(len(intenslist)))
					gsma*=1.02
					if len(smaintens[0])%2==0:
						plt.plot(isolist.coordinates()[0],isolist.coordinates()[1],c='w',linewidth=1)
				else:
					break
			plt.xlabel('X (pix)')
			plt.ylabel('Y (pix)')
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_galfix_ellipses.png')	
			plt.close(fig)
			print(smaintens[0])
			#print(np.power(sma,0.25))
####################################################################################################
		# AJUSTES DAS LEIS DE DE VAUCOULERS, SERSIC E SERSIC + EXPONENCIAL RESPECTIVAMENTE

			mag_iso_galfix = 22.5+(2.5*np.log10(1./NMGY)) -2.5*np.log10(smaintens[1]) + 2.5*(log10(0.1569166))
			
			#
			fixvpopt,fixvpcov = scp.curve_fit(devauc,smaintens[0],mag_iso_galfix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[1.e3,10],maxfev=10000000)
			reffref=np.max(smaintens[0])/2.
			#
			fixspopt,fixspcov = scp.curve_fit(sersic,smaintens[0],mag_iso_galfix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[1.e3,10,4],maxfev=10000000)
					
			chisq_sfix=np.sum(np.power((mag_iso_galfix-(sersic(smaintens[0],*fixspopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-3.)
			#
			fixsepopt,fixsepcov = scp.curve_fit(sersicexp,smaintens[0],mag_iso_galfix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[fixspopt[0],fixspopt[1],
			fixspopt[2],fixspopt[0]/10.,2.*fixspopt[0]],maxfev=10000000)
			
			chisq_fixse=np.sum(np.power((mag_iso_galfix-(sersicexp(smaintens[0],*fixsepopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-5.)
			for i in range(20):
				nfixsepopt,nfixsepcov = scp.curve_fit(sersicexp,smaintens[0],mag_iso_galfix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[np.random.uniform(1,1.e5),np.random.uniform(1,100),
				np.random.uniform(0.5,8),np.random.uniform(1,1.e5),np.random.uniform(1,200)],maxfev=10000000)
				chisq_fixse_temp=np.sum(np.power((mag_iso_galfix-(sersicexp(smaintens[0],*nfixsepopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-5.)
				if chisq_fixse_temp<chisq_fixse:
					fixsepopt=1.*nfixsepopt
					fixsepcov=1.*nfixsepcov
					chisq_fixse=chisq_fixse_temp

			chirat_2=chisq_sfix/chisq_fixse

			chisq_s_2=chisq_sfix

			print ('## Fits for fixed parameters (galfit) ##')
			print('dV fit results:\n    Ie =',format(abs(fixvpopt[0]), '.3E'),'\n    Re =',format(fixvpopt[1], '.3E'))
			print('Sersic fit results:\n    Ie =',format(abs(fixspopt[0]), '.3E'),'\n    Re =',format(fixspopt[1], '.3E'),'\n    n =',format(abs(fixspopt[2]), '.3E'))
			print('S+E fit results:\n    Ie =',format(abs(fixsepopt[0]), '.3E'),'\n    Re =',format(fixsepopt[1], '.3E'),'\n    n =',format(abs(fixsepopt[2]), '.3E'),'\n    I0 =',format(abs(fixsepopt[3]), '.3E'),'\n    Rd =',format(fixsepopt[4], '.3E'))
###########################################################################################################
		# SERSIC & SERSIC + EXP
			fig = plt.subplots(figsize=(12, 6),sharey=True)
			plt.subplots_adjust(hspace=0.35, wspace=0.35)
			plt.suptitle('Isofotas GALFIx')

			plt.subplot(1, 2, 1)
			plt.errorbar(np.power(smaintens[0],0.25),mag_iso_galfix,yerr=2*2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(smaintens[0],0.25),sersic(smaintens[0],*fixspopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_sfix, color='blue',linewidth=5)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso_galfix)+(np.max(mag_iso_galfix)-np.min(mag_iso_galfix))/10.,np.min(mag_iso_galfix)-(np.max(mag_iso_galfix)-np.min(mag_iso_galfix))/10.])
			plt.legend()
			plt.tight_layout()

			plt.subplot(1, 2, 2)
			plt.errorbar(np.power(smaintens[0],0.25),mag_iso_galfix,yerr=2*2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(smaintens[0],0.25),sersicexp(smaintens[0],*fixsepopt),label=r'S+E Fit, $\chi^2_\nu=$%.2f' % chisq_fixse ,color='orange',linewidth=3)
			plt.plot(np.power(smaintens[0],0.25),sersic(smaintens[0],*fixsepopt[0:3]),label='S', color='red',linewidth=2)
			plt.plot(np.power(smaintens[0],0.25),envel(smaintens[0],*fixsepopt[3:5]),label='E', color='yellow',linewidth=2)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso_galfix)+(np.max(mag_iso_galfix)-np.min(mag_iso_galfix))/10.,np.min(mag_iso_galfix)-(np.max(mag_iso_galfix)-np.min(mag_iso_galfix))/10.])
			plt.legend()
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_s_se_models_galfix.png')	
			plt.close()
			print(smaintens[0])	
##############################################################################################
##############################################################################################
##############################################################################################
########ELIPSES FIXAS COM OS VALORES OBTIDOS PELO PHOTUTILS PARA A ULTIMA ELIPSE

			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow(isoimage,vmin=0,vmax=1000,origin='lower')
			psma=5
			smaintens=[[],[],[]]
			while psma<=maxrad/0.396:
				isolist = phi.EllipseSample(isoimage,sma=psma,astep=0.02,x0=xc, y0=yc, eps=extval_ellip, position_angle=extval_pa, sclip=3.0, nclip=5)
				intenslist=isolist.extract()[2]
				if len(intenslist)>1:
					smaintens[0].append(psma*0.396)
					smaintens[1].append(np.average(intenslist))
					smaintens[2].append(np.std(intenslist)/np.sqrt(len(intenslist)))
					psma*=1.02
					if len(smaintens[0])%2==0:
						plt.plot(isolist.coordinates()[0],isolist.coordinates()[1],c='w',linewidth=1)
				else:
					break
			plt.xlabel('X (pix)')
			plt.ylabel('Y (pix)')
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'/photofix_ellipses.png')	
			plt.close(fig)
##############################################################################################
		# AJUSTES DAS LEIS DE DE VAUCOULERS, SERSIC E SERSIC + EXPONENCIAL RESPECTIVAMENTE
			mag_iso_phofix = 22.5+(2.5*np.log10(1./NMGY)) -2.5*np.log10(smaintens[1]) + 2.5*(log10(0.1569166))

			pixvpopt,pixvpcov = scp.curve_fit(devauc,smaintens[0],mag_iso_phofix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[1.e3,10],maxfev=10000000)
			#
			pixspopt,pixspcov = scp.curve_fit(sersic,smaintens[0],mag_iso_phofix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[1.e3,10,4],maxfev=10000000)

			chisq_pixs=np.sum(np.power((mag_iso_phofix-(sersic(smaintens[0],*pixspopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-3.)
			#
			pixsepopt,pixsepcov = scp.curve_fit(sersicexp,smaintens[0],mag_iso_phofix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[pixspopt[0],pixspopt[1]
			,pixspopt[2],pixspopt[0],2.*pixspopt[0]],maxfev=10000000)

			chisq_pixse=np.sum(np.power((mag_iso_phofix-(sersicexp(smaintens[0],*pixsepopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-5.)

			for i in range(20):
				npixsepopt,npixsepcov = scp.curve_fit(sersicexp,smaintens[0],mag_iso_phofix,sigma=2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),p0=[np.random.uniform(1,1.e5),np.random.uniform(1,100),
				np.random.uniform(0.5,8),np.random.uniform(1,1.e5),np.random.uniform(1,200)],maxfev=10000000)
				chisq_pixse_temp=np.sum(np.power((mag_iso_phofix-(sersicexp(smaintens[0],*npixsepopt)))/(2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1])),2))/(len(smaintens[0])-5.)
				if chisq_pixse_temp<chisq_pixse:
					pixsepopt=1.*npixsepopt
					pixsepcov=1.*npixsepcov
					chisq_pixse=chisq_pixse_temp

			chirat_3=chisq_pixs/chisq_pixse

			chisq_s_3=chisq_pixs	

			print ('## Fits for fixed parameters (external) ##')
			print('dV fit results:\n    Ie =',format(abs(pixvpopt[0]), '.3E'),'\n    Re =',format(pixvpopt[1], '.3E'))
			print('Sersic fit results:\n    Ie =',format(abs(pixspopt[0]), '.3E'),'\n    Re =',format(pixspopt[1], '.3E'),'\n    n =',format(abs(pixspopt[2]), '.3E'))
			print('S+E fit results:\n    Ie =',format(abs(pixsepopt[0]), '.3E'),'\n    Re =',format(pixsepopt[1], '.3E'),'\n    n =',format(abs(pixsepopt[2]), '.3E'),'\n    I0 =',format(abs(pixsepopt[3]), '.3E'),'\n    Rd =',format(pixsepopt[4], '.3E'))
##############################################################################################
		# SERSIC & SERSIC + EXP

			fig = plt.subplots(figsize=(12, 6),sharey=True)
			plt.subplots_adjust(hspace=0.35, wspace=0.35)
			plt.suptitle('Isofotas Photofix')

			plt.subplot(1, 2, 1)
			plt.errorbar(np.power(smaintens[0],0.25),mag_iso_phofix,yerr=2*2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(smaintens[0],0.25),sersic(smaintens[0],*pixspopt),label=r'S Fit, $\chi^2_\nu=$%.2f' % chisq_pixs  , color='blue',linewidth=5)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso_phofix)+(np.max(mag_iso_phofix)-np.min(mag_iso_phofix))/10.,np.min(mag_iso_phofix)-(np.max(mag_iso_phofix)-np.min(mag_iso_phofix))/10.])
			plt.legend()
			plt.tight_layout()

			plt.subplot(1, 2, 2)
			plt.errorbar(np.power(smaintens[0],0.25),mag_iso_phofix,yerr=2*2.5*np.log10(np.exp(1.))*np.divide(smaintens[2],smaintens[1]),label='Obs. profile',fmt='o',markersize=8,zorder=0, color='k')
			plt.plot(np.power(smaintens[0],0.25),sersicexp(smaintens[0],*pixsepopt),label=r'S+E Fit, $\chi^2_\nu=$%.2f' % chisq_se ,color='orange',linewidth=3)
			plt.plot(np.power(smaintens[0],0.25),sersic(smaintens[0],*pixsepopt[0:3]),label='S', color='red',linewidth=2)
			plt.plot(np.power(smaintens[0],0.25),envel(smaintens[0],*pixsepopt[3:5]),label='E', color='yellow',linewidth=2)
			plt.gca().invert_yaxis()
			plt.ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
			plt.xlabel(r'$R^{1/4}$ (arcsec)')
			plt.ylim([np.max(mag_iso_phofix)+(np.max(mag_iso_phofix)-np.min(mag_iso_phofix))/10.,np.min(mag_iso_phofix)-(np.max(mag_iso_phofix)-np.min(mag_iso_phofix))/10.])
			plt.legend()
			plt.tight_layout()
			plt.savefig('L07_maskcor/'+tipo+'/'+cluster+'_s_se_models_photofix.png')	
			plt.close()
			
			#output.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'%(cluster,sky,chisq_galfit,skyvalue,sigmasky,ellipgrad,pagrad,extval_ellip/ell0,extval_pa-pa0,chirat_1,chirat_2,chirat_3,chisq_s_1,chisq_s_2,chisq_s_3,spopt[0],spopt[1],spopt[2],sepopt[0],sepopt[1],sepopt[2],sepopt[3],sepopt[4],	vpopt[0],vpopt[1],fixspopt[0],fixspopt[1],fixspopt[2],fixsepopt[0],fixsepopt[1],fixsepopt[2],fixsepopt[3],fixsepopt[4],
#fixvpopt[0],fixvpopt[1],pixspopt[0],pixspopt[1],pixspopt[2],pixsepopt[0],pixsepopt[1],pixsepopt[2],pixsepopt[3],pixsepopt[4],
#pixvpopt[0],pixvpopt[1]))
##############################################################################################
