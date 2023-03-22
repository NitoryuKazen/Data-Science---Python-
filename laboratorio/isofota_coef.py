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

########################################################################################################################################################
#cluster,sky,chisq_galfit,skyvalue,sigmasky,ellipgrad,pagrad,extval_ellip/ell0,extval_pa-pa0,chirat_1,chirat_2,chisq_s_1,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0],sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],-sspopt[2],sspopt[3],sspopt[4],sspopt[5],raio,slope 30
########################################################################################################################################################
with open('/home/andrelpkaipper/Documentos/Projetos/L07/pargal_L07_compact.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('/home/andrelpkaipper/Documentos/Projetos/L07/pargal_L07_compact.dat','r')
ok=[]
output=open('iso_geral_values_coef.dat','r+')
for item in output.readlines():
	ok.append(item.split()[0])

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()	
	cluster=ll1[0]
	flagmask=int(float(ll1[12]))
	flagdelta=int(float(ll1[10]))
###################################################################3
	
	if [flagmask,flagdelta] == [0,0]:

		if cluster in ok:# or cluster == '1417':
		#	pass
		#else:
			print(cluster,ik)
			
			call('mkdir L07_coef/'+cluster,shell=True)	
			ajust1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[1].data
			ajust2 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits', memmap=True)[2].data
			header = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[2].header
			header1 = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/ajust-bcg-r.fits',memmap=True)[1].header
			mask = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask.fits', memmap=True)[0].data
			maskb = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_r_mask_b.fits', memmap=True)[0].data
			
			datag = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_g.fits', memmap=True)[0].data
			maskg = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_g_mask.fits', memmap=True)[0].data
			maskbg = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_g_mask_b.fits', memmap=True)[0].data
			headerg = fits.open('/home/andrelpkaipper/Documentos/Projetos/L07/'+cluster+'/bcg_g.fits', memmap=True)[0].header

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

			xcg=float(headerg['XC'])
			ycg=float(headerg['YC'])

########################################################################
		# sky calculation
			skyvalue=calc_sky(ajust1,mask,maskb,xc,yc,cluster)
			image = ajust1 - skyvalue
			negpixs = image[np.where((image<skyvalue) & (image>-10000.))]
			sigmasky = np.std(negpixs)/np.sqrt(1.-2./np.pi)
			print('Sigma sky =',sigmasky)
			sigmasky/=4.

			skyvalueg=calc_sky(datag,maskg,maskbg,xcg,ycg,cluster)
			imageg = datag - skyvalueg
			negpixsg = imageg[np.where((imageg<skyvalue) & (imageg>-10000.))]
			sigmaskyg = np.std(negpixsg)/np.sqrt(1.-2./np.pi)
			print('Sigma sky g=',sigmaskyg)
			sigmaskyg/=4.
##########################################################################3
		#AJUSTE DAS ISOFOTAS
			isoimage=ma.masked_where(mask==1,image)
			isoimage_g=ma.masked_where(mask==1,imageg)
			intens_g=[]
			if os.path.isfile('L07_coef/'+cluster+'/iso_table.dat'):
				temp=[[] for i in range(17)]
				iso_table=open('L07_coef/'+cluster+'/iso_table.dat','r')
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

				epopt, epcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(eps), sigma=ellip_err)
				ellipgrad=epopt[0]
				print('Ellipticity gradient =',format(epopt[0], '.3E'))
				ppopt, pcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(pa), sigma=pa_err)
				pagrad=ppopt[0]
				print('PA gradient =',format(ppopt[0], '.3E'))
			
				fig, ax = plt.subplots(1,2,figsize=(6, 12))
				ax[0].imshow(isoimage_g,vmin=0,vmax=1000,origin='lower')
				ax[1].imshow(isoimage,vmin=0,vmax=1000,origin='lower')
				plt.xlabel('X (pix)')
				plt.ylabel('Y (pix)')
				plt.tight_layout()
				plt.show()
				
				'''
				for i in range(len(intens)):
					isogal_g=phi.EllipseGeometry(x0=xcg, y0=ycg, sma=float(sma[i])/0.396, eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
					aper=php.EllipticalAperture((isogal_g.x0, isogal_g.y0), isogal_g.sma,isogal_g.sma * (1 - isogal_g.eps),isogal_g.pa)
					fig, ax = plt.subplots(figsize=(6, 6))
					ax.imshow(isoimage_g,vmin=0,vmax=1000,origin='lower')
					aper.plot(color='white')
					plt.xlabel('X (pix)')
					plt.ylabel('Y (pix)')
					plt.tight_layout()
					plt.show()

					isomodel_g=phi.Ellipse(isoimage_g,isogal_g)

					isolist_g=isomodel_g.fit_isophote(sma=float(sma[i]),step=0.02,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)
					
					print(isolist_g.intens)

					intens_g.append(isolist_g.intens)

					print(float(sma[i])/0.396,float(eps[i]),float(pa[i]),isolist_g.sma,isolist_g.eps,isolist_g.pa,i)

				print(len(intens_g),len(intens))
				print('------------------------------')
				print(intens_g,intens)
			else:			
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
				plt.savefig('L07_coef/'+cluster+'/'+cluster+'_iso_free.png')
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
					plt.savefig('L07_coef/'+cluster+'/'+cluster+'_iso_free.png')
					plt.close(fig)
	###########################################################################################

			#VALORES DE INTERESSE
				x0=isolist.x0[np.where(isolist.intens>sigmasky)]
				y0=isolist.y0[np.where(isolist.intens>sigmasky)]
				sma=isolist.sma[np.where(isolist.intens>sigmasky)]*0.396
				pa=isolist.pa[np.where(isolist.intens>sigmasky)]
				eps=isolist.eps[np.where(isolist.intens>sigmasky)]
				intens=isolist.intens[np.where(isolist.intens>sigmasky)]
				a3=isolist.a3[np.where(isolist.intens>sigmasky)]
				b3=isolist.b3[np.where(isolist.intens>sigmasky)]
				a4=isolist.a4[np.where(isolist.intens>sigmasky)]
				b4=isolist.b4[np.where(isolist.intens>sigmasky)]
				
				ellip_err=isolist.ellip_err[np.where(isolist.intens>sigmasky)]
				pa_err=isolist.pa_err[np.where(isolist.intens>sigmasky)]
				int_err=isolist.int_err[np.where(isolist.intens>sigmasky)]
				a3_err=isolist.a3_err[np.where(isolist.intens>sigmasky)]
				b3_err=isolist.b3_err[np.where(isolist.intens>sigmasky)]
				a4_err=isolist.a4_err[np.where(isolist.intens>sigmasky)]
				b4_err=isolist.b4_err[np.where(isolist.intens>sigmasky)]
				
				extval_ellip=np.average(eps,weights=np.power(sma,2))
				extval_pa=np.average(pa,weights=np.power(sma,2))
				maxrad=np.max(sma) 
				epopt, epcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(eps), sigma=ellip_err)
				ellipgrad=epopt[0]
				print('Ellipticity gradient =',format(epopt[0], '.3E'))
				ppopt, pcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(pa), sigma=pa_err)
				pagrad=ppopt[0]
				print('PA gradient =',format(ppopt[0], '.3E'))

				inptrue=open('L07_coef/'+cluster+'/iso_table.dat','w')
				inptrue.write('%f \t %f \t %f \n'%(extval_ellip,extval_pa,maxrad))		
				for r in range(len(sma)):
					inptrue.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],a3[r],b3[r],a4[r],b4[r],ellip_err[r],pa_err[r],int_err[r],a3_err[r],b3_err[r],a4_err[r],b4_err[r]))
				inptrue.close()

##########################################################################################3
			#BANDA G
			
#			isofotes_g=[]
#			for i in range(1):len(sma)

			
				#fit_isophote acho que funciona, fazer um loop com o ellipse geometry e apartir disso 
###############################################################################################
		# AJUSTES DAS LEIS DE DE VAUCOULERS, SERSIC E SERSIC + EXPONENCIAL RESPECTIVAMENTE

			mag_iso = 22.5+(2.5*np.log10(1./NMGY)) -2.5*np.log10(intens) + 2.5*(log10(0.1569166))

			spopt,spcov=scp.curve_fit(sersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[1.e3,10,4],maxfev=10000000)

			chisq_s=np.sum(np.power((mag_iso-(sersic(sma,*spopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-3.)
			#
			sepopt,sepcov = scp.curve_fit(sersicexp,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[spopt[0],spopt[1],spopt[2],spopt[0]/1000.,2.*spopt[0]],maxfev=10000000)

			chisq_se=np.sum(np.power((mag_iso-(sersicexp(sma,*sepopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)
			#
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
			sspopt,sspcov = scp.curve_fit(doublesersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[spopt[0],spopt[1],spopt[2],spopt[0]/1000.,2.*spopt[0],1],maxfev=10000000)

			chisq_ss=np.sum(np.power((mag_iso-(doublesersic(sma,*sspopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)

			for i in range(20):
				nsspopt,nsspcov = scp.curve_fit(doublesersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[np.random.uniform(1,1.e5),np.random.uniform(1,100),np.random.uniform(0.5,8),np.random.uniform(1,1.e5),np.random.uniform(1,200),np.random.uniform(0.5,4)],
maxfev=10000000)

				chisq_ss_temp=np.sum(np.power((mag_iso-(doublesersic(sma,*nsspopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)

				if chisq_ss_temp<chisq_ss:
					sspopt=1.*nsspopt
					sspcov=1.*nsspcov
					chisq_ss=chisq_ss_temp

			chirat_2=chisq_s/chisq_ss
			#

			b=2.*spopt[2]-1./3+4./405/spopt[2]+46./25515/spopt[2]**2+131./1148175/spopt[2]**3-2194697./30690717750/spopt[2]**4
	
			raio=sma[np.abs(mag_iso-20.66).argmin()]
				
			slope=-np.divide(2.5*b,spopt[2]*np.log(10))*(raio/spopt[1])**(1/spopt[2])

			print('Sersic fit results:\n    Ie =',format(abs(spopt[0]), '.3E'),'\n    Re =',format(spopt[1], '.3E'),'\n    n =',format(abs(spopt[2]), '.3E'))
			print('S+E fit results:\n    Ie =',format(abs(sepopt[0]), '.3E'),'\n    Re =',format(sepopt[1], '.3E'),'\n    n =',format(abs(sepopt[2]), '.3E'),'\n    I0 =',format(abs(sepopt[3]), '.3E'),'\n    Rd =',format(sepopt[4], '.3E'))
			print('S+S fit results:\n    Ie =',format(abs(sspopt[0]), '.3E'),'\n    Re =',format(sspopt[1], '.3E'),'\n    n =',format(abs(sspopt[2]), '.3E'),'\n    I0 =',format(abs(sspopt[3]), '.3E'),'\n    Rd =',format(sspopt[4], '.3E'),'\n    nd =',format(sspopt[5], '.3E'))
				
			output.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n'%(cluster,sky,chisq_galfit,skyvalue,sigmasky,ellipgrad,pagrad,extval_ellip/ell0,extval_pa-pa0,chirat_1,chirat_2,chisq_s_1,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0],sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],sspopt[2],sspopt[3],sspopt[4],sspopt[5],raio,slope))
			'''
			#output.close()
##############################################################################################

