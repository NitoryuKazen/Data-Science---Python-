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

def testgr(cluster,ellgalfit,pagalfit,imgr,imgg,sigskyg,sigskyr,c1,c2,c3,c4):
	
	#LEITURA DO ISOTABLE		
	
	temp=[[] for i in range(17)]
	iso_table=open('WHL_coef_gain/'+cluster+'/iso_table.dat','r')
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
	#####################################################################
	#TESTE G-R COM ISOFOTAS LIVRES & GALFIT
	
	isofree_g,isofree_r,galfix_r,galfix_g=[[[],[]],[[],[]],[[],[]],[[],[]]]
	isovec=[[],[],[],[]]
	smagr=sma/0.396
	smagal=np.geomspace(5,maxrad/0.396,len(smagr))
	for i in range(len(intens)):
		#####
		#ELIPSES LIVRE BANDA G E R
		#
		freegeo_g=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagr[i]), eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
		freesamp_g=phi.EllipseSample(imgg,sma=float(smagr[i]),sclip=3.0, nclip=5,geometry=freegeo_g)
		freesamp_g.update()					
		freeiso_g=phi.Isophote(freesamp_g,0,True,0)
		isovec[0].append(freeiso_g)
		isofree_g[0].append(freeiso_g.intens)
		isofree_g[1].append(freeiso_g.int_err)
		#
		freegeo_r=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagr[i]), eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
		freesamp_r=phi.EllipseSample(imgr,sma=float(smagr[i]),sclip=3.0, nclip=5,geometry=freegeo_r)
		freesamp_r.update()
		freeiso_r=phi.Isophote(freesamp_r,0,True,0)
		isovec[1].append(freeiso_r)
		isofree_r[0].append(freeiso_r.intens)
		isofree_r[1].append(freeiso_r.int_err)
		#####

		#ELIPSES DO GALFIT BANDA G E R RESPECTIVAMENTE
		#
		galgeo_g=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagal[i]),eps=ellgalfit,pa=pagalfit, fix_center=True,fix_eps=True,fix_pa=True)
		galsamp_g=phi.EllipseSample(imgg,sma=float(smagal[i]), sclip=3.0, nclip=5,geometry=galgeo_g)
		galsamp_g.update()
		galfot_g=phi.Isophote(galsamp_g,0,True,0)
		isovec[2].append(galfot_g)
		galfix_g[0].append(galfot_g.intens)
		galfix_g[1].append(galfot_g.int_err)
		#	
		galgeo_r = phi.EllipseGeometry(x0=xc, y0=yc,sma=float(smagal[i]),eps=ellgalfit,pa=pagalfit, fix_center=True,fix_eps=True,fix_pa=True)
		galsamp_r=phi.EllipseSample(imgr,sma=float(smagal[i]), sclip=3.0, nclip=5,geometry=galgeo_r)
		galsamp_r.update()
		galfot_r=phi.Isophote(galsamp_r,0,True,0)
		isovec[3].append(galfot_r)
		galfix_r[0].append(galfot_r.intens)
		galfix_r[1].append(galfot_r.int_err)
		#
	free_g=phi.IsophoteList(isovec[0])
	free_r=phi.IsophoteList(isovec[1])
	fixgal_g=phi.IsophoteList(isovec[2])
	fixgal_r=phi.IsophoteList(isovec[3])
	
	list_iso=[free_g,free_r,fixgal_g,fixgal_r]
	name_iso=['free_g','free_r','fixgal_g','fixgal_r']
	image_iso=[(imgg*c3)/c4,(imgr*c1)/c2,(imgg*c3)/c4,(imgr*c1)/c2]
	
	for item in list_iso:
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.imshow(image_iso[list_iso.index(item)],vmin=0,vmax=1500,origin='lower')
		paircont=0
		for raio in item.sma:
			iso = item.get_closest(raio)
			x, y, = iso.sampled_coordinates()
			if paircont%2==0:
				plt.plot(x, y, color='white',linewidth=1)
			paircont+=1
		plt.xlabel('X (pix)')
		plt.ylabel('Y (pix)')
		plt.tight_layout()
		plt.savefig('WHL_coef_gain/'+cluster+'/'+cluster+'_'+name_iso[list_iso.index(item)]+'.png')
		plt.close(fig)

#################################################################################################################
	
	iso_table_vg=open('WHL_coef_gain/'+cluster+'/iso_table_gr.dat','w')
	iso_table_vg.write('%f \t %f \t %f \t %f \t %f\n'%(extval_ellip,extval_pa,maxrad,sigskyg,sigskyr))		
	for r in range(len(sma)):
		iso_table_vg.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],a3[r],b3[r],a4[r],b4[r], ellip_err[r],pa_err[r],int_err[r],a3_err[r],b3_err[r], a4_err[r],b4_err[r],isofree_g[0][r],isofree_g[1][r],isofree_r[0][r],isofree_r[1][r],galfix_r[0][r],galfix_r[1][r],galfix_g[0][r],galfix_g[1][r]))
	iso_table_vg.close()		

	return free_g,fixgal_r

########################################################################################################################################################

#cluster,sky,chisq_galfit,skyvalue,sigmasky,ellipgrad,pagrad,extval_ellip/ell0,extval_pa-pa0,chirat_1,chirat_2,chisq_s_1,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0],sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],-sspopt[2],sspopt[3],sspopt[4],sspopt[5],raio,slope 30

########################################################################################################################################################
#14 a 27

convec=[[] for i in range(15)]
oldput=open('iso_geral_values_coef_WHL.dat','r')
for obj in oldput.readlines():
	convec[0].append((obj.split()[0]))
	convec[1].append(float(obj.split()[14]))
	convec[2].append(float(obj.split()[15]))
	convec[3].append(float(obj.split()[16]))
	convec[4].append(float(obj.split()[17]))
	convec[5].append(float(obj.split()[18]))
	convec[6].append(float(obj.split()[19]))
	convec[7].append(float(obj.split()[20]))
	convec[8].append(float(obj.split()[21]))
	convec[9].append(float(obj.split()[22]))
	convec[10].append(float(obj.split()[23]))
	convec[11].append(float(obj.split()[24]))
	convec[12].append(float(obj.split()[25]))
	convec[13].append(float(obj.split()[26]))
	convec[14].append(float(obj.split()[27]))

ok=[]
output=open('iso_geral_values_coef_WHL.dat','r+')#'check_isog.dat'
for item in output.readlines():
	ok.append(item.split()[0])
with open('/home/andre/Documents/Projetos/WHL/pargal_WHL_compact_astro.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('/home/andre/Documents/Projetos/WHL/pargal_WHL_compact_astro.dat','r')
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()	
	cluster=ll1[0]
	flagmask=int(float(ll1[11]))
	flagdelta=int(float(ll1[12]))
	print(flagmask,flagdelta)
########################################################################################################################################################
########################################################################################################################################################
	if [flagmask,flagdelta] != [0,0] or cluster in ok:
		pass
	else:
		print(cluster,ik)
		call('mkdir WHL_coef_gain/'+cluster,shell=True)	

		ajust1 = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/ajust-bcg-r.fits')[1].data
		ajust2 = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/ajust-bcg-r.fits')[2].data
		header = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/ajust-bcg-r.fits')[2].header
		headerr = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/ajust-bcg-r.fits')[1].header
		mask = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_r_mask.fits')[0].data
		maskb = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_r_mask_b.fits')[0].data

		datar0 = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_r.fits')[0].data
		datag0 = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_g.fits')[0].data
		headerg = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_g.fits')[0].header
		maskgr = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_gr_mask.fits')[0].data
		maskbgr = fits.open('/home/andre/Documents/Projetos/WHL/'+cluster+'/bcg_gr_mask_b.fits')[0].data
		
		xc=float(header['1_XC'].split()[0].replace('*',''))
		yc=float(header['1_YC'].split()[0].replace('*',''))
		re = float(header['1_RE'].split()[0].replace('*',''))
		mag = float(header['1_MAG'].split()[0].replace('*',''))
		n = float(header['1_N'].split()[0].replace('*',''))
		sky=float(header['2_SKY'].split()[0].replace('*',''))
		pa0=(float(header['1_PA'].split()[0].replace('*',''))+90.)*np.pi/180.
		ell0=1.-float(header['1_AR'].split()[0].replace('*',''))
		chisq_galfit=float(header['CHI2NU'])
		
		NMGYr=float(headerr['NMGY'])
		EXPTIMEr=float(headerr['EXPTIME'])
		
		NMGYg=float(headerg['NMGY'])
		EXPTIMEg=float(headerg['EXPTIME'])
		
		datar=(datar0*NMGYr)/EXPTIMEr
		datag=(datag0*NMGYg)/EXPTIMEg

		# sky calculation
		########################################33
		#CALCULO DA BANDA G E R 

		skyvalueg=calc_sky(datag,maskgr,maskbgr,xc,yc,cluster)
		imageg = datag - skyvalueg
		negpixsg = imageg[np.where((imageg<skyvalueg) & (imageg>-10000.))]
		sigmaskyg = np.std(negpixsg)/np.sqrt(1.-2./np.pi)
		print('Sigma sky g=',sigmaskyg)
		sigmaskyg/=4.

		#

		skyvaluer=calc_sky(datar,maskgr,maskbgr,xc,yc,cluster)
		imager = datar - skyvaluer
		negpixsr = imager[np.where((imager<skyvaluer) & (imager>-10000.))]
		sigmaskyr = np.std(negpixsr)/np.sqrt(1.-2./np.pi)
		print('Sigma sky r=',sigmaskyr)
		sigmaskyr/=4.
		
########################################################################################################################################################
########################################################################################################################################################
##########################LEITURA DO ISOTABLE################################################

		isoimage_g=ma.masked_where(maskgr==1,imageg)
		isoimage_r=ma.masked_where(maskgr==1,imager)

		if os.path.isfile('WHL_coef_gain/'+cluster+'/iso_table.dat'):
			temp=[[] for i in range(17)]
			iso_table=open('WHL_coef_gain/'+cluster+'/iso_table.dat','r')
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

			testcor=testgr(cluster,ell0,pa0,isoimage_r,isoimage_g,sigmaskyg, sigmaskyr,EXPTIMEr,NMGYr,EXPTIMEg,NMGYg)
		else:
			isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=pa0)  
			isomodel=phi.Ellipse(isoimage_r,isogal) 
			isolist=isomodel.fit_image(minsma=5,maxsma=np.max(isoimage_r.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)
			######################################################################################
			#VERIFICACAO DE CONVERGENCIA
		
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow((isoimage_r*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')
			paircont=0
			for sma in isolist.sma:
				if isolist.intens[np.where(isolist.sma==sma)][0]>sigmaskyr:
					iso = isolist.get_closest(sma)
					x, y, = iso.sampled_coordinates()
					if paircont%2==0:
						plt.plot(x, y, color='white',linewidth=1)
					paircont+=1
			plt.xlabel('X (pix)')
			plt.ylabel('Y (pix)')
			plt.tight_layout()
			plt.savefig('WHL_coef_gain/'+cluster+'/'+cluster+'_iso_free.png')
			plt.close(fig)
			isotry=0
			while len(isolist.sma)==0 and isotry<=10000: # caso nao haja convergencia, repetir
				isotry+=1
				isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=np.random.uniform(-180.,180.))
				isomodel=phi.Ellipse(isoimage_r,isogal)
				isolist=isomodel.fit_image(minsma=5,maxsma=np.max(isoimage_r.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)

				fig, ax = plt.subplots(figsize=(6, 6))
				ax.imshow((isoimage_r*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')

				paircont=0
				for sma in isolist.sma:
					if isolist.intens[np.where(isolist.sma==sma)][0]>sigmaskyr:
						iso = isolist.get_closest(sma)
						x, y, = iso.sampled_coordinates()
						if paircont%2==0:
							plt.plot(x, y, color='white',linewidth=1)
						paircont+=1
				plt.xlabel('X (pix)')
				plt.ylabel('Y (pix)')
				plt.tight_layout()
				plt.savefig('WHL_coef_gain/'+cluster+'/'+cluster+'_iso_free.png')
				plt.close(fig)
			if isotry<10000:
				print(isotry)
			###########################################################################################
			#VALORES DE INTERESSE
						
				x0=isolist.x0[np.where(isolist.intens>sigmaskyr)]
				y0=isolist.y0[np.where(isolist.intens>sigmaskyr)]
				sma=isolist.sma[np.where(isolist.intens>sigmaskyr)]*0.396
				pa=isolist.pa[np.where(isolist.intens>sigmaskyr)]
				eps=isolist.eps[np.where(isolist.intens>sigmaskyr)]
				intens=isolist.intens[np.where(isolist.intens>sigmaskyr)]
				a3=isolist.a3[np.where(isolist.intens>sigmaskyr)]
				b3=isolist.b3[np.where(isolist.intens>sigmaskyr)]
				a4=isolist.a4[np.where(isolist.intens>sigmaskyr)]
				b4=isolist.b4[np.where(isolist.intens>sigmaskyr)]
				#
				ellip_err=isolist.ellip_err[np.where(isolist.intens>sigmaskyr)]
				pa_err=isolist.pa_err[np.where(isolist.intens>sigmaskyr)]
				int_err=isolist.int_err[np.where(isolist.intens>sigmaskyr)]
				a3_err=isolist.a3_err[np.where(isolist.intens>sigmaskyr)]
				b3_err=isolist.b3_err[np.where(isolist.intens>sigmaskyr)]
				a4_err=isolist.a4_err[np.where(isolist.intens>sigmaskyr)]
				b4_err=isolist.b4_err[np.where(isolist.intens>sigmaskyr)]
				#
				extval_ellip=np.average(eps,weights=np.power(sma,2))
				extval_pa=np.average(pa,weights=np.power(sma,2))
				maxrad=np.max(sma) 
				epopt, epcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(eps), sigma=ellip_err)
				ellipgrad=epopt[0]
				print('Ellipticity gradient =',format(epopt[0], '.3E'))
				ppopt, pcov = scp.curve_fit(linfunc,np.log10(sma), np.log10(pa), sigma=pa_err)
				pagrad=ppopt[0]
				print('PA gradient =',format(ppopt[0], '.3E'))
				#
				inptrue=open('WHL_coef_gain/'+cluster+'/iso_table.dat','w')
				inptrue.write('%f \t %f \t %f \n'%(extval_ellip,extval_pa,maxrad))		
				for r in range(len(sma)):
					inptrue.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],a3[r],b3[r],a4[r],b4[r],ellip_err[r],pa_err[r],int_err[r],a3_err[r],b3_err[r],a4_err[r],b4_err[r]))
				inptrue.close()
				#
				testcor=testgr(cluster,ell0,pa0,isoimage_r,isoimage_g,sigmaskyg, sigmaskyr,EXPTIMEr,NMGYr,EXPTIMEg,NMGYg)
			else:
				pass			
########################################################################################################################################################
########################################################################################################################################################
# AJUSTES DAS LEIS DE DE VAUCOULERS, SERSIC E SERSIC + EXPONENCIAL RESPECTIVAMENTE
	#aqui que eu vou fazer o loop
		if isotry < 10000:
			mag_iso = 22.5-2.5*np.log10(intens) + 2.5*(log10(0.1569166))
			#
			spopt,spcov=scp.curve_fit(sersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[1.e3,10,4],maxfev=10000000)
			chisq_s=np.sum(np.power((mag_iso-(sersic(sma,*spopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-3.)
			#
			sepopt,sepcov = scp.curve_fit(sersicexp,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[spopt[0],spopt[1],spopt[2],spopt[0]/1000.,2.*spopt[0]],maxfev=10000000)
			chisq_se=np.sum(np.power((mag_iso-(sersicexp(sma,*sepopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)
			#
			sspopt,sspcov = scp.curve_fit(doublesersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[spopt[0],spopt[1],spopt[2],spopt[0]/1000.,2.*spopt[0],1],maxfev=10000000)
			chisq_ss=np.sum(np.power((mag_iso-(doublesersic(sma,*sspopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)
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
			for i in range(20):
				nsspopt,nsspcov = scp.curve_fit(doublesersic,sma,mag_iso,sigma=2.5*np.log10(np.exp(1.))*int_err/intens,p0=[np.random.uniform(1,1.e5),np.random.uniform(1,100),np.random.uniform(0.5,8),np.random.uniform(1,1.e5),np.random.uniform(1,200), np.random.uniform(0.5,4)],maxfev=10000000)
				chisq_ss_temp=np.sum(np.power((mag_iso-(doublesersic(sma,*nsspopt)))/(2.5*np.log10(np.exp(1.))*int_err/intens),2))/(len(intens)-5.)

				if chisq_ss_temp<chisq_ss:
					sspopt=1.*nsspopt
					sspcov=1.*nsspcov
					chisq_ss=chisq_ss_temp

			chirat_2=chisq_s/chisq_ss
			#
			print('Sersic fit results:\n    Ie =',format(abs(spopt[0]), '.3E'),'\n    Re =',format(spopt[1], '.3E'),'\n    n =',format(abs(spopt[2]), '.3E'))
			print('S+E fit results:\n    Ie =',format(abs(sepopt[0]), '.3E'),'\n    Re =',format(sepopt[1], '.3E'),'\n    n =',format(abs(sepopt[2]), '.3E'),'\n    I0 =',format(abs(sepopt[3]), '.3E'),'\n    Rd =',format(sepopt[4], '.3E'))
			print('S+S fit results:\n    Ie =',format(abs(sspopt[0]), '.3E'),'\n    Re =',format(sspopt[1], '.3E'),'\n    n =',format(abs(sspopt[2]), '.3E'),'\n    I0 =',format(abs(sspopt[3]), '.3E'),'\n    Rd =',format(sspopt[4], '.3E'),'\n    nd =',format(sspopt[5], '.3E'))

			b=2.*spopt[2]-1./3+4./405/spopt[2]+46./25515/spopt[2]**2+131./1148175/spopt[2]**3-2194697./30690717750/spopt[2]**4

			raio=sma[np.abs(mag_iso-20.66).argmin()]
			raio_index=np.where(sma==raio)[0][0]
			slope=-np.divide(2.5*b,spopt[2]*np.log(10))*(raio/spopt[1])**(1/spopt[2])

			output.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %i \n'%(cluster,sky,chisq_galfit,skyvaluer,sigmaskyr,ellipgrad,pagrad,extval_ellip/ell0, extval_pa-pa0,chirat_1,chirat_2,chisq_s_1,chisq_se,chisq_ss,spopt[0],spopt[1],spopt[2],sepopt[0], sepopt[1],sepopt[2],sepopt[3],sepopt[4],sspopt[0],sspopt[1],sspopt[2],sspopt[3],sspopt[4],sspopt[5],raio,slope,raio_index))
		else:
			pass
	#output.close()
		
############################################################################################
