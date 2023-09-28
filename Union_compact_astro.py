from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
import copy
from subprocess import call
import os
import os.path
import scipy.stats as sst
import scipy.ndimage as ssn
import datetime 
from astropy.io import fits
import numpy.ma as ma
#######################
def codband(band):
	if band=='u':
		cob=0
	elif band=='g':
		cob=1
	elif band=='r':
		cob=2
	elif band=='i':
		cob=3
	elif band=='z':
		cob=4
	return cob

###################################

def gain(camcol,cband,run):
	if int(run)<1100:
		gainvec=[[1.62,3.32,4.71,5.165,4.745],[1.595,3.855,4.6,6.565,5.155],[1.59,3.845,4.72,4.86,4.885],[1.6,3.995,4.76,4.885,4.775],[1.47,4.05,4.725,4.64,3.48],[2.17,4.035,4.895,4.76,4.69]]
	else:
		gainvec=[[1.62,3.32,4.71,5.165,4.745],[1.825,3.855,4.6,6.565,5.155],[1.59,3.845,4.72,4.86,4.885],[1.6,3.995,4.76,4.885,4.775],[1.47,4.05,4.725,4.64,3.48],[2.17,4.035,4.895,4.76,4.69]]
	ga=gainvec[int(camcol)-1][cband]

	return ga

#########################################

def downframes(cluster,run,camcol,field):

	frame = 'http://data.sdss3.org/sas/dr12/boss/photoObj/frames/301/%d/%d/frame-r-%06d-%d-%04d.fits.bz2' % (run,camcol,run,camcol,field)
	psf = 'http://data.sdss3.org/sas/dr12/boss/photo/redux/301/%d/objcs/%d/psField-%06d-%d-%04d.fit' % (run,camcol,run,camcol,field)
	call('wget -r -nd -q --directory-prefix='+cluster+' '+frame,shell=True)
	call('wget -r -nd -q --directory-prefix='+cluster+' '+psf,shell=True)
	call('cp -i /home/andre/Documentos/L07/galfit /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/read_PSF /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/base_default.sex /home/andre/Documentos/L07/'+cluster, shell=True)
	call('bunzip2 */frame*.bz2',shell=True)
	return

##################################################################33
# FLAG DA MASCARA
def masktest(mask):
	#
	close_mask=mask[int(mask.shape[0]*0.25):int(mask.shape[0]*0.75),int(mask.shape[1]*0.25):int(mask.shape[1]*0.75)]
	#
	npix = len(np.where(close_mask==0)[0])
	nmask = len(np.where(close_mask==1)[0])
	#
	if nmask/(npix+nmask) >= 0.2:
		flagvalue=1
	else:
		flagvalue=0
	#
	return flagvalue
#################################################################
def lnt(cluster,X,Y):

#SERSIC UNICO
	par = open('pargal_WHL_compact_astro.dat','a')
	info_err=open('aux_pargal_WHL_compact_astro.dat','a')
	
	os.chdir(cluster+'/')
	call('rm ajust-bcg-exp.fits',shell=True)
	call('res_assim2.fits',shell=True)
	call('./galfit feedme.r',shell=True)

	ajust1 = fits.getdata('ajust-bcg-r.fits',1)
	ajust3 = fits.getdata('ajust-bcg-r.fits',3)
	mask = fits.getdata('bcg_r_mask.fits')
	mask_b = fits.getdata('bcg_r_mask_b.fits')
	header = fits.getheader('ajust-bcg-r.fits',2)
##############################################################
	re = float(header['1_RE'].split()[0].replace('*',''))
	mag = float(header['1_MAG'].split()[0].replace('*',''))
	magb=mag+2.5*log10(2.)
	n = float(header['1_N'].split()[0].replace('*',''))
	axis = float(header['1_AR'].split()[0].replace('*',''))
	pa = float(header['1_PA'].split()[0].replace('*',''))
	chi2 = float(header['CHI2NU'])
	xc=float(header['1_XC'].split()[0].replace('*',''))
	yc=float(header['1_YC'].split()[0].replace('*',''))
	zeropoints=float(header['MAGZPT'])
##################################################################3
	re_err = float(header['1_RE'].split()[2].replace('*',''))
	mag_err = float(header['1_MAG'].split()[2].replace('*',''))
	n_err = float(header['1_N'].split()[2].replace('*',''))
	axis_err = float(header['1_AR'].split()[2].replace('*',''))
	pa_err = float(header['1_PA'].split()[2].replace('*',''))
#######################################################################################
	flagmask=masktest(mask)
	#CALCULO DO DELTA

	cord = np.nonzero(mask_b)

	badxc=np.average(cord[0])	
	badyc=np.average(cord[1])

	goodcenter_dist=float(np.sqrt((badyc-Y)**2+(badxc-X)**2))

	badcenter_dist=np.average(np.sqrt((cord[0]-badyc)**2+(cord[1]-badxc)**2))
	
	delta=float(goodcenter_dist/badcenter_dist)

	if delta < 0.3:
		flagdelta=0
	else:
		flagdelta=1
###############################################################################

	#CALCULO DO RFF
	sbk=np.std(ajust1[np.where((mask_b == 0) & (mask==0))])

	nn2=len(np.nonzero(mask_b)[0])
	xy = np.sum(np.absolute(ajust3[np.where(mask_b == 1)]))
	xn = np.sum(ajust1[np.where(mask_b == 1)])
	
	rff=(xy-0.8*sbk*nn2)/xn
	
##############################################################
# INICIO DA ASSIMETRIA

#IMG SMALL 
	xxc = X-5
	xxc2 = X+5
	yyc = Y-5
	yyc2 = Y+5
	
	bcg_small = ajust1[int(yyc):int(yyc2),int(xxc):int(xxc2)]# img bcg
	mask_small = mask[int(yyc):int(yyc2),int(xxc):int(xxc2)]# mask normal 
	mask_b_small = mask_b[int(yyc):int(yyc2),int(xxc):int(xxc2)]# mask bcg
	
	dxb=0
	dyb=0
	speb=0
	matrix = ([1,0],[0,1])
	# CALCULO DA ASSIMETRIA 10X10

	for a in range(1000):
		#
		dx = np.random.normal(dxb,0.1)
		dy = np.random.normal(dyb,0.1)			
		#
		bcg_small_temp = ssn.affine_transform(bcg_small,matrix,offset=[dx,dy],mode='constant')# img bcg
		mask_b_small_temp = ssn.affine_transform(mask_b_small,matrix,offset=[dx,dy],mode='constant')# mask bcg
		mask_small_temp = ssn.affine_transform(mask_small,matrix,offset=[dx,dy],mode='constant')# mask normal
		#
		bcg_small_rot=np.rot90(bcg_small_temp,2)#img bcg rot
		mask_b_small_rot=np.rot90(mask_b_small_temp,2)#mask bcg rot
		mask_small_rot=np.rot90(mask_small_temp,2)#mask normal rot		
		
		#############
		vas=bcg_small_temp[np.where(((bcg_small_temp != 0.0) & (bcg_small_rot != 0.0)) & ((mask_b_small_temp==1) & (mask_small_temp == 0) & (mask_b_small_rot == 1) & (mask_small_rot == 0)))] #bcg normal
		
		var= bcg_small_rot[np.where(((bcg_small_temp != 0.0) & (bcg_small_rot != 0.0)) & ((mask_b_small_temp==1) & (mask_small_temp == 0) & (mask_b_small_rot == 1) & (mask_small_rot == 0)))] #bcg rotacionada
		
	
		spe = (1-sst.spearmanr(vas,var)[0])
		if spe < speb or speb == 0:
			speb = spe
			dxb = dx
			dyb = dy
################################################################
# ASSIMETRIA GRANDE

	bcg_data = ssn.affine_transform(ajust1,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# img bcg
	mask_b_data = ssn.affine_transform(mask_b,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask bcg
	mask_data = ssn.affine_transform(mask,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask normal
	
	bcg_rot= np.rot90((bcg_data),2)#img bcg rot
	mask_b_rot=np.rot90((mask_b_data),2)#mask bcg rot
	mask_rot=np.rot90((mask_data),2)#mask normal rot

	vas=bcg_data[np.where(((bcg_data!=0.0) & (bcg_rot!=0.0)) & ((mask_data == 0) & (mask_rot == 0)) & ((mask_b_data == 1) | (mask_b_rot == 1)))] #bcg normal
	
	var=bcg_rot[np.where(((bcg_data!=0.0) & (bcg_rot!=0.0)) & ((mask_data == 0) & (mask_rot == 0)) & ((mask_b_data == 1) | (mask_b_rot == 1)))] #bcg rotacionada

	res=np.sum(np.absolute(vas-var))

	vas1=np.sum(vas)

	nn2=len(vas)
		
	A1=((res-1.127*sbk*nn2)/vas1)/2.
	A0 = (1-sst.spearmanr(vas,var)[0])		
	ou1=open('feedme_exp.r','w')
	ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00   1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n\n # Object number: X MAIN SOURCE DISK \n 0) expdisk \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3)  %f \t 1 \t\t #  total   magnitude \n 4)  %f \t 1 \t\t#  Rs[pix] \n 9) %f \t 1 \t\t #  axis ratio (b/a) \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'%('bcg_r.fits','ajust-bcg-exp.fits','sigma-r.fits','bcg_r_psf_b.fits','bcg_r_mask.fits',1,ajust1.shape[0],1,ajust1.shape[1],zeropoints,xc,yc,mag,re/4.,axis,pa,xc,yc,magb,re,axis,pa))

	ou1.close()

	call('./galfit feedme_exp.r',shell=True)

	if os.path.isfile('ajust-bcg-exp.fits'):
	
		header_exp = fits.getheader('ajust-bcg-exp.fits',2)
	
		re_exp = float(header_exp['1_RE'].split()[0].replace('*',''))
		mag_exp = float(header_exp['1_MAG'].split()[0].replace('*',''))
		n_exp = float(header_exp['1_N'].split()[0].replace('*',''))
		axis_exp = float(header_exp['1_AR'].split()[0].replace('*',''))
		pa_exp = float(header_exp['1_PA'].split()[0].replace('*',''))
		chi2_exp = float(header_exp['CHI2NU'])
		xc_exp=float(header_exp['1_XC'].split()[0].replace('*',''))
		yc_exp=float(header_exp['1_YC'].split()[0].replace('*',''))

		rs_exp= float(header_exp['2_RS'].split()[0].replace('*',''))
		magd_exp = float(header_exp['2_MAG'].split()[0].replace('*',''))
		axisd_exp = float(header_exp['2_AR'].split()[0].replace('*',''))
		pad_exp= float(header_exp['2_PA'].split()[0].replace('*',''))

	##################################################################3
		re_err_exp = float(header_exp['1_RE'].split()[2].replace('*',''))
		mag_err_exp = float(header_exp['1_MAG'].split()[2].replace('*',''))
		n_err_exp = float(header_exp['1_N'].split()[2].replace('*',''))
		axis_err_exp = float(header_exp['1_AR'].split()[2].replace('*',''))
		pa_err_exp = float(header_exp['1_PA'].split()[2].replace('*',''))

		rs_err_exp = float(header_exp['2_RS'].split()[2].replace('*',''))
		magd_err_exp = float(header_exp['2_MAG'].split()[2].replace('*',''))
		axisd_err_exp = float(header_exp['2_AR'].split()[2].replace('*',''))
		pad_err_exp = float(header_exp['2_PA'].split()[2].replace('*',''))		
	else:
		chi2_exp,re_exp,mag_exp,n_exp,axis_exp,pa_exp,rs_exp,magd_exp,axisd_exp,pad_exp=0,0,0,0,0,0,0,0,0,0
		re_err_exp,mag_err_exp,n_err_exp,axis_err_exp,pa_err_exp,rs_err_exp, magd_err_exp,axisd_err_exp,pad_err_exp=0,0,0,0,0,0,0,0,0

	
	ou1=open('feedme_ss.r','w')
	ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00    1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n\n # Object number: X MAIN SOURCE - DISC \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00    1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'% ('bcg_'+band+'.fits','ajust-bcg-'+band+'_ss.fits','sigma-'+band+'.fits','bcg_'+band+'_psf_b.fits','bcg_'+band+'_mask.fits',1,ajust1.shape[0],1,ajust1.shape[1],zeropoints,xc,yc,mag,re/4.,axis,pa,xc,yc,magb,re,axis,pa))
	ou1.close()
	
	call('./galfit feedme_ss.r',shell=True)
	
	if os.path.isfile('ajust-bcg-r_ss.fits'):
		header_ss=fits.getheader('ajust-bcg-r_ss.fits',2)

		chi2_ss = float(header_ss['CHI2NU'])

		re_ss = float(header_ss['1_RE'].split()[0].replace('*',''))
		mag_ss = float(header_ss['1_MAG'].split()[0].replace('*',''))
		n_ss = float(header_ss['1_N'].split()[0].replace('*',''))
		axis_ss = float(header_ss['1_AR'].split()[0].replace('*',''))
		pa_ss = float(header_ss['1_PA'].split()[0].replace('*',''))

		rd_ss = float(header_ss['2_RE'].split()[0].replace('*',''))
		magd_ss = float(header_ss['2_MAG'].split()[0].replace('*',''))
		nd_ss = float(header_ss['2_N'].split()[0].replace('*',''))
		axisd_ss = float(header_ss['2_AR'].split()[0].replace('*',''))
		pad_ss = float(header_ss['2_PA'].split()[0].replace('*',''))

	##################################################################3

		re_err_ss = float(header_ss['1_RE'].split()[2].replace('*',''))
		mag_err_ss = float(header_ss['1_MAG'].split()[2].replace('*',''))
		n_err_ss = float(header_ss['1_N'].split()[2].replace('*',''))
		axis_err_ss = float(header_ss['1_AR'].split()[2].replace('*',''))
		pa_err_ss = float(header_ss['1_PA'].split()[2].replace('*',''))

		rd_err_ss = float(header_ss['2_RE'].split()[2].replace('*',''))
		magd_err_ss = float(header_ss['2_MAG'].split()[2].replace('*',''))
		nd_err_ss = float(header_ss['2_N'].split()[2].replace('*',''))
		axisd_err_ss = float(header_ss['2_AR'].split()[2].replace('*',''))
		pad_err_ss = float(header_ss['2_PA'].split()[2].replace('*',''))


	else:

		chi2_ss,re_ss,mag_ss,n_ss,axis_ss,pa_ss,rd_ss,magd_ss,nd_ss,axisd_ss,pad_ss=0,0,0,0,0,0,0,0,0,0,0
		re_err_ss,mag_err_ss,n_err_ss,axis_err_ss,pa_err_ss,rd_err_ss, magd_err_ss,nd_err_ss,axisd_err_ss,pad_err_ss=0,0,0,0,0,0,0,0,0,0
		

	par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(cluster,chi2,re,mag,n,axis,pa,float(rff),float(A0),float(A1),delta,flagdelta,flagmask,chi2_exp,re_exp,mag_exp,n_exp,axis_exp,pa_exp,rs_exp,magd_exp,axisd_exp,pad_exp,
chi2_ss,re_ss,mag_ss,n_ss,axis_ss,pa_ss,rd_ss,magd_ss,nd_ss,axisd_ss,pad_ss))
	par.close()
	
	info_err.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(cluster,re_err,mag_err,n_err,axis_err,pa_err,re_err_exp,mag_err_exp,n_err_exp,axis_err_exp,pa_err_exp,rs_err_exp,magd_err_exp,axisd_err_exp,pad_err_exp,
re_err_ss,mag_err_ss,n_err_ss,axis_err_ss,pa_err_ss,rd_err_ss,magd_err_ss,nd_err_ss,axisd_err_ss,pad_err_ss))
	info_err.close()

	call('rm bcg_r.fits',shell=True)
	call('rm sigma-r.fits',shell=True)
	call('rm galfit.*',shell=True)
	call('rm bcg_r_psf.fits',shell=True)
	call('rm check*',shell=True)
	
	os.chdir('../')

	return
	
###############################################################################################################################################################

ok=[]
inpcheck = open('pargal_WHL_compact_astro.dat','r+')
for item in inpcheck.readlines():	
	ok.append(item.split()[0])
with open('data_indiv_clean.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_clean.dat','r')
band='r'
cband=codband(band)
for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	cluster=ll1[0]
	run=str(ll1[2]).zfill(6)
	camcol=str(ll1[3])
	field=str(ll1[4]).zfill(4)
	x0=float(ll1[5])
	y0=float(ll1[6])
	if cluster in ok:
		call(cluster+'/res_assim2.fits',shell=True)
		pass
	else:
		with open('base_default.sex','r') as inp2:
			ninp2=len(inp2.readlines())
		inp2=open('base_default.sex','r')
		out1=open(cluster+'/base_default.sex','w')
		for j in range(0,ninp2):
			ls2=inp2.readline()
			ll2=ls2.split()
			if len(ll2)>0 and ll2[0]=='CATALOG_NAME':

				ll2[1]=ll1[0]+'/out_sex_large.cat'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='DETECT_MINAREA':
				ll2[1]='100'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])

			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]='128'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='CHECKIMAGE_NAME':
				ll2[1]=cluster+'/check1_large.fits,'+cluster+'/check2_large.fits,'+cluster+'/check3_large.fits'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			else:
				out1.write('%s' % ls2)
		inp2.close()
		out1.close()
		call('sex '+cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits -c '+cluster+'/base_default.sex',shell=True)

		header = fits.open(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',memmap=True)[0].header
		data = fits.open(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',memmap=True)[0].data
	
		CRPIX1=float(header['CRPIX1'])
		CRPIX2=float(header['CRPIX2'])
		CRVAL1=float(header['CRVAL1'])
		CRVAL2=float(header['CRVAL2'])
		CD1_1=float(header['CD1_1'])
		CD1_2=float(header['CD1_2'])
		CD2_1=float(header['CD2_1'])
		CD2_2=float(header['CD2_2'])
		EXPTIME=float(header['EXPTIME'])
		NMGY=float(header['NMGY'])

#############################################3
# obtencao do numero da bcg e formatacao do tamanho da imagem 

		with open(cluster+'/out_sex_large.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_large.cat','r')
		dist=1000.
		xb=-1.
		for i in range(0,ninfa):
			lsa=infa.readline()
			lla=lsa.split()
			if lla[0]!='#':
				X=float(lla[7])
				Y=float(lla[8])
				XI=(CD1_1*(X-CRPIX1)+CD1_2*(Y-CRPIX2))*pi/180.
				ETA=(CD2_1*(X-CRPIX1)+CD2_2*(Y-CRPIX2))*pi/180.
				p=(XI**2+ETA**2)**0.5
				c=np.arctan(p)
				RA=np.arctan(XI*sin(c)/(p*cos(CRVAL2*pi/180.)*cos(c)-ETA*sin(CRVAL2*pi/180.)*sin(c)))*180./pi+CRVAL1
				DEC=np.arcsin(cos(c)*sin(CRVAL2*pi/180.)+ETA*sin(c)*cos(CRVAL2*pi/180.)/p)*180./pi
				subt=float(lla[12])
				if ((RA-x0)**2+(DEC-y0)**2)**0.5<dist:
					xb=float(lla[0])
					dist=((RA-x0)**2+(DEC-y0)**2)**0.5
					ARA=RA
					ADEC=DEC
					AX=X
					AY=Y
					thetabcg=float(lla[11])
					siz=1.5*float(lla[9])*float(lla[4])
					razax=float(lla[10])/float(lla[9])
					mag=float(lla[2])
					countbcg=float(lla[1])
		infa.close()
		infx=int(floor(AX))-int(floor(siz))
		supx=int(floor(AX))+int(floor(siz))
		infy=int(floor(AY))-int(floor(siz))
		supy=int(floor(AY))+int(floor(siz))
		xc=int(floor(siz))
		yc=int(floor(siz))
	#INFX	
		if infx<=0:
			xc=int(siz+infx)
			ninfx=1
		elif infx>0:
			ninfx=infx
	#INFY:
		if infy<=0:
			yc=int(siz+infy)
			ninfy=1
		elif infy>0:

			ninfy=infy
	#SUPX

		if supx>data.shape[1]:
			nsupx=data.shape[1]
		elif supx<=data.shape[1]:
			nsupx=supx
	#SUPY
	
		if supy>data.shape[0]:
			nsupy=data.shape[0]
		elif supy<=data.shape[0]:
			nsupy=supy	
##########################################################

		inp2=open('base_default.sex','r')
		out1=open(cluster+'/base_default.sex','w')
	
		for j in range(0,ninp2):
			ls2=inp2.readline()
			ll2=ls2.split()
			if len(ll2)>0 and ll2[0]=='CATALOG_NAME':
				ll2[1]=cluster+'/out_sex_small.cat'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])

			elif len(ll2)>0 and ll2[0]=='DETECT_MINAREA':
				ll2[1]='3'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]='64'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			elif len(ll2)>0 and ll2[0]=='CHECKIMAGE_NAME':
				ll2[1]=cluster+'/check1_small.fits,'+cluster+'/check2_small.fits,'+cluster+'/check3_small.fits'
				lstrin=' '
				for k in range(0,len(ll2)):
					lstrin+=ll2[k]+' '
				out1.write('%s\n' % lstrin[1:len(lstrin)])
			else:
				out1.write('%s' % ls2)
		inp2.close()
		out1.close()
		call('sex '+cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits -c '+cluster+'/base_default.sex',shell=True)
	
#################################################################################
# obtencao do numero da bcg do small.cat
		with open(cluster+'/out_sex_small.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_small.cat','r')
		dist=100000.
		yb0=-1.
		for i in range(0,ninfa):
			lsa=infa.readline()
			lla=lsa.split()
			if lla[0]!='#':
				X=float(lla[7])
				Y=float(lla[8])
				XI=(CD1_1*(X-CRPIX1)+CD1_2*(Y-CRPIX2))*pi/180.
				ETA=(CD2_1*(X-CRPIX1)+CD2_2*(Y-CRPIX2))*pi/180.
				p=(XI**2+ETA**2)**0.5
				c=np.arctan(p)
				RA=np.arctan(XI*sin(c)/(p*cos(CRVAL2*pi/180.)*cos(c)-ETA*sin(CRVAL2*pi/180.)*sin(c)))*180./pi+CRVAL1
				DEC=np.arcsin(cos(c)*sin(CRVAL2*pi/180.)+ETA*sin(c)*cos(CRVAL2*pi/180.)/p)*180./pi
				if ((RA-x0)**2+(DEC-y0)**2)**0.5<dist:
					yb=float(lla[0])
					dist=((RA-x0)**2+(DEC-y0)**2)**0.5
					VRA=RA
					VDEC=DEC
					VX=X
					VY=Y

	#######################################################
		# MASCARA COMECA AQUI
	######################################
			# mask normal 
		#####################

		cks = fits.open(cluster+'/check3_small.fits')[0].data
		ckl = fits.open(cluster+'/check3_large.fits')[0].data

		data_check_small = cks[ninfy:nsupy,ninfx:nsupx]
		data_check_large = ckl[ninfy:nsupy,ninfx:nsupx]
		
		pre_mask_miss = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		pre_mask_dirty = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		
		############
		#PRE MASK BCG
		pre_mask_miss[(data_check_small == yb)] = 1
		pre_mask_miss[(data_check_large == xb)] = 1
		#PRE MASK NORMAL
		pre_mask_dirty[(data_check_small != yb) & (data_check_small != 0)] = 1
		pre_mask_dirty[(data_check_large != xb) & (data_check_large != 0)] = 1

		##################################3		
		#MASK CLEANER
		#
		mask_miss = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		mask_dirty = np.zeros(ma.shape(data_check_large),dtype=np.intc)
		#
		bcg_pixeis = np.nonzero(pre_mask_miss)
		coords_distances=[bcg_pixeis[0],bcg_pixeis[1],np.sqrt((bcg_pixeis[0]-yc)**2+(bcg_pixeis[1]-xc)**2)]
		###########################################################
		cont=[[int(yc)],[int(xc)]]
		sorteddists=sorted(coords_distances[2])
		sorteddists.append(sorteddists[-1]+1)
		##############################################################3		
		for ii in range(len(sorteddists)):
			for i in range(len(coords_distances[0])):
				if sorteddists[ii]<=coords_distances[2][i]<sorteddists[ii+1]:
					ypix,xpix=coords_distances[0][i],coords_distances[1][i]
					laterals=[]
					for k in range(-1,2):
						for kk in range(-1,2):
							laterals.append((ypix+k,xpix+kk))
					contiguos_array=[]
					for k in range(len(cont[0])):
						contiguos_array.append((cont[0][k],cont[1][k]))
		############################################################3
					if len(set(laterals) & set(contiguos_array))>0:
						cont[0].append(ypix)
						cont[1].append(xpix)

		contiguos=tuple(np.asarray(cont))
		
		mask_miss[contiguos] = 1
		pre_mask_miss[contiguos]=0

		mask_dirty[pre_mask_dirty == 1] = 1
		mask_dirty[pre_mask_miss == 1] = 1
				
		fits.writeto(cluster+'/bcg_r_mask_b.fits',mask_miss,header=header,overwrite=True)
		fits.writeto(cluster+'/bcg_r_mask.fits',mask_dirty,header=header,overwrite=True)

########################################################## 
# inicio do sigma.fits e imagem da bcg:
		
		header.set('GAIN', value=gain(camcol,cband,run), comment=None, before=None, after=None)
	
		data1 = data[ninfy:nsupy,ninfx:nsupx]
		data2 = (((data1-subt)*EXPTIME)/NMGY)

		sig = data2*gain(camcol,cband,run)	

		vec = sig[np.where((mask_miss == 0) & (mask_dirty == 0))]

		z = np.sum(np.power(vec-np.average(vec),2))/(len(vec)-1.)
		sig1 = (np.sqrt(np.absolute(sig)+z))/gain(camcol,cband,run)

		fits.writeto(cluster+'/sigma-'+band+'.fits',sig1, header=header,overwrite=True)
		fits.writeto(cluster+'/bcg_'+band+'.fits',data2, header=header,overwrite=True)
##########################################################
# prepare psf file
		
		call('./read_PSF '+cluster+'/psField-'+str(run).zfill(6)+'-'+str(camcol)+'-'+str(field).zfill(4)+'.fit '+str(cband+1)+' '+str(AX)+' '+str(AY)+' '+cluster+'/bcg_'+band+'_psf.fits',shell=True)
		
		datapsf = fits.open(cluster+'/bcg_'+band+'_psf.fits')[0].data

		datapsf=np.subtract(datapsf,1000.)
		fits.writeto(cluster+'/bcg_'+band+'_psf_b.fits',datapsf, header=header,overwrite=True)

# zeropoints:

		NMGY = float(header['NMGY'])
		a=22.5 

		b=2.5
		c=0.1569166

		zeropoints = a+(b*log10(1./NMGY))
		mag = mag+zeropoints - 2.5*(log10(EXPTIME)) + 2.5*(log10(c))

#arquivo constr.all:

		ou3=open(cluster+'/constr.all','w')
		ou3.write(' 1 x -5 5 \n 1 y -5 5') 

# finish calculations and print input data to galfit

		ou1=open(cluster+'/feedme.r','w')
		ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00    1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'% ('bcg_'+band+'.fits','ajust-bcg-r.fits','sigma-'+band+'.fits','bcg_'+band+'_psf_b.fits','bcg_r_mask.fits',1,nsupx-ninfx,1,nsupy-ninfy,zeropoints,xc,yc,mag,siz/4.5,razax,thetabcg))
		ou1.close()
		ou3.close()
		lnt(ll1[0],xc,yc)
		
###########################
		
