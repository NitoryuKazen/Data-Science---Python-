from math import sin,cos,tan,pi,floor,log10,sqrt
import numpy as np
import pyfits
import copy
from subprocess import call
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as msc
import scipy.stats as sst
import scipy.ndimage as ssn
import datetime 
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
	print 'Downloading FRAME for '+cluster+'...'
	call('wget -r -nd -q --directory-prefix='+cluster+' '+frame,shell=True)
	print 'Downloading PSF for '+cluster+'...'
	call('wget -r -nd -q --directory-prefix='+cluster+' '+psf,shell=True)
	call('cp -i /home/andre/Documentos/L07/galfit /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/read_PSF /home/andre/Documentos/L07/'+cluster, shell=True)
	call('cp -i /home/andre/Documentos/L07/base_default.sex /home/andre/Documentos/L07/'+cluster, shell=True)
	call('bunzip2 */frame*.bz2',shell=True)
	return
##################################################################33
# FLAG DA MASCARA
def masktest(image,mask):
	nmask=0.
	npix=0.
	for i in range(int(0.25*image.shape[0]),int(0.75*image.shape[0])):
		for j in range(int(0.25*image.shape[1]),int(0.75*image.shape[1])):
			if mask[i,j] == 0:
				npix+=1.
			if mask[i,j] == 1:
				nmask+=1
	flagvalue=0
	if nmask/(npix+nmask) >= 0.2:
		flagvalue+=1
	else:
		flagvalue+=0
	
	return flagvalue
#################################################################
def lnt(cluster,X,Y):

#SERSIC UNICO

	par = open('pargal_L07_compact.dat','a')
	info_err=open('aux_pargal_L07_compact.dat','a')
	
	os.chdir(cluster+'/')

	call('./galfit feedme.r',shell=True)

	ajust1 = pyfits.getdata('ajust-bcg-r.fits',1)
	ajust3 = pyfits.getdata('ajust-bcg-r.fits',3)
	mask = pyfits.getdata('bcg_r_mask.fits')
	mask_b = pyfits.getdata('bcg_r_mask_b.fits')
	header = pyfits.getheader('ajust-bcg-r.fits',2)
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
	flagmask=masktest(ajust1,mask)
	#CALCULO DO DELTA
	dist_c=[]
	cord=[[],[]]
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			if mask_b[i,j] == 1:
				dist_c.append(float(np.sqrt((i-Y)**2+(j-X)**2)))
				cord[0].append(j)
				cord[1].append(i)
	
	badxc=np.average(cord[0])	
	badyc=np.average(cord[1])
	
	goodcenter_dist=float(np.sqrt((badyc-Y)**2+(badxc-X)**2))

	badcenter_dist=[]
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			if mask_b[i,j] == 1:
				badcenter_dist.append(float(np.sqrt((i-badyc)**2+(j-badxc)**2)))
	dist_bad=np.average(badcenter_dist)
	
	delta=float(goodcenter_dist/dist_bad)

	flagdelta=0
	if delta < 0.3:
		flagdelta=+0
	else:
		flagdelta=+1
###############################################################################
	p=0.
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			if mask_b[i,j] == 1:
				p+=1.
	d=0.
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			if mask[i,j] == 1 and mask_b[i,j] == 1:
				d+=1.
				
	if p == 0.:
		gama=1.
	else:
		gama=float(d/p)
###############################################################################

	#CALCULO DO RFF
	xs3=[]
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			if mask_b[i,j]==0 and mask[i,j]==0:
				xs3.append(ajust1[i,j])
	sbk=np.std(xs3)
	xy=0
	xn=0
	nn2=0
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			nn2+=(1.-mask[i,j])*(mask_b[i,j])
			xy+=np.absolute((ajust3[i,j])*((1.-mask[i,j])*(mask_b[i,j])))
			xn+=(ajust1[i,j])*((1.-mask[i,j])*(mask_b[i,j]))
	rff=(xy-0.8*sbk*nn2)/xn	
##############################################################
# INICIO DA ASSIMETRIA

#IMG SMALL 
	xxc = X-5
	xxc2 = X+5
	yyc = Y-5
	yyc2 = Y+5
	data0 = ajust1[yyc:yyc2,xxc:xxc2]# img bcg
	data01 = mask[yyc:yyc2,xxc:xxc2]# mask normal 
	data02 = mask_b[yyc:yyc2,xxc:xxc2]# mask bcg
	
	# CALCULO DA ASSIMETRIA 10X10
	dxb = 0 
	dyb = 0
	speb = 0 
	for a in range(1000):
		matrix = ([1,0],[0,1])
		dx = np.random.normal(dxb,0.1)
		dy = np.random.normal(dyb,0.1)			
		data30 = ssn.interpolation.affine_transform(data0,matrix,offset=[dx,dy],mode='constant')# img bcg
		data31 = ssn.interpolation.affine_transform(data02,matrix,offset=[dx,dy],mode='constant')# mask bcg
		data32 = ssn.interpolation.affine_transform(data01,matrix,offset=[dx,dy],mode='constant')# mask normal
		data30r= np.rot90((data30),2)#img bcg rot
		data31r=np.rot90((data31),2)#mask bcg rot
		data32r=np.rot90((data32),2)#mask normal rot
#############
		vas=[]#bcg normal
		var=[]#bcg rotacionada
		for i in range(0,yyc2-yyc):
			for j in range(0,xxc2-xxc):
				if data30[i,j] != 0.0 and data30r[i,j] != 0.0:
					if data32[i,j] == 0 and data31[i,j] == 1 and data32r[i,j] == 0 and data31r[i,j] == 1:
						vas.append(data30[i,j])
						var.append(data30r[i,j])
		spe = (1-sst.spearmanr(vas,var)[0])
		if spe < speb or speb == 0:
			speb = spe
			dxb = dx
			dyb = dy
################################################################
# ASSIMETRIA GRANDE
	matrix = ([1,0],[0,1])
	data30 = ssn.interpolation.affine_transform(ajust1,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# img bcg
	data31 = ssn.interpolation.affine_transform(mask_b,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask bcg
	data32 = ssn.interpolation.affine_transform(mask,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask normal
	data30r= np.rot90((data30),2)#img bcg rot
	data31r=np.rot90((data31),2)#mask bcg rot
	data32r=np.rot90((data32),2)#mask normal rot

	vas=[]#bcg normal
	var=[]#bcg rotacionada
	datat=copy.deepcopy(data30)
	datatr=copy.deepcopy(data30r)	
	res=0
	vas1=0
	nn2=0
	for i in range(0,ajust1.shape[0]):
		for j in range(0,ajust1.shape[1]):
			if data30[i,j] != 0.0 and data30r[i,j] != 0.0:
				if data32[i,j] == 0 and data32r[i,j] == 0 and (data31[i,j] == 1  or data31r[i,j] == 1):
					vas.append(data30[i,j])
					var.append(data30r[i,j])
					datat[i,j]=1
					datatr[i,j]=1
					res+=np.absolute(data30[i,j]-data30r[i,j])
					vas1+=(data30[i,j])
					nn2+=1.
	A1=((res-1.127*sbk*nn2)/vas1)/2.
	A0 = (1-sst.spearmanr(vas,var)[0])


	ou1=open('feedme_exp.r','w')
	ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00   1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n\n # Object number: X MAIN SOURCE DISK \n 0) expdisk \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3)  %f \t 1 \t\t #  total   magnitude \n 4)  %f \t 1 \t\t#  Rs[pix] \n 9) %f \t 1 \t\t #  axis ratio (b/a) \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'%('bcg_r.fits','ajust-bcg-exp.fits','sigma-r.fits','bcg_r_psf_b.fits','bcg_r_mask.fits',1,ajust1.shape[0],1,ajust1.shape[1],zeropoints,xc,yc,mag,re/4.,axis,pa,xc,yc,magb,re,axis,pa))
	ou1.close()

	call('./galfit feedme_exp.r',shell=True)

	if os.path.isfile('ajust-bcg-exp.fits'):
		data_exp = pyfits.getdata('ajust-bcg-exp.fits',1)
		header_exp = pyfits.getheader('ajust-bcg-exp.fits',2)
	
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

		par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(cluster,chi2,re,mag,n,axis,pa,float(rff),float(A0),float(A1),delta,gama,flagdelta,flagmask,chi2_exp,re_exp,mag_exp,n_exp,axis_exp,pa_exp,rs_exp,magd_exp,axisd_exp,pad_exp))
		par.close()
		
		info_err.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(cluster,re_err,mag_err,n_err,axis_err,pa_err,re_err_exp,mag_err_exp,n_err_exp,axis_err_exp,pa_err_exp,rs_err_exp,magd_err_exp,axisd_err_exp,pad_err_exp))

	else:
		par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t fail \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \n'%(cluster,chi2,re,mag,n,axis,pa,float(rff),float(A0),float(A1),delta,gama,flagdelta,flagmask))
		par.close()

		info_err.write('%s \t %f \t %f \t %f \t %f \t %f \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0\n'%(cluster,re_err,mag_err,n_err,axis_err,pa_err))
	call('rm bcg_r.fits',shell=True)
	call('rm sigma-r.fits',shell=True)
	call('rm galfit.*',shell=True)
	call('rm bcg_r_psf.fits',shell=True)
	call('rm check*',shell=True)
	os.chdir('../')
	return

######################################################################

ok=[]
with open('pargal_L07_compact.dat','r') as inp2:
	for item in inp2.readlines():	
		ok.append(item.split()[0])
with open('data_indiv_ecd.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('data_indiv_ecd.dat','r')
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
		pass
	else:
		#call('rm -rf /mnt/Dados1/andre/WHL/'+str(cluster),shell=True)
		#call('rmdir /mnt/Dados1/andre/WHL/'+str(cluster),shell=True)
		#downframes(str(ll1[0]),int(ll1[2]),int(ll1[3]),int(ll1[4]))
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

		header = pyfits.getheader(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',0)
		data = pyfits.getdata(cluster+'/frame-'+band+'-'+run+'-'+camcol+'-'+field+'.fits',0)
	
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

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10a=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] == yb:
					data10a[i,j] = 0 
				elif data10a[i,j] == 0:
					data10a[i,j] = 0 
				elif data10a[i,j] != yb and data10a[i,j] != 0:
					data10a[i,j] = 1

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11a=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11a[i,j] == xb:
					data11a[i,j] = 0 
				elif data11a[i,j] == 0:
					data11a[i,j] = 0 
				elif data11a[i,j] != xb and data11a[i,j] != 0:
					data11a[i,j] = 1

		mask_miss=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] != 0 or data11a[i,j] != 0:
					mask_miss[i,j] = 1
				else:
					mask_miss[i,j] = 0
###################################################
	# mask da bcg 

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10b=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] == yb:
					data10b[i,j] = 1 
				elif data10b[i,j] == 0:
					data10b[i,j] = 0 
				elif data10b[i,j] != yb and data10b[i,j] != 0:
					data10b[i,j] = 0

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11b=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11b[i,j] == xb:
					data11b[i,j] = 1 
				elif data11b[i,j] == 0:
					data11b[i,j] = 0 
				elif data11b[i,j] != xb and data11b[i,j] != 0:
					data11b[i,j] = 0

		mask_dirty=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] != 0 or data11b[i,j] != 0:
					mask_dirty[i,j] = 1
				else:
					mask_dirty[i,j] = 0 

		#MASK CLEANER
		
		mask_b=copy.deepcopy(mask_dirty)
		coords_distances=[[],[],[]]
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if mask_dirty[i,j] ==1:
					coords_distances[0].append(i)
					coords_distances[1].append(j)
					coords_distances[2].append(np.sqrt((i-yc)**2+(j-xc)**2))
		contiguos=[[int(yc)],[int(xc)]]
		sorteddists=sorted(coords_distances[2])
		sorteddists.append(sorteddists[-1]+1)
		for ii in range(len(sorteddists)):
			for i in range(len(coords_distances[0])):
				if sorteddists[ii]<=coords_distances[2][i]<sorteddists[ii+1]:
					ypix=coords_distances[0][i]
					xpix=coords_distances[1][i]
					laterals=[]
					for k in range(-1,2):
						for kk in range(-1,2):
							laterals.append((ypix+k,xpix+kk))
					contiguos_array=[]
					for k in range(len(contiguos[0])):
						contiguos_array.append((contiguos[0][k],contiguos[1][k]))
					if len(set(laterals) & set(contiguos_array))>0:
						contiguos[0].append(ypix)
						contiguos[1].append(xpix)
						mask_b[ypix,xpix]=1
					else:
						mask_b[ypix,xpix]=0
		pyfits.writeto(cluster+'/bcg_'+band+'_mask_b.fits',mask_b,header=header,clobber=True)
		
		mask=copy.deepcopy(mask_miss)
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if mask_b[i,j] == 0 and mask_dirty[i,j] == 1:
					mask[i,j] = 1
				else:
					mask[i,j] = mask_miss[i,j]
		pyfits.writeto(cluster+'/bcg_'+band+'_mask.fits',mask,header=header,clobber=True)
########################################################## 
# inicio do sigma.fits:

		header.append(card='GAIN',useblanks=True, bottom=False, end=False)
		header['GAIN'] = gain(camcol,cband,run)
	
		sigma = copy.deepcopy(data[ninfy:nsupy,ninfx:nsupx])
		sig = ((sigma*EXPTIME)/NMGY)*gain(camcol,cband,run)	

		vec = []
		for i in range(0,nsupy-ninfy):
			for o in range(0,nsupx-ninfx):
				if mask[i,o] == 0 and mask_b[i,o] == 0 : 
					vec.append(sig[i,o]) 	
		z = np.sum(np.power(vec-np.average(vec),2))/(len(vec)-1.)
		sig1 = (np.sqrt(np.absolute(sig)+z))/gain(camcol,cband,run)
		pyfits.writeto(cluster+'/sigma-'+band+'.fits',sig1,header=header,clobber=True)

########################################################
# imagem da bcg usada pelo galfit, == ajust_..[1]
		data1 = copy.deepcopy(data[ninfy:nsupy,ninfx:nsupx])
		data2 = (((data1-subt)*EXPTIME)/NMGY)
		pyfits.writeto(cluster+'/bcg_'+band+'.fits',data2,header=header,clobber=True)

##########################################################
# prepare psf file

		call('./read_PSF '+cluster+'/psField-'+str(run).zfill(6)+'-'+str(camcol)+'-'+str(field).zfill(4)+'.fit '+str(cband+1)+' '+str(AX)+' '+str(AY)+' '+cluster+'/bcg_'+band+'_psf.fits',shell=True)
	
		headerpsf = pyfits.getheader(cluster+'/bcg_'+band+'_psf.fits',0)
		datapsf = pyfits.getdata(cluster+'/bcg_'+band+'_psf.fits',0)
		datapsf=np.subtract(datapsf,1000.)
		pyfits.writeto(cluster+'/bcg_'+band+'_psf_b.fits',datapsf,clobber=True)

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
		ou1.write('# IMAGE and GALFIT CONTROL PARAMETERS \n A) %s \t\t # Input data image (FITS file) \n B) %s \t\t # Output data image block \n C) %s \t\t # Sigma image name (made from data if blank or "none") \n D) %s \t\t # Input PSF image and (optional) diffusion kernel \n E) 1 \t\t # PSF fine sampling factor relative to data \n F) %s \t\t # Bad pixel mask (FITS image or ASCII coord list) \n G) constr.all \t\t # File with parameter constraints (ASCII file) \n H) %i %i %i %i  \t\t # Image region to fit (xmin xmax ymin ymax) \n I) 117   117 \t\t # Size of the convolution box (x y) \n J) %f \t\t # Magnitude photometric zeropoint \n K) 0.396127  0.396127 \t\t # Plate scale (dx dy)    [arcsec per pixel] \n O) regular \t\t # Display type (regular, curses, both) \n P) 0 \t\t # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n\n\n # Object number: X MAIN SOURCE - BCG \n 0) sersic \t\t #  object type \n 1) %i  %i  1 1 \t\t #  position x, y \n 3) %f     1 \t\t #  Integrated magnitude \n 4)  %f    1 \t\t #  R_e (half-light radius)   [pix] \n 5)  4.00    1 \t\t #  Sersic index n (de Vaucouleurs n=4) \n 9) %f     1 \t\t #  axis ratio (b/a)  \n 10)  %f   1 \t\t #  position angle (PA) [deg: Up=0, Left=90] \n Z) 0 \t\t #  output option (0 = resid., 1 = Do not subtract) \n # sky \n\n\n 0) sky \n  1) 0.00 \t\t 1 \t\t # sky background \t\t  [ADU counts] \n 2) 0.000 \t\t 0 \t\t # dsky/dx (sky gradient in x) \n 3) 0.000 \t\t 0 \t\t # dsky/dy (sky gradient in y) \n Z) 0 \t\t\t\t #  Skip this model in output image?  (yes=1, no=0)'% ('bcg_'+band+'.fits','ajust-bcg-'+band+'.fits','sigma-'+band+'.fits','bcg_'+band+'_psf_b.fits','bcg_'+band+'_mask.fits',1,nsupx-ninfx,1,nsupy-ninfy,zeropoints,xc,yc,mag,siz/4.5,razax,thetabcg))
		ou1.close()
		ou3.close()
		lnt(ll1[0],xc,yc)
###########################
