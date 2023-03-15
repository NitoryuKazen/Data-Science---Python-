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
 
#####################################

def cutaster(x):
 tempx=''
 for item in x:
 	if item != '*':
 		tempx+=item
 return tempx

###########################################

def lnt(cluster,X,Y):

#PROGRAMA PARTE 1 

	par = open('pargal_L07_clean.dat','a')
	info_err=open('aux_pargal_L07_clean.dat','a')
	
	os.chdir(cluster+'/')
########

	ajust1 = pyfits.getdata('ajust-bcg-r.fits',1)
	ajust3 = pyfits.getdata('ajust-bcg-r.fits',3)
	mask = pyfits.getdata('bcg_r_mask.fits')
	mask_b = pyfits.getdata('bcg_r_mask_b.fits')
	header = pyfits.getheader('ajust-bcg-r.fits',2)
 
#########

	re = header['1_RE']
	mag = header['1_MAG']
	n = header['1_N']
	axis = header['1_AR']
	pa = header['1_PA']
	chi2 = float(header['CHI2NU'])
	xc=header['1_XC']
	yc=header['1_YC']
	imsec=(header['FITSECT'])
	zeropoints=float(header['MAGZPT'])
#########

  # prefixo 1 significa valor encontrado e 2 siginifica a incerteza desse valor
	inc1 = re.split()
	inc2 = mag.split()
	inc3 = n.split()
	inc4 = axis.split()
	inc5 = pa.split()
	inc11=xc.split()
	inc22=yc.split()
	re1 = float(cutaster(inc1[0]))
	re2 = float(cutaster(inc1[2]))
	mag1 = float(cutaster(inc2[0]))
	mag2 = float(cutaster(inc2[2]))
	axis1 = float(cutaster(inc4[0]))
	axis2 = float(cutaster(inc4[2]))
	pa1 = float(cutaster(inc5[0]))
	pa2 = float(cutaster(inc5[2]))
	n1 = float(cutaster(inc3[0]))
	n2 = float(cutaster(inc3[2]))
	xc1=float(cutaster(inc11[0]))
	yc1=float(cutaster(inc22[0]))
	
	inc3x=imsec.split()[0].split(',')[0].split(':')[1]
	inc3y=imsec.split()[0].split(',')[1].split(':')[1].split(']')[0]
##############
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
	xx=0
	for i in range(0,ajust1.shape[0]): 
		for j in range(0,ajust1.shape[1]): 
			nn2+=(1.-mask[i,j])*(mask_b[i,j])
			xy+=np.absolute((ajust3[i,j])*((1.-mask[i,j])*(mask_b[i,j])))
			xn+=(ajust1[i,j])*((1.-mask[i,j])*(mask_b[i,j]))
		#	xx+=(np.power((ajust3[i,j])*((1.-mask[i,j])*(mask_b[i,j]))/(sbk*((1.-mask[i,j])*(mask_b[i,j]))),2))
	rff=(xy-0.8*sbk*nn2)/xn
	#chi=0.1*(xx)	
	print xy, sbk, sbk*nn2, nn2, xn, rff, xx	#################################																																																																										

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
	assim=[]
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
		data34=data30 - data30r
#############
		# A0
		vas=[]#bcg normal
		var=[]#bcg rotacionada
		test=[]
		for i in range(0,yyc2-yyc):
			for j in range(0,xxc2-xxc):
				if data30[i,j] != 0.0 and data30r[i,j] != 0.0:
					test.append(data30[i,j])
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
	data34=data30 - data30r
	print(Y+dyb)-ajust1.shape[0]/2.
	print(X+dxb)-ajust1.shape[1]/2.
	print dyb,dxb
	print ajust1.shape[0],ajust1.shape[1]
	print data30.shape[0],data30.shape[1]

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
				test.append(data30[i,j])
				if data32[i,j] == 0 and data32r[i,j] == 0 and (data31[i,j] == 1  or data31r[i,j] == 1):
					vas.append(data30[i,j])
					var.append(data30r[i,j])
					datat[i,j]=1
					datatr[i,j]=1
					res+=np.absolute(data30[i,j]-data30r[i,j])
					vas1+=(data30[i,j])
					nn2+=1.
	A1=((res-1.127*sbk*nn2)/vas1)/2.
	print res, vas1, A1, sbk, nn2
	print len(var), len(vas)
	A0 = (1-sst.spearmanr(vas,var)[0])
	assim.append(float(A0))
	assim.append(float(dxb))
	assim.append(float(dyb))
	assim.append(float((Y+dyb)-ajust1.shape[0]/2.))
	assim.append(float((Y+dxb)-ajust1.shape[1]/2.))
	assim.append(float(A1))
	pyfits.writeto('res_assim2.fits',data34,clobber=True)		
	print assim, A0

	par.write('%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n'%(ll1[0],chi2,re1,mag1,n1,axis1,pa1,float(rff),float(assim[0]),float(assim[5])))
	par.close()
	info_err.write('%s \t %f \t %f \t %f \t %f \t %f \tr %f \t %f \n'%(ll1[0],re2,mag2,n2,axis2,pa2,float(assim[3]),float(assim[4])))
	os.chdir('../')
	return

'''
HEADER DE TUDO QUE SAI DO ARQUIVO pargal_*.dat aux_pargal_*.dat (RESPECTIVAMENTE):

0-ID
1-CHI2
2-RAIO EFETIVO
3-MAGNITUDE
4-N DE SERSIC
5-AXIS
6-ANGULO
7-RFF
8-ASSIMETRIA(SPEARMAN)
9-ASSIMETRIA(ABRAHAM)

0-ID
1-RAIO EFETIVO(INCERTEZA)
2-MAGNITUDE(INCERTEZA)
3-N DE SERSIC(INCERTEZA)
4-AXIS(INCERTEZA)
5-ANGULO(INCERTEZA)
6-DXB(INTERPOLATION)
7-DYB(INTERPOLATION)
'''
#####################################################################

#AQUI E O PROGRAMA QUE PRODUZ FIGURAS DE 3 A 3. PODE COMENTAR ELE SE QUISER. 

def mkf(cluster):

	os.chdir(cluster+'/')	

############################

	input_image='ajust-bcg-r.fits'
	input_image_2='bcg_r_mask.fits'
	input_image_3='bcg_r_mask_b.fits'
	input_image_4='sigma-r.fits'	

###############################

	data1 = pyfits.getdata(input_image,1)
	data2 = pyfits.getdata(input_image,2)
	data3 = pyfits.getdata(input_image,3)

	data4 = pyfits.getdata(input_image_2)
	data5 = pyfits.getdata(input_image_3)
	data6 = pyfits.getdata(input_image_4)

#############################

	data1b=np.power(abs(data1),0.25)
	data2b=np.power(abs(data2),0.25)
	data3b=np.power(abs(data3),0.25)
	
	data4b=np.power(abs(data4),0.25)
	data5b=np.power(abs(data5),0.25)
	data6b=np.power(abs(data6),0.25)

	vmin=np.percentile(data2b,60)
	vmax=np.percentile(data2b,99.99)

###################################	
# ajuste da imagem da bcg original

	fig1=plt.figure(figsize=(12,4))
	f1=plt.subplot(131)
	f1.set_title('Original')
	f1.tick_params(axis='x', labelsize=8)
	f1.tick_params(axis='y', labelsize=8)
	plt.imshow(data1b,origin="lower",cmap = cm.Greys_r).set_clim(vmin,vmax)
	f2=plt.subplot(132)
	f2.set_title('Modelo')
	f2.tick_params(axis='x', labelsize=8)
	f2.tick_params(axis='y', labelsize=8)
	plt.imshow(data2b,origin="lower",cmap = cm.Greys_r).set_clim(vmin,vmax)
	f3=plt.subplot(133)
	f3.set_title('Residuos')
	f3.tick_params(axis='x', labelsize=8)
	f3.tick_params(axis='y', labelsize=8)
	plt.imshow(data3b,origin="lower",cmap = cm.Greys_r).set_clim(vmin,vmax)
	fig1.tight_layout()
	fig1.savefig(ll1[0]+'1.eps')
	plt.close(fig1)
	
################################
# imagens distintas : mascara a e b, e sigma

	fig2=plt.figure(figsize=(12,4))
	f4=plt.subplot(131)
	f4.set_title('Mascara A')
	f4.tick_params(axis='x', labelsize=8)
	f4.tick_params(axis='y', labelsize=8)
	plt.imshow(data4b,origin="lower",cmap = cm.Greys_r).set_clim(0,1)
	f5=plt.subplot(132)
	f5.set_title('Mascara B')
	f5.tick_params(axis='x', labelsize=8)
	f5.tick_params(axis='y', labelsize=8)
	plt.imshow(data5b,origin="lower",cmap = cm.Greys_r).set_clim(0,1)
	f6=plt.subplot(133)
	f6.set_title('Sigma')
	f6.tick_params(axis='x', labelsize=8)
	f6.tick_params(axis='y', labelsize=8)
	plt.imshow(data6b,origin="lower",cmap = cm.Greys_r).set_clim(np.percentile(data6b,60),np.percentile(data6b,99.99))
	fig2.tight_layout()
	fig2.savefig(ll1[0]+'2.eps')
	plt.close(fig2)

	call('rm bcg_r.fits',shell=True)
#	call('rm bcg_r_mask.fits',shell=True)
#	call('rm bcg_r_mask_b.fits',shell=True)
	call('rm sigma-r.fits',shell=True)
	call('rm galfit.*',shell=True)
	call('rm bcg_r_psf_b.fits',shell=True)
	os.chdir('../')
######################################################################

ok=[]
out4 = open('testemask.dat','a')
with open('pargal_L07_clean.dat','r') as inp2:
	for item in inp2.readlines():	
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
	#	pass
	#else:
		print datetime.datetime.now(), ik, cluster
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
		print siz, xc , yc, infx, infy, supx, supy
		
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

		print 'centro bcg em x',xc
		print 'centro bcg em y',yc
		print 'ninfx',ninfx
		print 'ninfy',ninfy 
		print 'nsupx',nsupx
		print 'nsupy',nsupy
		print siz+infy 
		
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
			elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
				ll2[1]=str(siz/10.)
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
	
		# primeira parte

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10a=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		with open(cluster+'/out_sex_small.cat','r') as infa:
		  ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_small.cat','r')
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] == yb:
					data10a[i,j] = 0 
				elif data10a[i,j] == 0:
					data10a[i,j] = 0 
				elif data10a[i,j] != yb and data10a[i,j] != 0:
					data10a[i,j] = 1
		infa.close()
#######
		# segunda parte

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11a=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		with open(cluster+'/out_sex_large.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_large.cat','r')
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11a[i,j] == xb:
					data11a[i,j] = 0 
				elif data11a[i,j] == 0:
					data11a[i,j] = 0 
				elif data11a[i,j] != xb and data11a[i,j] != 0:
					data11a[i,j] = 1
		infa.close()
#######
		# terceira parte 
		mask=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10a[i,j] != 0 or data11a[i,j] != 0:
					mask[i,j] = 1
				else:
					mask[i,j] = 0
		pyfits.writeto(cluster+'/bcg_'+band+'_mask.fits',mask,header=header,clobber=True)
###################################################
	# mask da bcg 
	# primeira parte  

		cks = pyfits.getdata(cluster+'/check3_small.fits')
		data10b=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		with open(cluster+'/out_sex_small.cat','r') as infa:
		  ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_small.cat','r')
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] == yb:
					data10b[i,j] = 1 
				elif data10b[i,j] == 0:
					data10b[i,j] = 0 
				elif data10b[i,j] != yb and data10b[i,j] != 0:
					data10b[i,j] = 0
		infa.close()
#######
		# segunda parte 

 		ckl = pyfits.getdata(cluster+'/check3_large.fits')
		data11b=copy.deepcopy(ckl[ninfy:nsupy,ninfx:nsupx])
		with open(cluster+'/out_sex_large.cat','r') as infa:
			ninfa=len(infa.readlines())
		infa=open(cluster+'/out_sex_large.cat','r')
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data11b[i,j] == xb:
					data11b[i,j] = 1 
				elif data11b[i,j] == 0:
					data11b[i,j] = 0 
				elif data11b[i,j] != xb and data11b[i,j] != 0:
					data11b[i,j] = 0
		infa.close()
#######
		#terceira parte
	
		mask_b=copy.deepcopy(cks[ninfy:nsupy,ninfx:nsupx])
		for i in range(0,nsupy-ninfy):
			for j in range(0,nsupx-ninfx):
				if data10b[i,j] != 0 or data11b[i,j] != 0:
					mask_b[i,j] = 1
				else:
					mask_b[i,j] = 0 
		pyfits.writeto(cluster+'/bcg_'+band+'_mask_b.fits',mask_b,header=header,clobber=True)

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
		print z
		print len(vec)
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
		if mask[xc,yc] == 0:
			out4.write('%s 0 \n' % cluster)
		else:
			out4.write('%s 1 \n' % cluster)
	
# chama o calculo do rff, remove as imagens.

	
		os.chdir(cluster+'/')
		call('rm check*',shell=True)
		call('rm bcg_r_psf.fits',shell=True)
		call('./galfit feedme.r',shell=True)
		os.chdir('../')
		lnt(ll1[0],xc,yc)
		#mkf(ll1[0])
out4.close()
###########################
