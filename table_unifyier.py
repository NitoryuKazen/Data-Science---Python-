###############################################
############################################
with open('pargal_L07_compact_astro.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('pargal_L07_compact_astro.dat','r')

#35 - cModelAbsMag_r
#36 - logMass
#37 - age
#38 - metallicity
#39 - petroR90_r
#40 - petroR50_r

with open('pargal_L07_compact_casjobs_v2.dat','r') as inp4:
	ninp4=len(inp4.readlines())
inp4=open('pargal_L07_compact_casjobs_v2.dat','r')

with open('../isophotest/graph_stats.dat','r') as inp2:
	ninp2=len(inp2.readlines())
inp2=open('../isophotest/graph_stats.dat','r')

with open('data_indiv_ecd.dat','r') as inp3:
	ninp3=len(inp3.readlines())
inp3=open('data_indiv_ecd.dat','r')

graph_rff=open('graph_stats_rff.dat','w')

cluster,rff,eta,tipo,magabs,logmass,age,metal,c90,c50=parinfo=[[] for i in range(10)]#rff,a1-rrf

for ik in range(0,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	ls3=inp3.readline()
	ll3=ls3.split()
	ls4=inp4.readline()
	ll4=ls4.split()
	cluster.append(ll1[0])
	rff.append(float(ll1[7]))
	eta.append(float(ll1[7]) - float(ll1[9]))
	tipo.append(ll3[7])
	magabs.append(float(ll4[35]))
	logmass.append(float(ll4[36]))
	age.append(float(ll4[37]))
	metal.append(float(ll4[38]))
	c90.append(float(ll4[39]))
	c50.append(float(ll4[40]))
	
cl=[]
linha=[]
for j in range(0,ninp2):
	ls2=inp2.readline()
	ll2=ls2.split()
	cl.append(ll2[0])
	linha.append(ls2[:-2])

for ii in range(len(cl)):
	if c50[cluster.index(cl[ii])] == 0.0:
		graph_rff.write('%s %f %f %f %f %f %f %f %s \n'%(linha[ii],magabs[cluster.index(cl[ii])],logmass[cluster.index(cl[ii])],age[cluster.index(cl[ii])],metal[cluster.index(cl[ii])],0.0, rff[cluster.index(cl[ii])],eta[cluster.index(cl[ii])],tipo[cluster.index(cl[ii])]))
	else:
		graph_rff.write('%s %f %f %f %f %f %f %f %s \n'%(linha[ii],magabs[cluster.index(cl[ii])],logmass[cluster.index(cl[ii])],age[cluster.index(cl[ii])],metal[cluster.index(cl[ii])], c90[cluster.index(cl[ii])]/c50[cluster.index(cl[ii])], rff[cluster.index(cl[ii])],eta[cluster.index(cl[ii])],tipo[cluster.index(cl[ii])]))
