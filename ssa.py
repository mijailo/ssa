#Este scrypt transforma los datos de un archivo y los convierte en un arreglo de numpy
import numpy as np
import gc ##Para liberar memoria (garbage collector)

def ssa(Serie,N,L):	#Rutina que hace descomposicion en componentes principales de una Serie con N puntos y dimension de encaje L

	K=N-L+1
	ind=[]
	for i in range(0,N):
		ind.append(np.array([[i-j,j] for j in xrange(max(0,i-K+1),min(i+1,L))]))

	X=np.empty([K,L])	#Para la matriz de trayectoria X de dimension KxL
	for m in range(0,K):
		for n in range(0,L):
			X[m,n]=Serie[np.remainder(m+n,N)]

	U,s,Vt=np.linalg.svd(X,full_matrices=True)	#Descomposicion en valores singulares
	Y=np.empty([min(K,L),K,L])	#Para hacer la descomposicion del espacio fase en subespacios ortogonales Y
	g=np.empty([min(K,L),N])	#Habra un total de min(K,L) componentes reconstruidas de la serie

	for m in range(0,min(K,L)):
		Y[m]=s[m]*np.dot(np.reshape(U[:,m],[K,1]),np.reshape(Vt[m,:],[1,L]))

	for m in range(0,min(K,L)):
		for i in range(0,N):
			g[m,i]=np.mean(np.take(Y[m],[x[0]*L+x[1] for x in ind[i]]))

	return g
	gc.collect()	#se obtiene una matriz de min(K,L) por N, donde cada renglon es una componente principal

def autocorr(Serie):	#obtiene la funcion de autocorrelacion C de una Serie, que se anula en L (salidas: [C,L]) (no usa Wiener Khinchin)
	C = np.correlate(Serie, Serie, mode='full')
	C=C[C.size/2:]
	C=C/np.abs(C[0]-np.mean(C))
	C=C-np.mean(C)
	return [C,np.argmin(C>0.)]
