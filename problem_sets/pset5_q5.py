import numpy as np
np.set_printoptions(precision=4) #settings to make arrays easier to read
np.set_printoptions(suppress=True)

j=2
m=np.arange(j-1,-j-0.5,-1) #the non-zero values for m in the raising operator
#start by making J+
vec=np.sqrt(j*(j+1)-m*(m+1))
jp=np.diag(vec,1)
jm=jp.conj().T #lowering is adjoint of raising


#if we're working in the z-basis, then
#Jp=Jx+iJy, Jm=Jx-iJy, and so Jy=(Jp-Jm)/2i
jy=(jp-jm)/2J
ey,vy=np.linalg.eigh(jy)
ind=-1  #which eigenvalue we're using.  can change to e.g. -1 to see
        #uncertainty relation go from inequality to equality
print('eigenvector  ',vy[:,ind],' for Jy has eigenvalue ',ey[ind])

Psi=vy[:,ind]

jz=np.diag(np.arange(j,-j-0.5,-1))
jx=(jp+jm)/2

zmean=Psi.conj()@jz@Psi
zsqr=Psi.conj()@jz@jz@Psi
zerr=np.sqrt(zsqr-zmean*zmean)

xmean=Psi.conj()@jx@Psi
xsqr=Psi.conj()@jx@jx@Psi
xerr=np.sqrt(xsqr-xmean*xmean)
print('x,z uncertainties: ',xerr,zerr)
print('unc(x)*unc(z),ymean/2: ',xerr*zerr,ey[ind]/2)
