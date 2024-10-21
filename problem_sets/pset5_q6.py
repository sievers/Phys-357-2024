import numpy as np
np.set_printoptions(precision=4) #settings to make arrays easier to read
np.set_printoptions(suppress=True)

#import code from q5 that sets up our matrices
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

#starting here - R=exp(-i J*th/hbar).  we haven't put the
#hbars in J, so don't need to divide
th=np.pi/2
Ry=vy@np.diag(np.exp(-1J*ey*th))@(vy.conj().T)
print("Ry abs:")
print(np.abs(Ry))

ex,vx=np.linalg.eigh(jx)

#make |2,2>_z state
Psi=np.zeros(2*j+1)
Psi[0]=1

Psi_new=Ry@Psi
print('inner product of Ry@|2,2>_z with |2,2>_x is: ',np.conj(vx[:,-1])@Psi_new) #given we haven't re-ordered the eigenvalues, vx[:,-1] is the |2,2>_x state
print('Ry@|2,2>_x is ',Ry@vx[:,-1]) #this comes out to be proportional to |2,-2>_z

