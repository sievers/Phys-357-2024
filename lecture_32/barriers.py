import numpy as np
from matplotlib import pyplot as plt
plt.ion()

fac=1
V0=15*fac
x=np.linspace(-6,6,301)
V=V0*np.exp(-0.5*(fac*x)**2)
a=np.diff(x)

E0=1.0

k=np.sqrt((2+0J)*(V-E0))
mask=np.where(V>E0)
pred=np.prod(np.exp(-2*a[mask]*k[mask]))
print('predicted transmission: ',pred)

#build up a matrix to work out the exact transmissions.
#at every interior boundary, if we define psi to be
#exp (+-k (x-x0)) where x0 is the left edge of the rectangle
#in questions, then psi=a exp(k (x-x0))+b exp(-k(x-x0))
#the psi on our right is  c exp(k'(x-x0'))+ d exp(-k'(x-x0')
#where x0' is the position of the right edge.  x0'-x0=a, the
#width of that rectangle.  That means at the right edge, the
#continuity of psi means a*exp(ka)+b*exp(-ka) = c+d
#and slope continuity is k(a*exp(ka)-b*exp(-ka))=k'(c-d)
#this gives us two equations every time we add another boundary
#so build these up until we get the full set of transmission equations
mat=np.zeros([2*len(x),2*len(x)],dtype='complex')
for i in range(len(x)-1):
    aa=np.exp(k[i]*a[i])
    bb=np.exp(-k[i]*a[i])
    mat[2*i,2*i:(2*i+4)]=np.asarray([aa,bb,1,1])
    mat[2*i+1,(2*i):(2*i+4)]=np.asarray([k[i]*aa,-k[i]*bb,k[i+1],-k[i+1]])

#we have to do a little bit of funny stuff
#at the first/last cells.  At the beginning,
#we have set the incoming amplitude to be 1
#so we only have to solve for the reflected
#we're also going be super sloppy for the
#transmitted part, because a) we can figure out
#the transmitted amplitude once we're anywhere past
#the barrier, and 2) it's hard and I'm lazy.
mm=mat[:-1,1:]
rhs=np.zeros(mm.shape[0],dtype='complex')
rhs[0]=1
rhs[1]=k[0]
mm[-1,-1]=1

#now that we've found the couping between the coefficients,
#we can solve for the amplitudes in every region just by
#inverting the matrix.
psi=np.linalg.inv(mm)@rhs

#we can find the total probability in each region by squaring and adding
#the decaying and growing parts
amps=np.abs(psi[1::2])**2+np.abs(psi[2::2])**2
T=amps[-2]
print('measured transmission: ',T,T/pred)
R=np.abs(psi[0])**2
print('transmission plus reflection: ',T+R)
#plt.clf();plt.plot(x[1:],amps);plt.show()
plt.plot(x[1:],amps)
plt.semilogy()

#now find the current, using psi and psi' at the bin edges
#at edge, psi = a exp(kx) + b exp(-kx), for x at edge=a+b
#gradient is    a k - bk
psi_edge=psi[1::2]+psi[2::2]

psi_grad=(psi[1::2]-psi[2::2])*k[1:]
curr = (np.conj(psi_edge)*psi_grad -np.conj(psi_grad)*psi_edge)/2j
#print('current is ',np.median(curr),np.median(curr)/T)
