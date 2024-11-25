import numpy as np

x=np.linspace(-2,2,301)
V0=0.3
sig=0.75
E_0=1.0
#V=V0*np.exp(-0.5*x**2/sig**2) #a gaussian barrier
V=0.0+V0*(np.abs(x)<sig)  #a rectangular barrier

#get the length of each of our barriers
a=np.diff(x)
q=np.sqrt((2+0J)*(V-E_0)) #we add the 0J so the square root is happy for imag
q_final=q[0] #we're assuming the final potential is the same as the
#initial potential, but it doesn't have to be.
#get the number of barriers. 
nb=len(a)

gamma=np.exp(q[1:]*a)

#we have 2nb interior paramters, one reflected amplitude on the left
#and one transmitted amplitude on the right
n=2*nb+2

#first, do the right-hand side
rhs=np.zeros(n,dtype='complex')
t0=1.0 #because why not
rhs[0]=t0
rhs[1]=t0*q[0]

#now let's fill in our matrix
mat=np.zeros([n,n],dtype='complex')

#first two equations are slightly different since we aren't solving for t0
mat[0,:3]=np.asarray([-1,1,1])
mat[1,:3]=np.asarray([q[0],q[1],-q[1]])

#as are the last two.  if we're sending off into the same potential as we
#started with, then the q for the transmitted region is the
#same as the inital q, so plug that in.  If we wanted something else,
#we could put it in here
mat[-2,-3:]=np.asarray([gamma[-1],1/gamma[-1],-1])
mat[-1,-3:]=np.asarray([q[-1]*gamma[-1],-q[-1]/gamma[-1],-q_final])

#finally, loop over interior barriers:
for i in range(nb-1):
    mat[2*i+2,(2*i+1):(2*i+5)]=np.asarray([gamma[i],1/gamma[i],-1,-1])
    mat[2*i+3,(2*i+1):(2*i+5)]=np.asarray([q[i+1]*gamma[i],-q[i+1]/gamma[i],-q[i+2],q[i+2]])

#get the wave function amplitudes everywhere
#by inverting our matrix, and multiplying by the RHS
amps=np.linalg.inv(mat)@rhs
#and get the transmission - this is the last amplitude divided by the
#incoming ampmlitude, which is then squared to get the transmission
#probability.
T=np.abs(amps[-1])**2/t0**2
print('transmitted fraction: ',T)


#We'll do the approximate solution where we assume each
#piece where E<V decays like exp(-qa), and see how close
#we are to the exact solution.
mask=np.where(V>E_0)
pred=np.prod(np.exp(-q[mask]*a[mask]))
pred=np.abs(pred)**2/t0**2

print('prediction: ',pred)
print('ratio of truth to prediction: ',T/pred)
