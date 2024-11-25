import numpy as np
from matplotlib import pyplot as plt
plt.ion()

V0=1.0
Evec=np.linspace(V0,3*V0,301)
Evec=Evec[1:] #we have to be more careful for E=V0 exactly
TT=0*Evec
sig=3.0
x=np.linspace(-6,6,301)

#we want 2ka=2n pi for perfect transmission.
#critical waves are then k=n pi /a (which is 2*sig the way we set
#things up).  We're using k here since this only happens
#for barriers where E>V, as the perfect tranmissions requires that
#Psi oscillates inside the barrier.  Once we have that, we can
#calculate the k's and hence the E's at which we expect 100% transmission
#through the barrier.
kcrit=np.arange(1,4)*np.pi/(2*sig)
#With h,m=1, then E_k=p^2/2m = k^2/2.  E_tot=
#k^2/2+V_0. If you want to see transmission, you should make
#sure to calculate T for energies around these values.  Generally,
#the fatter the barrier, the longer the wave can be to fit one period
#inside the barrier, and the more fun this plot looks.
Ecrit=V0+kcrit**2/2
print("critical Energies for barrier of width "+repr(2*sig)+" and height "+repr(V0)+" are:")
print(Ecrit)

#loop over incoming energyies and see what the tranmission fraction looks like
for iii in range(len(Evec)):
    E_0=Evec[iii]
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
    #print('transmitted fraction: ',T)
    TT[iii]=T
    
plt.clf()
plt.plot(Evec,TT)
plt.plot(Ecrit,0*Ecrit+1,'*')
plt.ylim([0,1.01])
plt.xlabel("E incident")
plt.ylabel("Transmission Fraction")
plt.title("Tranmission vs. Energy")
        
