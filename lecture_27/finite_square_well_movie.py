import numpy as np
from matplotlib import pyplot as plt
plt.ion()
#script to plot movies of the finite square well solutions.
#we'll define the potential to be V for x<0 and x>a, and
#0 for 0<x<a.
#We know the solution to x<0 has to be exp(alpha x) where
#alpha is set by sqrt(V-E).  It can't have any exp(-alpha x)
#because then psi would diverge as x->-infinity.
#For now, we'll define the wave function to the left of the well
#to be exp(alpha x) - don't worry, we can always normalize properly
#one we have an unnormalized but otherwise valid wave function.
#In the interior, we know that the wave function is sines/cosines
#with alpha' set by sqrt(E).  If psi in the middle is c_1 cos(alpha' x)
#+c_2 sin(alpha' x), then we have to pick c_1 and c_2 to match the
#value/derivative of the wave function in the left region.  That
#uniquely defines both c_1 and c_2.  For the region to the right
#of the well, the solution has to be c_r exp(-alpha x).  We
#have one parameter to adjust that has to match both a value and
#slope, and in general we can't do that except for carefully chosen
#valued of E.  In the movie this script makes, loop over a range of
#energies, and set c_r to make psi continuous.  You'll see that in general
#there is a kink in the slope at the right edge of the well, except
#for a discrete set of energies.  

#potential outside the region
V=200
#pick a set of energies to plot our
#wave function.  since period depends on
#sqrt(energy), I make the energy levels
#equally spaced in energy squared.  This is purely
#to keep the first part of the movie from being
#extremely boring, but you could asbolutely just
#do E=np.linspace(0,V,<large number>).
#incidentally, we stop at E=V because otherwise we have sines/cosines
#outside the well instead of exponentials, and our state is no longer
#bound.
E=V*(np.linspace(0,0.9,3001)**2)

#get the exponential factors/wave vectors ouside and inside the well.
a_out=np.sqrt(V-E)
a_in=np.sqrt(E)
a=2 #pick a width for the well.  


#set up x values for the left, middle, and right regions
#where middle is the region inside the well.
xl=np.linspace(-4,0,2001)
dx=xl[2]-xl[1]
xm=np.arange(0,a+dx,dx)
xr=np.linspace(a,a+4,2001)

for i in range(1,len(E)-1): #cut off first/last points
    #we said we would force the wave function in the left region go to 1
    #at x=0, which makes the math easy.
    psi_l=np.exp(xl*a_out[i])
    slope=a_out[i]  #this is the slope of the exponential at x=0
    #need to find  c1 cos(a_in[i]x)+c2 sin(a_in[i]x)
    #that matches value and slope of a_out
    #at x=0, c1 must be 1 since
    #exp(a_out x)=1 at x=0.  Deriv is a_out exp()=a_out
    #so deriv of inside, which is c2 a_in cos (0) (+sin that evaluates to 0)
    #so a_out=c2 a_in, c2=a_out/a_in
    c1=1
    c2=a_out[i]/a_in[i]
    psi_mid=c1*np.cos(xm*a_in[i])+c2*np.sin(xm*a_in[i])
    #get the wave function at x=a.  the decaying exponential
    #must match.  We won't in general be able to match
    #both slope and height because we can only set one term
    #for the decaying exponential on the right.  
    psi_end=c1*np.cos(a*a_in[i])+c2*np.sin(a*a_in[i])
    #if we define the right hand psi to be c_r*exp(-(x-a)*a_out)
    #then at x=a, the value is c_r, and so c_r=psi_end.  This will
    #give us a continuous wave function, but in general there will be
    #kinks in the slope.  That's not allowed by the Schrodinger
    #equation, so only valid solutions will be where the slope
    #is continuous
    psi_r=np.exp(-(xr-a)*a_out[i])*psi_end

    #now that we have our wave function, plot away.
    plt.clf()
    plt.plot(xl,psi_l)
    plt.plot(xm,psi_mid)
    plt.plot(xr,psi_r)
    plt.title('E= {:8.4}'.format(E[i]))
    plt.pause(0.001)
    
    
