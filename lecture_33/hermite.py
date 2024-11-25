import numpy as np
#code to calculate Hermite polynomials, which arise in calculating
#the quantum SHO.  The Nth eigenstate of the SHO is
#H_n exp(-x^2/2)
#if we want to normalize, integral of H_n^2 exp is sqrt(pi) 2**n n!, so
#we'd divide by the square root of that

def update_p(p):
    #recurrence relation comes from H_n+1 exp(-x^2/2)=(x-d/dx)H_n exp(-x^2/2)
    #this equals 2x H_n exp() - (H_n)' exp()
    #to get the next polynomial part, multiply H_n by 2 and shift right by one
    #then to get the -H_n' part, we multiply H_n,k by k, and shift left by one
    #(since derivative of e.g. x^3 is 3x^2, so we multiply by 3, then
    # shift x left by one to turn an x^3 into an x^2)
    #shifting left by one is equivalent to dividing by x, as long as the
    #index is not zero.
    p_new=0*p
    p_new[1:]=2*p[:-1] #the 2x H_n term
    vec=np.arange(len(p))
    p_new[:-1]=p_new[:-1]-vec[1:]*p[1:] #the -kH_n/x term
    return p_new

def print_p(p):
    #slightly convoluted code to print out the polynomials
    #in human-readable form.  There's way more detail here
    #than is actually appropriate, but hey, they look pretty...
    #note - the coefficients are increasing in powers of x, but we're used
    #to seeing them printed in decreasing order, which is why the
    #loop decreases i instead of increasing it.
    n=len(p)
    mystr=''
    first=True
    for i in range(n-1,-1,-1):
        if i==0:
            xstr=''
        elif i==1:
            xstr='x'
        else:
            xstr='x^'+repr(i)
        if p[i]>0:
            if first:
                mystr=repr(p[i])+xstr
                first=False
            else:
                mystr=mystr+' +'+repr(p[i])+xstr
        elif p[i]<0:
            mystr=mystr+' '+repr(p[i])+xstr
    return(mystr)

n=20
p_org=np.zeros(n,dtype='int')
p_org[0]=1
p=p_org.copy()
for i in range(6):
    print('H_'+repr(i),'= ',print_p(p))
    p=update_p(p)
