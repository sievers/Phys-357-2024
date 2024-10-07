import numpy as np
#some options to make the printing of matrices prettier
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
def de_angle(mat):
    #handy routine to make the first row real
    #for a matrix.  You're allowed to do this with
    #matrices of eigenvectors
    mm=mat.copy()
    for i in range(mm.shape[1]):
        mm[:,i]=mm[:,i]/np.exp(1J*np.angle(mm[0,i]))
    return mm

#we'll ignore factors of hbar here since they are
#guaranteed to cancel
#write out our angular momentum operators in the z-basis that we
#worked out from problem 4
Jz=np.diag([1,0,-1])
Jx=np.asarray([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
Jy=np.asarray([[0,-1J,0],[1J,0,-1J],[0,1J,0]])/np.sqrt(2)

#calculate the commutators
JxJy=Jx@Jy-Jy@Jx
JyJz=Jy@Jz-Jz@Jy
JzJx=Jz@Jx-Jx@Jz
#these should all be zero if JxJy=ihJz etc.
print("question 5 answer check:")
print('error in [Jx,Jy]=ihJz: ',np.sum(np.abs(JxJy-1J*Jz)))
print('error in [Jy,Jz]=ihJx: ',np.sum(np.abs(JyJz-1J*Jx)))
print('error in [Jz,Jx]=ihJy: ',np.sum(np.abs(JzJx-1J*Jy)))

#now do q6
print('')
print('Answers to q6:')
#get x eigenvalues/eigenvectors
ex,vx=np.linalg.eigh(Jx)
#by default, eigh sorts in increasing order.  We've been
#working with J decreasing, so flip the order to make it
#consistent
ex=np.flipud(ex)
vx=np.fliplr(vx)
vx=de_angle(vx)
print('Eigenvectors of Jx:')
print(vx)

#repeat for Jy
ey,vy=np.linalg.eigh(Jy)
ey=np.flipud(ey)
vy=np.fliplr(vy)
vy=de_angle(vy)
print('Eigenvectors of Jy:')
print(vy)

#raising operator for Jx
Jxp=Jy+1J*Jz
Jyp=Jz+1J*Jx #and Jy
print('Raising operator for Jx:')
print(Jxp)
print('Raising operator for Jy:')
print(Jyp)

vx_raised=Jxp@vx

#when you look at these, you'll see that the first column
# goes away, 
print('Raised x eigenstates: ')
print(vx_raised)


vy_raised=Jyp@vy
print('raised y eigenstates:')
print(vy_raised)

Jxm=Jxp.conj().T
Jym=Jyp.conj().T
vx_lowered=Jxm@vx
vy_lowered=Jym@vy

#so - we want the raising operators operating on the eigenstates
#to return eigenstates with m+1, except for m=j, in which case we get 0.
#however, they are not normalized, so we have to be a little careful
#when we check.  Jx times raised states should give us (m+1) times raised
#states, so what we'll check is that Jx@vx_raised is (0,j,j-1,...) times
#vx_raised.  for the case of spin-1, that diagonal matrix has entries
#(0,1,0) where the first zero is because the raising operator kills the
#m=j state, raises m=0 to m=1, and raises m=-1 to m=0.
raised_eigs=np.asarray([0,1,0])
raised_errs=np.zeros(3)
for i in range(3):
    Jxvx=Jx@vx_raised[:,i]
    raised_errs[i]=np.sum(np.abs(Jxvx-raised_eigs[i]*vx_raised[:,i]))
print("Error in raised x-states being expected eigenvectors: ",raised_errs)
#repeat for y
for i in range(3):
    Jyvy=Jy@vy_raised[:,i]
    raised_errs[i]=np.sum(np.abs(Jyvy-raised_eigs[i]*vy_raised[:,i]))
print("Error in raised y-states being expected eigenvectors: ",raised_errs)

#check lowering.  now our eigenvalues should be (j-1,j-2,j-3...-j+1,0)
#for spin-1, m=1 goes to m=0, m=0 goes to m=-1, and m=-1 goes to zero
#which leaves with with (0,-1,0)
lowered_eigs=np.asarray([0,-1,0])
lowered_errs=np.zeros(3)
for i in range(3):
    Jxvx=Jx@vx_lowered[:,i]
    lowered_errs[i]=np.sum(np.abs(Jxvx-lowered_eigs[i]*vx_lowered[:,i]))
print("Error in lowered x-states being expected eigenvectors: ",lowered_errs)

for i in range(3):
    Jyvy=Jy@vy_lowered[:,i]
    lowered_errs[i]=np.sum(np.abs(Jyvy-lowered_eigs[i]*vy_lowered[:,i]))
print("Error in lowered y-states being expected eigenvectors: ",lowered_errs)

