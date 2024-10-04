import numpy as np

#generate the x kets.  colums should be (1,1) and (1,-1)/sqrt(2)
xket=np.asarray([[1,1],[1,-1]])/np.sqrt(2)
xbra=xket.conj().T

#generate the y kets the same way
yket=np.asarray([[1,1],[1J,-1J]])/np.sqrt(2)
ybra=yket.conj().T

#generic spin-1/2 angular momentum operator
#for the basis you're working in
J=np.diag([1,-1]) #we'll need to multiply by hbar/2
#generate Jx by rotating into Jx, applying
#the angular momentum operator, and rotate back
Jx=xket@J@xbra
print('Jx (plus hbar/2) is:')
print(Jx)
print('Jx hermitian check (should be zero): ',np.sum(np.abs(Jx-Jx.conj().T)))
#now repeat for Jy
Jy=yket@J@ybra
print('Jy (plus hbar/2) is: ')
print(Jy)
print('Jy hermitian check (should be zero): ',np.sum(np.abs(Jy-Jy.conj().T)))

zket=np.eye(2)

zket_y=ybra@zket
xket_y=ybra@xket
yket_y=ybra@yket
zbra_y=zket_y.conj().T
xbra_y=xket_y.conj().T
ybra_y=yket_y.conj().T

Jz_y=zket_y@J@zbra_y
Jx_y=xket_y@J@xbra_y
Jy_y=yket_y@J@ybra_y
print("Jz in y: ")
print(Jz_y)
print("Jx in y: ")
print(Jx_y)
print("Jy in y: ")
print(Jy_y)

print('Jz in y Hermitian check: ',np.sum(np.abs(Jz_y-Jz_y.conj().T)))
print('Jx in y Hermitian check: ',np.sum(np.abs(Jx_y-Jx_y.conj().T)))
print('Jy in y Hermitian check: ',np.sum(np.abs(Jy_y-Jy_y.conj().T)))

