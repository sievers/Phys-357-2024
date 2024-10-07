import numpy as np

#helper routine to convert (theta,phi) to (x,y,z) so we can
#use matrices
def theta_phi2xyz(theta,phi):
    return np.asarray([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])


def genbasis(xyz):
    #The answer to Q3 is in this routine, including comments
    out=np.zeros([3,3])
    #find a new vector perpendicular to our input vector
    #we can do that by taking the cross of the input against
    #a random vector
    ranvec=np.random.randn(3)
    tmp=np.cross(xyz,ranvec)
    #properly normalize do get a unit vector
    xhat=tmp/np.sqrt(tmp@tmp)
    #and we know that z cross x is y
    yhat=np.cross(xyz,xhat)
    #save the output.  This is effectively a matrix of kets.
    #we can transform from the new basis to the old basis with this matrix.
    #we can go from old to new with the bras, which is the transpose of
    #this matrix (everything is strictly real, so we can ignore conjugates)
    out[:,0]=xyz
    out[:,1]=xhat
    out[:,2]=yhat
    return out

def genrot(xyz,gamma):
    basis=genbasis(xyz)
    rmat=np.zeros([3,3])
    rmat[0,0]=1
    rmat[1,1]=rmat[2,2]=np.cos(gamma)
    rmat[1,2]=-np.sin(gamma)
    rmat[2,1]=np.sin(gamma)
    mat=basis@rmat@(basis.T)
    return mat

theta=np.pi/4
phi=np.pi/6
gamma=0.01

xyz=theta_phi2xyz(theta,phi)
rotmat=genrot(xyz,gamma)
print('Q4 answer: ')
print('rotation matrix of ',theta,phi,gamma,' is : ')
print(rotmat)
print('')
print('Q5: ')
lat_greenwich=51.476852
#lat_greenwich=89.476852
xyz_greenwich=theta_phi2xyz((90-lat_greenwich)*np.pi/180,0)
lat_montreal=45.50884
lon_montreal=-73.58781
xyz_montreal=theta_phi2xyz((90-lat_montreal)*np.pi/180,lon_montreal*np.pi/180)
#we need to find the direction perpendicular to the north pole and greenwich, so
#when we rotate about it, we can move Greenwich to the pole
xyz_north=np.asarray([0,0,1]) #the north pole
axis=np.cross(xyz_greenwich,xyz_north) 
axis=axis/np.sqrt(axis@axis) #we should normalize our rotation axis

gamma=(90-lat_greenwich)*np.pi/180 #this is the angle between greenwich and the north pole
rot_gm=genrot(axis,-gamma)
greenwich_new=rot_gm.T@xyz_greenwich
print("greenwich new position (should be 0,0,1): ",greenwich_new)

montreal_new=rot_gm.T@xyz_montreal
print("montreal's new position: ",montreal_new)
print("Expected z coordinate to be ",np.dot(xyz_greenwich,xyz_montreal))
montreal_lat=180/np.pi*(np.pi/2-np.arccos(montreal_new[2]))
#we can get the longitude by taking the arctan2.  Note that numpy expects
#the first argument of arctan2 to by y, not x.
montreal_lon=180/np.pi*(np.arctan2(montreal_new[1],montreal_new[0]))
print('montreal new lat/lon are ',montreal_lat,montreal_lon)

#answer to Q6:
#let's loop over several random angles, and show commutator is as expected
niter=10
gamma=1e-3 #pick a small gamma so we can tell which terms are large/small
print('')
print('Q6: error sum should be much smaller than commutator sum')
for i in range(niter):
    #create a normalized random vector
    vec=np.random.randn(3)
    vec=vec/np.sqrt(vec@vec)
    #generate a random basis that has that vector as a principle axis
    basis=genbasis(vec)
    #now generate the rotation matrices about those axex.
    #commutator of small rotations about two axes should be rotation about the
    #third axis by the angle squared, with an I removed
    rx=genrot(basis[:,0],gamma)
    ry=genrot(basis[:,1],gamma)
    rz=genrot(basis[:,2],gamma**2)
    rxry=rx@ry-ry@rx
    pred=rz-np.eye(3)
    comm_sum=np.sum(np.abs(rxry))
    err_sum=np.sum(np.abs(rxry-pred))
    print('commutator/error sums:',comm_sum,err_sum)
