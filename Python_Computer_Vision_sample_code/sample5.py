"""Problem Set 4: Geometry."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"

# Utility code
def read_points(filename):
    """Read point data from given file and return as NumPy array."""
    with open(filename) as f:
        return np.array([[float(pt)
                          for pt in line.split()]
                          for line in f.readlines()])


# Assignment code
def solve_least_squares(pts3d, pts2d):
    """Solve for transformation matrix M that maps each 3D point to corresponding 2D point using the least squares method.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        M: transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points
    """

    # TODO: Your code here
    r,c=np.shape(pts3d)[:2]
    A_matrix=np.zeros([2*r,11])
    b_matrix=np.zeros([2*r,1])
    m12=np.zeros([12,1])
  
    for i in range (0, r):
        A_matrix[2*i,:]=[pts3d[i][0],pts3d[i][1],pts3d[i][2],1,0,0,0,0, -pts2d[i][0]*pts3d[i][0],-pts2d[i][0]*pts3d[i][1],-pts2d[i][0]*pts3d[i][2]]
        A_matrix[2*i+1,:]=[0,0,0,0,pts3d[i][0],pts3d[i][1],pts3d[i][2],1,-pts2d[i][1]*pts3d[i][0],-pts2d[i][1]*pts3d[i][1],-pts2d[i][1]*pts3d[i][2]]
        b_matrix[2*i][0]=pts2d[i][0]
        b_matrix[2*i+1][0]=pts2d[i][1]
    m,residuals,rank,s = np.linalg.lstsq(A_matrix, b_matrix)
    for i in range(0,11):
        m12[i,0]=m[i,0]
    m12[11,0]=1
    M= np.reshape(m12, (-1, 4))
   
    error=residuals
 
    return M, error


def project_points(pts3d, M):
    """Project each 3D point to 2D using matrix M.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        M: projection matrix, NumPy array of shape (3, 4)

    Returns
    -------
        pts2d_projected: projected 2D points, NumPy array of shape (N, 2)
    """

    # TODO: Your code here
    r,c=np.shape(pts3d)[:2]
    pts3d_T=np.transpose(pts3d)
    oneArray=np.ones([1,r])
    pts3d_append=np.append(pts3d_T,oneArray,axis=0)
    temp=np.zeros([3,r])
    pts2d_projected=np.zeros([2,r])
    temp=np.dot(M,pts3d_append)
    for i in range(0,r):
        temp[0][i]=temp[0][i]/temp[2][i]
        temp[1][i]=temp[1][i]/temp[2][i]
    pts2d_projected=np.delete(temp, 2, 0)
    pts2d_projected=np.transpose(pts2d_projected)
    return pts2d_projected      


def get_residuals(pts2d, pts2d_projected):
    """Compute residual error for each point.

    Parameters
    ----------
        pts2d: observed 2D (image) points, NumPy array of shape (N, 2)
        pts2d_projected: 3D (object) points projected to 2D, NumPy array of shape (N, 2)

    Returns
    -------
        residuals: residual error for each point (L2 distance between observed and projected 2D points)
    """

    # TODO: Your code here
    r,c=np.shape(pts2d)[:2]
    residuals=np.zeros(r)
    for i in range (0,r):
        residuals[i]=np.sqrt((pts2d[i][0]-pts2d_projected[i][0])**2+(pts2d[i][1]-pts2d_projected[i][1])**2)
   
    return residuals


def calibrate_camera(pts3d, pts2d):
    """Find the best camera projection matrix given corresponding 3D and 2D points.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        bestM: best transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points for bestM
    """

    # TODO: Your code here
    # NOTE: Use the camera calibration procedure in the problem set
    random_index8=np.zeros(8)
    random_index12=np.zeros(12)
    random_index16=np.zeros(16)
    random4=np.zeros(4)#maybe 
    case4_3d=np.zeros([4,3])
    case4_2d=np.zeros([4,2])
    case8_3d=np.zeros([8,3])
    case8_2d=np.zeros([8,2])
    case12_3d=np.zeros([12,3])
    case12_2d=np.zeros([12,2])
    case16_3d=np.zeros([16,3])
    case16_2d=np.zeros([16,2])
    residuals=np.zeros([10,3])
    r4=np.zeros([10,3])
    M_matrix=np.zeros([30,3,4])
    bestM=np.zeros([3,4])
    
    for i in range (0,10):
        a=np.arange(20)
        np.random.shuffle(a)
        random_index8=a[:8]
        for j in range(0,8):
            case8_3d[j,:]=pts3d[random_index8[j],:]
            case8_2d[j,:]=pts2d[random_index8[j],:]
        M_matrix[i,:,:],residuals[i][0]=solve_least_squares(case8_3d,case8_2d)
        pts3dCopy=np.delete(pts3d,random_index8,0)
        pts2dCopy=np.delete(pts2d,random_index8,0)
        b=np.arange(12)
        np.random.shuffle(b)
        random4=b[:4]
        for j in range(0,4):
            case4_3d[j,:]=pts3dCopy[random4[j],:]
            case4_2d[j,:]=pts2dCopy[random4[j],:]
        project4=project_points(case4_3d, M_matrix[i,:,:])
        r4[i][0]=(np.sum(get_residuals(case4_2d, project4)))/4

        
        
        a=np.arange(20)
        np.random.shuffle(a)
        random_index12=a[:12]       
        for k in range(0,12):
            case12_3d[k,:]=pts3d[random_index12[k],:]
            case12_2d[k,:]=pts2d[random_index12[k],:]
        pts3dCopy=np.delete(pts3d,random_index12,0)
        pts2dCopy=np.delete(pts2d,random_index12,0)
        M_matrix[i+10,:,:],residuals[i][1]=solve_least_squares(case12_3d,case12_2d)
        b=np.arange(8)
        np.random.shuffle(b)
        random4=b[:4]
        for j in range(0,4):
            case4_3d[j,:]=pts3dCopy[random4[j],:]
            case4_2d[j,:]=pts2dCopy[random4[j],:]
        project4=project_points(case4_3d, M_matrix[i+10,:,:])
        r4[i][1]=(np.sum(get_residuals(case4_2d, project4)))/4

                     
        
        a=np.arange(20)
        np.random.shuffle(a)
        random_index16=a[:16]      
        for h in range(0,16):
            case16_3d[h,:]=pts3d[random_index16[h],:]
            case16_2d[h,:]=pts2d[random_index16[h],:]
        pts3dCopy=np.delete(pts3d,random_index16,0)
        pts2dCopy=np.delete(pts2d,random_index16,0)
        M_matrix[i+20,:,:],residuals[i][2]=solve_least_squares(case16_3d,case16_2d)
        project4=project_points(pts3dCopy, M_matrix[i+20,:,:])
        r4[i][2]=(np.sum(get_residuals(pts2dCopy, project4)))/4

                     
    min_index=np.where(r4==np.amin(r4))
    index=10*min_index[1]+min_index[0]
    bestM=M_matrix[index][:][:].reshape((3,4))
    print('residuals for the three cases of 8, 12,16 points:')
    print(r4)
    print('the best M is:')
    print(bestM)
  
    projectP2=project_points(pts3d, bestM)
    error=np.sum(get_residuals(pts2d, projectP2))
                             

      
    return bestM, error


def compute_fundamental_matrix(pts2d_a, pts2d_b):
    """Compute fundamental matrix given corresponding points from 2 images of a scene.

    Parameters
    ----------
        pts2d_a: 2D points from image A, NumPy array of shape (N, 2)
        pts2d_b: corresponding 2D points from image B, NumPy array of shape (N, 2)

    Returns
    -------
        F: the fundamental matrix
    """

    # TODO: Your code here

    r,c=np.shape(pts2d_a)[:2]
    A_matrix=np.zeros([r,8])
    b_matrix=np.zeros([r,1])
    f9=np.zeros([9,1])
    
    for i in range (0,r):
        A_matrix[i,:]=[pts2d_a[i,0]*pts2d_b[i,0],pts2d_a[i,0]*pts2d_b[i,1],pts2d_a[i,0],pts2d_a[i,1]*pts2d_b[i,0],pts2d_a[i,1]*pts2d_b[i,1],pts2d_a[i,1],pts2d_b[i,0],pts2d_b[i,1]]
        b_matrix[i][0]=-1
    f,residuals,rank,s = np.linalg.lstsq(A_matrix, b_matrix)
    for i in range(0,8):
        f9[i,0]=f[i,0]
    f9[8,0]=1
    F= np.reshape(f9, (-1, 3))
  
    return F


# Driver code
def main():
    """Driver code."""

    # 1a
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)  # TODO: implement this
    M_Norm=M*-0.5968
    print('M is:')
    print(M)
    
    


    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M_Norm)  # TODO: implement this
    
    #print(pts2d_projected)
    

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)  # TODO: implement this
    print('residuals are:')
    print(residuals)
    # TODO: Print the <u, v> projection of the last point, and the corresponding residual

    # 1b
    # Read points
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    # NOTE: These points are not normalized

    # TODO: Use the functions from 1a to implement calibrate_camera() and find the best transform (bestM)
    bestM,error=calibrate_camera(pts3d,pts2d_pic_b)
    

    # 1c
    # TODO: Compute the camera location using bestM
    Mt=np.transpose(bestM)
    Mr=np.delete(Mt,3,0)
    Q=np.transpose(Mr)
    m4=bestM[:,3]
    C=np.dot(-(np.linalg.inv(Q)),m4)
    print('C is:')
    print(C)

    # 2a
    # TODO: Implement compute_fundamental_matrix() to find the raw fundamental matrix
    pts2d_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_b = read_points(os.path.join(input_dir, PIC_B_2D))
    F=compute_fundamental_matrix(pts2d_a, pts2d_b)
    print('F is:')
    print(F)
    
 
    # 2b
    # TODO: Reduce the rank of the fundamental matrix
    U, s, V = np.linalg.svd(F)
    S=np.zeros(3)
    S[1]=s[1]
    S[0]=s[0]
    Sig=np.zeros((3, 3), dtype=complex)
    Sig[:3,:3]=np.diag(S)
    F2=np.dot(np.dot(U,Sig),V)
    print('F rank 2 is:')
    print(F2)
    

    # 2c
    # TODO: Draw epipolar lines
    pic_a= cv2.imread(os.path.join(input_dir, PIC_A))
    pic_b= cv2.imread(os.path.join(input_dir, PIC_B))
    Ha,Wa=pic_a.shape[:2]
    Hb,Wb=pic_b.shape[:2]
    
    Patl=[0,0,1]
    Pabl=[0,Ha-1,1]
    Patr=[Wa-1,0,1]
    Pabr=[Wa-1,Ha-1,1]
    lLa=np.cross(Patl,Pabl)
    lRa=np.cross(Patr,Pabr)
    Pbtl=[0,0,1]
    Pbbl=[0,Hb-1,1]
    Pbtr=[Wb-1,0,1]
    Pbbr=[Wb-1,Hb-1,1]
    lLb=np.cross(Pbtl,Pbbl)
    lRb=np.cross(Pbtr,Pbbr)
    r,c=np.shape(pts2d_a)[:2]
    la=np.zeros([r,3])
    lb=np.zeros([r,3])
    Plb_L=np.zeros([r,3])
    Plb_R=np.zeros([r,3])
    Pla_L=np.zeros([r,3])
    Pla_R=np.zeros([r,3])
    oneArr=np.ones([r,1])
    pts2d_A=np.append(pts2d_a, oneArr, axis=1)
    pts2d_B=np.append(pts2d_b, oneArr, axis=1)
    F2=np.real(F2)
    for i in range (0,r):
        lb[i,:]=np.dot(np.transpose(F2),pts2d_A[i,:])
        Plb_L[i,:]=np.cross(lb[i],lLb)
        Plb_R[i,:]=np.cross(lb[i],lRb)
        cv2.line(pic_b, (int(Plb_L[i,0]/Plb_L[i,2]),int(Plb_L[i,1]/Plb_L[i,2])), (int(Plb_R[i,0]/Plb_R[i,2]),int(Plb_R[i,1]/Plb_R[i,2])), (0,255,0),1)
        la[i,:]=np.dot(F2,pts2d_B[i,:])
        Pla_L[i,:]=np.cross(la[i],lLa)
        Pla_R[i,:]=np.cross(la[i],lRa)
        cv2.line(pic_a, (int(Pla_L[i,0]/Pla_L[i,2]),int(Pla_L[i,1]/Pla_L[i,2])), (int(Pla_R[i,0]/Pla_R[i,2]),int(Pla_R[i,1]/Pla_R[i,2])), (0,255,0),1)
    
    cv2.imwrite(os.path.join(output_dir, 'ps4_2_c_1.png'), pic_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4_2_c_2.png'), pic_b)

    #2d
    r,c=np.shape(pts2d_a)[:2]
    Ua_sum=0
    Va_sum=0
    Ub_sum=0
    Vb_sum=0
    for i in range (0,20):
        Ua_sum=Ua_sum+pts2d_a[i,0]
        Ub_sum=Ub_sum+pts2d_b[i,0]
        Va_sum=Va_sum+pts2d_a[i,1]
        Vb_sum=Vb_sum+pts2d_b[i,1]
    Cau=Ua_sum/r
    Cav=Va_sum/r
    Cbu=Ub_sum/r
    Cbv=Vb_sum/r
    for i in range (0,20):
        pts2d_a[i,0]=pts2d_a[i,0]-Cau
        pts2d_a[i,1]=pts2d_a[i,1]-Cav
        pts2d_b[i,0]= pts2d_b[i,0]-Cbu
        pts2d_b[i,1]= pts2d_b[i,1]-Cbv
    s= np.std(pts2d_a.ravel())
    sa=np.array([[1/s,0,0],[0,1/s,0],[0,0,1]])
    ma=np.array([[1,0,-Cau],[0,1,-Cav],[0,0,1]])
    Ta=np.dot(sa,ma)

    s= np.std(pts2d_b.ravel())
    sb=np.array([[1/s,0,0],[0,1/s,0],[0,0,1]])
    mb=np.array([[1,0,-Cbu],[0,1,-Cbv],[0,0,1]])
    Tb=np.dot(sb,mb)
    pts2d_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_b = read_points(os.path.join(input_dir, PIC_B_2D))
   
    pts2d_aNorm=np.dot(Ta,np.transpose(pts2d_A))
  
    pts2d_bNorm=np.dot(Tb,np.transpose(pts2d_B))
    pts2d_aNorm=np.transpose(np.delete(pts2d_aNorm,2,0))
    pts2d_bNorm=np.transpose(np.delete(pts2d_bNorm,2,0))
   
    F=compute_fundamental_matrix(pts2d_aNorm, pts2d_bNorm)
    U, s, V = np.linalg.svd(F)
    S=np.zeros(3)
    S[1]=s[1]
    S[0]=s[0]
    Sig=np.zeros((3, 3), dtype=complex)
    Sig[:3,:3]=np.diag(S)
    F2=np.dot(np.dot(U,Sig),V)
    print('F rank 2 based on normalized points')
    print(np.real(F2))

    #2e
    F_better=np.dot(np.dot(np.transpose(Tb),F2),Ta)
    print('F_better is:')
    print(np.real(F_better))

   
    
  

if __name__ == '__main__':
    main()
