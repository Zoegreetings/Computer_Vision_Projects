"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
import cv2

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def disparity_ssd(L, R, maxd, windowsize):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    pass  # TODO: Your code here
    h,w=L.shape[:2]
    d_value=np.zeros([h,w])
    ssd=np.zeros([h,w])+255**3
    filter=np.ones([windowsize,windowsize])/windowsize*windowsize
    for d in range (-maxd,maxd):
        dtemp=np.zeros([h,w])
        M=np.float32([[1,0,d],[0,1,0]])
        rshift=cv2.warpAffine(R, M, (w,h))
        diff=L-rshift
        diff=diff**2
        ssd_mat=cv2.filter2D(diff,-1,filter)
        ssd_arr=np.array(ssd_mat)
        mask=np.greater(ssd,ssd_arr)
        ssd[mask]=ssd_arr[mask]
        dtemp=dtemp+np.abs(d)
        d_value[mask]=dtemp[mask]
    dArray=d_value*255
    dst=np.zeros([h,w])
    cv2.normalize(d_value,dst,0,255,cv2.NORM_MINMAX)  
         
    return dst


def disparity_ncorr(L, R, maxd, windowsize):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    pass  # TODO: Your code here
    h,w=L.shape[:2]
    d_value=np.zeros([h,w])
    ncorr=np.zeros([h,w])
    filter=np.ones([windowsize,windowsize])/windowsize*windowsize
    for d in range (-maxd,maxd):
        dtemp=np.zeros([h,w])
        M=np.float32([[1,0,d],[0,1,0]])
        rshift=cv2.warpAffine(R, M, (w,h))
        L_arr=np.array(L)
        rshift_arr=np.array(rshift)
        if d>=0:
            for i in range (0,d):
                rshift_arr[:,i]=R[:,0]  
        if d<0:
            for i in range (0,np.abs(d)):
                rshift_arr[:,w-i-1]=R[:,w-1]
        LR_dot=L_arr*rshift_arr
        LR_mat=cv2.filter2D(LR_dot,-1,filter)
        LL=L_arr*L_arr
        LL_mat=cv2.filter2D(LL,-1,filter)
        rr=rshift_arr*rshift_arr
        rr_mat=cv2.filter2D(rr,-1,filter)
        sqr=np.sqrt(LL_mat*rr_mat)
        ncorr_arr=LR_mat/sqr
    
        mask=np.greater(ncorr_arr,ncorr)
        ncorr[mask]=ncorr_arr[mask]
        dtemp=dtemp+np.abs(d)
        d_value[mask]=dtemp[mask]
    dArray=d_value*255
    dst=np.zeros([h,w])
    cv2.normalize(d_value,dst,0,255,cv2.NORM_MINMAX)
    return dst
        
        


def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    maxd=7
    windowsize=5
    D_L = disparity_ssd(L, R, maxd, windowsize)  # TODO: implemenet disparity_ssd()
    D_R = disparity_ssd(R, L, maxd,windowsize)
    
    cv2.imwrite(os.path.join('output', 'ps3-1-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-1-a-2.png'), D_R)
    # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1 / 255.0)  
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1 / 255.0)
    maxd=140
    windowsize=5
    D_L = disparity_ssd(L, R, maxd,windowsize) 
    D_R = disparity_ssd(R, L, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-2-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-2-a-2.png'), D_R)

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    # TODO: Boost contrast in one image and apply again
    h,w=L.shape[:2]
    data=(np.array(L)).astype(float)*255
    sigma=20
    noise=np.random.randn(h,w)*sigma
    L_noisy=data+noise
    cv2.imwrite(os.path.join('output', 'temp.png'), L_noisy)
    L_noise = cv2.imread(os.path.join('output', 'temp.png'), 0) * (1 / 255.0)
    D_L = disparity_ssd(L_noise, R, maxd,windowsize) 
    D_R = disparity_ssd(R, L_noise, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-3-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-3-a-2.png'), D_R)

    L_increase=L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1 / 255.0)*1.1
    D_L = disparity_ssd(L_increase, R, maxd,windowsize) 
    D_R = disparity_ssd(R, L_increase, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-3-b-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-3-b-2.png'), D_R)

    
    
    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1 / 255.0)  
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1 / 255.0)
    maxd=140
    windowsize=5
    D_L = disparity_ncorr(L, R, maxd,windowsize) 
    D_R = disparity_ncorr(R, L, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-4-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-4-a-2.png'), D_R)

    h,w=L.shape[:2]
    data=(np.array(L)).astype(float)*255
    sigma=20
    noise=np.random.randn(h,w)*sigma
    L_noisy=data+noise
    cv2.imwrite(os.path.join('output', 'temp1.png'), L_noisy)
    L_noise = cv2.imread(os.path.join('output', 'temp1.png'), 0) * (1 / 255.0)
    D_L = disparity_ncorr(L_noise, R, maxd,windowsize) 
    D_R = disparity_ncorr(R, L_noise, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-4-b-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-4-b-2.png'), D_R)

    L_increase=L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1 / 255.0)*1.1
    D_L = disparity_ncorr(L_increase, R, maxd,windowsize) 
    D_R = disparity_ncorr(R, L_increase, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-4-b-3.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-4-b-4.png'), D_R)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results
    L = cv2.imread(os.path.join('input', 'pair2-L.png'), 0) * (1 / 255.0)  
    R = cv2.imread(os.path.join('input', 'pair2-R.png'), 0) * (1 / 255.0)
    maxd=130
    windowsize=9
    D_L = disparity_ncorr(L, R, maxd,windowsize) 
    D_R = disparity_ncorr(R, L, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'ps3-5-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps3-5-a-2.png'), D_R)
    D_L2 = disparity_ssd(L, R, maxd,windowsize) 
    D_R2 = disparity_ssd(R, L, maxd,windowsize)
    cv2.imwrite(os.path.join('output', 'temp2.png'), D_L2)
    cv2.imwrite(os.path.join('output', 'temp3.png'), D_R2)

    

if __name__ == "__main__":
    main()
