"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2


import os
from math import pi

input_dir = "ps2\input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "ps2\output"  # write images to os.path.join(output_dir, <filename>)

def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """
    # TODO: Your code here
    arr=np.array(img_edges)
    r,c=np.shape(arr)
    row_col=np.transpose(np.nonzero(arr))
    theta_N=int((pi-0)/theta_res);
    theta=np.linspace(0,pi,theta_N,False)
    H_Col=theta_N
    H_Row=2*np.sqrt(r**2+c**2)
    H=np.zeros([H_Row, H_Col])
    rho=np.zeros(H_Row)

    for x in range (0, np.shape(row_col)[0]):
        for z in range (0, theta_N):
            rho_i=row_col[x][0]*np.cos(theta[z])+row_col[x][1]*np.sin(theta[z])
            H_r=np.sqrt(r**2+c**2)+round(rho_i/rho_res)*rho_res
            H_c=z
            H[H_r][H_c]+=1
            rho[H_r]=rho_i
   
  
    return H, rho, theta

    

def hough_peaks(H, Q):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    # TODO: Your code here
    H_arr=np.array(H)
    row,col=H_arr.shape
    peaks=[]
    for i in range(0,Q):
        max=np.amax(H_arr)
        if max<=80:
            break
        else:
            I1,I2=np.where(H_arr==max)
            peaks.append([I1[0],I2[0]])
            H_arr[I1[0]][I2[0]]=0
            
    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value

    pass"""  # TODO: Your code here (nothing to return, just draw on img_out directly)
    h_out,w_out=img_out.shape[:2]
    y_ori=0
    x_last=w_out-1
    x_ori=0
    y_last=h_out-1
    p_row=len(peaks)
    for j in range (0, p_row):
        theta_j=theta[peaks[j][1]]
        rho_j=rho[peaks[j][0]]
        if theta_j!=0:
            x0=int(y_ori*(-np.cos(theta_j)/np.sin(theta_j))+rho_j/(np.sin(theta_j)))
            x=int(y_last*(-np.cos(theta_j)/np.sin(theta_j))+rho_j/(np.sin(theta_j)))
            cv2.line(img_out, (x0, y_ori), (x, y_last), (0,255,0),2)
        else:
            y0=int(y_ori*(-np.sin(theta_j)/np.cos(theta_j))+rho_j/(np.cos(theta_j)))
            y=int(y_last*(-np.sin(theta_j)/np.cos(theta_j))+rho_j/(np.cos(theta_j)))
            cv2.line(img_out, (x_ori,y0), (x_last,y), (0,255,0),2)


def hough_circles_acc(img_edges, r):
    """Compute Hough Transform for circles on edge image."""
    arr=np.array(img_edges)
    rr,cc=np.shape(arr)
    H=np.zeros([rr,cc],dtype=int)
    row_col=np.transpose(np.nonzero(arr))
    row_arr=row_col[:,0]
    col_arr=row_col[:,1]
    theta_res=pi/180
    theta_N=int((2*pi-0)/theta_res)+1;
    theta=np.linspace(0,2*pi,theta_N,True)
    xgrid=np.meshgrid(row_arr,theta)
    ygrid=np.meshgrid(col_arr,theta)
    a_preMask=xgrid[0]-r*np.cos(xgrid[1])
    b_preMask=ygrid[0]-r*np.sin(ygrid[1])
    a_preMask=np.ravel(a_preMask)
    b_preMask=np.ravel(b_preMask)
    #filter out points that are not in bounds
    m_a=np.ma.masked_where(a_preMask>=rr, a_preMask)
    m_b=np.ma.masked_where(np.ma.getmask(m_a), b_preMask)
    a_temp=np.ma.compressed(m_a)
    b_temp=np.ma.compressed(m_b)
    m_b2=np.ma.masked_where(b_temp>=cc, b_temp)
    m_a2=np.ma.masked_where(np.ma.getmask(m_b2), a_temp)
    a_axis=np.ma.compressed(m_a2)
    b_axis=np.ma.compressed(m_b2)

    m_a=np.ma.masked_where(a_axis<0, a_axis)
    m_b=np.ma.masked_where(np.ma.getmask(m_a), b_axis)
    a_temp=np.ma.compressed(m_a)
    b_temp=np.ma.compressed(m_b)
    m_b2=np.ma.masked_where(b_temp<0, b_temp)
    m_a2=np.ma.masked_where(np.ma.getmask(m_b2), a_temp)
    a_axis=np.ma.compressed(m_a2)
    b_axis=np.ma.compressed(m_b2)
    for i in range (0,np.size(a_axis)):
        H_r=a_axis[i]
        H_c=b_axis[i]
        H[H_r][H_c]+=1
   
    return H

def find_circles(img_edges, r1, r2):
    centers=[]
    radii=[]
    temp1=[]
    temp2=[]
   
    for r in range (r1,r2+1,2):
        H=hough_circles_acc(img_edges, r)
        peaks=hough_peaks(H, 10)
        p=len(peaks)
        #remove the overlapping circles, and draw the left circles 
        for i in range (0,p-1):
            ai=peaks[i][0]
            bi=peaks[i][1]
            counter=0
            for j in range (i+1,p):
                if np.sqrt((ai-peaks[j][0])**2+(bi-peaks[j][1])**2)<1.5*r:
                    break
                else:
                    counter+=1
                    continue
            if counter==p-i-1:
                centers.append((ai,bi))
                radii.append(r)
        centers.append((peaks[p-1][0],peaks[p-1][1]))
        radii.append(r)
    
    
    return centers, radii
                
            

def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    # TODO: Compute edge image (img_edges)
    img_edges=cv2.Canny(img, 100, 200)
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges)  # TODO: implement this, try calling with different parameters

    # TODO: Store accumulator array (H) as ps2-2-a-1.png
    h1,w1=H.shape[:2]
    dst=np.zeros([h1,w1])
    cv2.normalize(H,dst,0,255,cv2.NORM_MINMAX) 
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), dst)


    # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 10)  # TODO: implement this, try different parameters
    img_copy=cv2.imread(os.path.join(output_dir, 'ps2-2-a-1.png'))
    p=len(peaks)
    for i in range (0,p):
        a=peaks[i][0]
        b=peaks[i][1]
        cv2.circle(img_copy, (b,a), 3, (0,255,0), -1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), img_copy)
    
    # TODO: Store a copy of accumulator array image (from 2-a), with peaks highlighted, as ps2-2-b-1.png

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png
    
    # 3-a
    # TODO: Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter
    img_noise = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'))
    blur = cv2.GaussianBlur(img_noise,(5,5),4)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), blur) 
    # 3-b
    # TODO: Compute binary edge images for both original image and smoothed version
    img_noisy = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'))
    img_smooth = cv2.imread(os.path.join(output_dir, 'ps2-3-a-1.png'))
    img_noisy_edge=cv2.Canny(img_noisy,100,200)
    img_smooth_edge=cv2.Canny(img_smooth, 100, 200)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'),img_noisy_edge)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img_smooth_edge) 
    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    img_smooth_edges = cv2.imread(os.path.join(output_dir, 'ps2-3-b-2.png'),0)
    H1, rho1, theta1 = hough_lines_acc(img_smooth_edges)
    h2,w2=H1.shape[:2]
    dst1=np.zeros([h2,w2])
    cv2.normalize(H1,dst1,0,255,cv2.NORM_MINMAX)
    peaks1 = hough_peaks(H1, 10)
    p=len(peaks1)
    cv2.imwrite(os.path.join(output_dir, 'temp.png'), H1)
    img_temp=cv2.imread(os.path.join(output_dir, 'temp.png'))
    
    for i in range (0,p):
        a=peaks1[i][0]
        b=peaks1[i][1]
        cv2.circle(img_temp, (b,a), 5, (0,255,0), -1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), img_temp)
    
    img_noisy_edges = cv2.imread(os.path.join(output_dir, 'ps2-3-b-1.png'),0)
    H2, rho2, theta2 = hough_lines_acc(img_smooth_edges)
    h3,w3=H2.shape[:2]
    dst2=np.zeros([h3,w3])
    cv2.normalize(H2,dst2,0,255,cv2.NORM_MINMAX)
    peaks2 = hough_peaks(H2, 10)
    hough_lines_draw(img_noisy, peaks2, rho2, theta2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_noisy)
    
    # 4
    # TODO: Like problem 3 above, but using ps2-input1.png
    img2 = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'))
    grey=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),4)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), blur)

    #create edge image
    img2_copy = cv2.imread(os.path.join(output_dir, 'ps2-4-a-1.png'))
    img2_edge=cv2.Canny(img2_copy,100,200)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), img2_edge)

    #find lines
    img2_edges = cv2.imread(os.path.join(output_dir, 'ps2-4-b-1.png'),0)
    H3, rho3, theta3 = hough_lines_acc(img2_edges)
    h4,w4=H3.shape[:2]
    dst3=np.zeros([h4,w4])
    cv2.normalize(H3,dst3,0,255,cv2.NORM_MINMAX)
    temp2=cv2.imwrite(os.path.join(output_dir, 'temp2.png'), dst3)
    peaks3 = hough_peaks(H3, 20)

    img2_edges_copy=cv2.imread(os.path.join(output_dir, 'temp2.png'))
    p=len(peaks3)
    for i in range (0,p):
        a=peaks3[i][0]
        b=peaks3[i][1]
        cv2.circle(img2_edges_copy, (b,a), 3, (0,255,0), -1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-1.png'), img2_edges_copy) 

    hough_lines_draw(img2, peaks3, rho3, theta3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-2.png'), img2)

    
    # 5
    # TODO: Implement Hough Transform for circles
    img2 = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'))
    grey=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),4)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), blur)
    img2_copy = cv2.imread(os.path.join(output_dir, 'ps2-5-a-1.png'))
    img_edges=cv2.Canny(img2_copy,100,200)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img_edges)
     
    temp3 = cv2.imread(os.path.join(output_dir, 'ps2-5-a-1.png'))
    img2_edges = cv2.imread(os.path.join(output_dir, 'ps2-5-a-2.png'),0)
    H=hough_circles_acc(img2_edges, 20)
    peaks = hough_peaks(H, 10)
    p=len(peaks)
    #remove the overlapping circles, and draw the left circles 
    for i in range (0,p-1):
        ai=peaks[i][0]
        bi=peaks[i][1]
        counter=0
        for j in range (i+1,p):
            if np.sqrt((ai-peaks[j][0])**2+(bi-peaks[j][1])**2)<10:
                break
            else:
                counter+=1
                continue
        if counter==p-i-1:
            cv2.circle(temp3, (bi,ai), 20, (0,255,0), 1)
    cv2.circle(temp3, (peaks[p-1][1],peaks[p-1][0]), 20, (0,255,0), 1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-3.png'), temp3)

    #5b
    img_edges = cv2.imread(os.path.join(output_dir, 'ps2-5-a-2.png'),0)
    centers, radii=find_circles(img_edges, 20, 28)
    temp4 = cv2.imread(os.path.join(output_dir, 'ps2-5-a-1.png'))
    for i in range (0, len(centers)):
        cv2.circle(temp4, (centers[i][1],centers[i][0]), radii[i], (0,255,0), 1)
        #print(centers[i][1],centers[i][0],radii[i])
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), temp4)
    
    
    # 6
    # TODO: Find lines a more realtistic image, ps2-input2.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'))
    blur = cv2.GaussianBlur(img,(5,5),4)
    cv2.imwrite(os.path.join(output_dir, 'temp5.png'), blur) 
    temp=cv2.imread(os.path.join(output_dir, 'temp5.png'))
    img_edge=cv2.Canny(temp,100,200)
    cv2.imwrite(os.path.join(output_dir, 'temp6.png'), img_edge) 
    img_edges=cv2.imread(os.path.join(output_dir, 'temp6.png'),0)
    H, rho, theta = hough_lines_acc(img_edges)
    h,w=H.shape[:2]
    dst=np.zeros([h,w])
    cv2.normalize(H,dst,0,255,cv2.NORM_MINMAX)
    peaks = hough_peaks(H, 10)
    smooth1=cv2.imread(os.path.join(output_dir, 'temp5.png'))
    hough_lines_draw(smooth1, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-a-1.png'), smooth1)
    

    # 7
    # TODO: Find circles in the same realtistic image, ps2-input2.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'))
    blur = cv2.GaussianBlur(img,(5,5),4)
    cv2.imwrite(os.path.join(output_dir, 'temp6.png'), blur) 
    tempim=cv2.imread(os.path.join(output_dir, 'temp6.png'),0)
    img_edges=cv2.Canny(tempim,100,200)
    cv2.imwrite(os.path.join(output_dir, 'temp7.png'), img_edges)
    tempimg=cv2.imread(os.path.join(output_dir, 'temp7.png'),0)
    centers, radii=find_circles(tempimg, 24, 36)
    tempim=cv2.imread(os.path.join(output_dir, 'temp6.png'))
    for i in range (0, len(centers)):
       cv2.circle(tempim, (centers[i][1],centers[i][0]), radii[i], (0,255,0), 1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-7-a-1.png'), tempim)

    # 8
    # TODO: Find lines and circles in distorted image, ps2-input3.png
    


if __name__ == "__main__":
    main()
