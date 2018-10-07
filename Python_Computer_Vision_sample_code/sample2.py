"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
def optic_flow_LK(A, B, kernel):
    """Compute optic flow using the Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """

    # TODO: Your code here
    r,c=np.shape(B)
    A_arr=np.array(A)[0:r,0:c]
    B_arr=np.array(B)
    It_arr=B_arr-A_arr
    
    U=np.zeros([r,c])
    V=np.zeros([r,c])
    Ix = cv2.Sobel(A_arr,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(A_arr,cv2.CV_64F,0,1,ksize=5)
    Ix2=Ix*Ix
    Iy2=Iy*Iy
    IxIy=Ix*Iy
    IxIt=Ix*It_arr
    IyIt=Iy*It_arr
    filter=np.ones([kernel,kernel])
    
    Ix2_sum=cv2.filter2D(Ix2,-1,filter)
    Iy2_sum=cv2.filter2D(Iy2,-1,filter)
    IxIy_sum=cv2.filter2D(IxIy,-1,filter)
    IxIt_sum=cv2.filter2D(IxIt,-1,filter)
    IyIt_sum=cv2.filter2D(IyIt,-1,filter)
    for i in range(0,r):
        for j in range (0,c):
            temp=np.array([[Ix2_sum[i,j],IxIy_sum[i,j]],[IxIy_sum[i,j],Iy2_sum[i,j]]])
            tempT=np.transpose(temp)
            temp_inv=np.linalg.inv(np.dot(tempT,temp))
            b=np.array([[-IxIt_sum[i,j]],[-IyIt_sum[i,j]]])
            tempTb=np.dot(tempT,b)
            U[i,j],V[i,j]=np.dot(temp_inv,tempTb)
    return U, V


def reduce(image):
    """Reduce image to the next smaller level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, half size
    """

    # TODO: Your code here
    imgArr=np.array(image)
    image1=np.pad(imgArr, ((2,2),(2,2)), mode='constant', constant_values=0)
    r,c=np.shape(imgArr)
   
    reduced_image=np.zeros([r/2,c/2])
    hr,wr=np.shape(reduced_image)
    temp=np.zeros([5,5])
    #w=np.array([[1,1,1,1,1],[1,2,2,2,1],[1,2,4,2,1],[1,2,2,2,1],[1,1,1,1,1]])*(1/36)
    w=np.ones([5,5])*(1/25)
    for i in range(0,hr):
        for j in range(0,wr):
            for m in range(0,5):
                for n in range(0,5):
                    temp[m,n]=image1[2*i+m,2*j+n]
            TimesArr=w*temp
            reduced_image[i][j]=np.sum(TimesArr)
    return reduced_image


def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid of given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        levels: number of levels in the resulting pyramid

    Returns
    -------
        g_pyr: Gaussian pyramid, with g_pyr[0] = image
    """
    g_pyr=[]
    g_pyr.append(image)
    for i in range (1,levels):
        g_pyr.append(reduce(g_pyr[i-1]))
    # TODO: Your code here
    return g_pyr


def expand(image):
    """Expand image to the next larger level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, double size
    """

    # TODO: Your code here
    imgArr=np.array(image)
    r,c=np.shape(imgArr)
    image1=np.pad(imgArr, ((2,2),(2,2)), mode='constant', constant_values=0)
    
    expanded_image=np.zeros([r*2,c*2])   
    he,we=np.shape(expanded_image)
    temp=np.zeros([5,5])
    
    #w=np.array([[1,1,1,1,1],[1,2,2,2,1],[1,2,4,2,1],[1,2,2,2,1],[1,1,1,1,1]])*(1/36)
    w=np.ones([5,5])*(1/25)
    for i in range(0,he):
        for j in range(0,we):
            for m in range(-4,1):
                for n in range(-4,1):
                    
                    temp[m,n]=image1[(i-m)/2,(j-n)/2]
            TimesArr=w*temp
            expanded_image[i][j]=np.sum(TimesArr)
        
    return expanded_image


def laplacian_pyramid(g_pyr):
    """Create a Laplacian pyramid from a given Gaussian pyramid.

    Parameters
    ----------
        g_pyr: Gaussian pyramid, as returned by gaussian_pyramid()

    Returns
    -------
        l_pyr: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1]
    """

    # TODO: Your code here
    l_pyr=[]
    L=len(g_pyr)
    for i in range(0,L-1):
        r2,c2=np.shape(expand(g_pyr[i+1]))
        l_pyr.append((g_pyr[i])[0:r2,0:c2]-expand(g_pyr[i+1]))    
    l_pyr.append(g_pyr[L-1])
    return l_pyr


def warp(image, U, V):
    """Warp image using X and Y displacements (U and V).

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        warped: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]

    """
    r,c=np.shape(U)
    imgArr=np.array(image)[0:r][0:c]
    #r,c=np.shape(imgArr)
    warped=np.zeros([r,c])
    xrange=np.linspace(0, c-1, num=c, endpoint=True)
    yrange=np.linspace(0, r-1, num=r, endpoint=True)
    X,Y = np.meshgrid(xrange,yrange)
    XU=X+U
    YV=Y+V
    XU1=XU.astype(np.float32)
    YV1=YV.astype(np.float32)
    
    (map1,map2)=cv2.convertMaps(XU1, YV1, cv2.CV_16SC2,nninterpolation=True)
   
    
    warped=cv2.remap(imgArr,map1, map2,cv2.INTER_NEAREST)
    # TODO: Your code here
    return warped


def hierarchical_LK(A, B,n,kernel):
    """Compute optic flow using the Hierarchical Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """

    # TODO: Your code here
    A_g_pyr = gaussian_pyramid(A, n)
    B_g_pyr = gaussian_pyramid(B, n)
    blurA= (cv2.GaussianBlur(A_g_pyr[n-1],(3,3),5))
    blurB= (cv2.GaussianBlur(B_g_pyr[n-1],(3,3),5))
    U,V=optic_flow_LK(blurA, blurB,kernel)
    h,w=A_g_pyr[n-1].shape[:2]
    
    for k in range (n-2,-1,-1):
        U=2*expand(U)
        V=2*expand(V)
        C=warp(B_g_pyr[k],U,V)
        blurAk= (cv2.GaussianBlur(A_g_pyr[k],(3,3),5))
        blurC= (cv2.GaussianBlur(C,(3,3),5))
        Dx,Dy=optic_flow_LK(blurAk, blurC,kernel)
        U=U+Dx
        V=V+Dy
    
    return U, V

def UVimages(A,B,kernel):
    A_blur= (cv2.GaussianBlur(A,(3,3),3))
    B_blur= (cv2.GaussianBlur(B,(3,3),3))
    U, V = optic_flow_LK(A_blur, B_blur,kernel)
    
    h,w=A.shape[:2]
    dst1=np.zeros([h,w])
    dst2=np.zeros([h,w])
    UNorm=cv2.normalize(U,dst1,0,255,cv2.NORM_MINMAX)
    VNorm=cv2.normalize(V,dst2,0,255,cv2.NORM_MINMAX)
    
    cv2.imwrite(os.path.join(output_dir, "temp1.png"), UNorm)
    cv2.imwrite(os.path.join(output_dir, "temp2.png"), VNorm)
    U_grey=cv2.imread(os.path.join(output_dir, "temp1.png"), 0)
    V_grey=cv2.imread(os.path.join(output_dir, "temp2.png"), 0)
    U_color = cv2.applyColorMap(U_grey, cv2.COLORMAP_JET)
    V_color = cv2.applyColorMap(V_grey, cv2.COLORMAP_JET)
    image_pair = np.concatenate((U_color, V_color), axis=1)
    return image_pair

def ImageCombine(g_pyr):
    L=len(g_pyr)
    CombinedImg=g_pyr[0]
    r0,c0=np.shape(CombinedImg)
    dst=np.zeros([r0,c0])
    CombinedImg=cv2.normalize(CombinedImg,dst,0,255,cv2.NORM_MINMAX)
    
    for i in range (1,L):
        ri,ci=np.shape(g_pyr[i])
        dsti=np.zeros([r0,ci])
        temp=np.pad(g_pyr[i], ((0,r0-ri),(0,0)), mode='constant', constant_values=np.amin(g_pyr[i]))
        temp=cv2.normalize(temp,dsti,0,255,cv2.NORM_MINMAX)
        CombinedImg=np.concatenate((CombinedImg, temp), axis=1)
        
    # TODO: Your code here
    return CombinedImg

def UVImagePair(A,B,h,w,kernel):
    
    blurA= (cv2.GaussianBlur(A,(3,3),5))
    blurB= (cv2.GaussianBlur(B,(3,3),5))
    U, V = optic_flow_LK(blurA,blurB,kernel)
    resized_U = cv2.resize(U, (w, h))
    resized_V = cv2.resize(V, (w, h))
    resized_U=resized_U*(w*h/(A.shape[0]*A.shape[1]))
    resized_V=resized_V*(h*w/(A.shape[0]*A.shape[1]))
    d1=np.zeros([h,w])
    d2=np.zeros([h,w])
    UNorm=cv2.normalize(resized_U,d1,0,255,cv2.NORM_MINMAX)
    VNorm=cv2.normalize(resized_V,d2,0,255,cv2.NORM_MINMAX)
   
    cv2.imwrite(os.path.join(output_dir, "temp1.png"), UNorm)
    cv2.imwrite(os.path.join(output_dir, "temp2.png"), VNorm)
    U_grey=cv2.imread(os.path.join(output_dir, "temp1.png"), 0)
    V_grey=cv2.imread(os.path.join(output_dir, "temp2.png"), 0)
    image_pairG = np.concatenate((U_grey, V_grey), axis=1)
    image_pair=cv2.applyColorMap(image_pairG, cv2.COLORMAP_JET)
    return image_pair,resized_U,resized_V
def combineUV(U,V):
    h,w=np.shape(U)
    dst1=np.zeros([h,w])
    dst2=np.zeros([h,w])
    UNorm=cv2.normalize(U,dst1,0,255,cv2.NORM_MINMAX)
    VNorm=cv2.normalize(V,dst2,0,255,cv2.NORM_MINMAX)
    
    cv2.imwrite(os.path.join(output_dir, "temp1.png"), UNorm)
    cv2.imwrite(os.path.join(output_dir, "temp2.png"), VNorm)
    U_grey=cv2.imread(os.path.join(output_dir, "temp1.png"), 0)
    V_grey=cv2.imread(os.path.join(output_dir, "temp2.png"), 0)
    U_color = cv2.applyColorMap(U_grey, cv2.COLORMAP_JET)
    V_color = cv2.applyColorMap(V_grey, cv2.COLORMAP_JET)
    image_pair = np.concatenate((U_color, V_color), axis=1)
    return image_pair

# Driver code
def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    #An extra function UVimage is implemented to produce UV image based on optic_flow_LK method
    
    image_pair=UVimages(Shift0,ShiftR2,11)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-1.png"), image_pair) 
    # TODO: Similarly for Shift0 and ShiftR5U5
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0)/255.0 
    ShiftR5U5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0)/255.0
    image_pair=UVimages(Shift0,ShiftR5U5,11)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-2.png"), image_pair) 
    # 1b
    # TODO: Similarly for ShiftR10, ShiftR20 and ShiftR40
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    image_pair=UVimages(Shift0,ShiftR10,11)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-1.png"), image_pair)
    image_pair=UVimages(Shift0,ShiftR20,11)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-2.png"), image_pair)
    image_pair=UVimages(Shift0,ShiftR40,11)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-3.png"), image_pair)

    
    # 2a
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)  # TODO: implement this
    # TODO: Save pyramid images as a single side-by-side image (write a utility function?)
    combinedImg=ImageCombine(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps6-2-a-1.png"), combinedImg)
    # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)  # TODO: implement this
    # TODO: Save pyramid images as a single side-by-side image
    combinedImg=ImageCombine(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps6-2-b-1.png"), combinedImg)
   
  

    # 3a
    yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, 4)
    # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    h,w=yos_img_02.shape[:2]
    image_pair12,resized_U12, resized_V12 = UVImagePair(yos_img_02_g_pyr[2], yos_img_01_g_pyr[2],h,w,28)
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    yos_img_02_warped = warp(yos_img_02, resized_U12, resized_V12)  # TODO: implement this
    # TODO: Save difference image between yos_img_02_warped and original yos_img_01
    diffImg=yos_img_02_warped-yos_img_01
    h,w=yos_img_01.shape[:2]
    d=np.zeros([h,w])
    diff=cv2.normalize(diffImg,d,0,255,cv2.NORM_MINMAX)
    # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white
    yos_img_03 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.0
    yos_img_03_g_pyr = gaussian_pyramid(yos_img_03, 4)
    image_pair23,resized_U23, resized_V23 = UVImagePair(yos_img_03_g_pyr[2], yos_img_02_g_pyr[2],h,w,28)
    yos_img_03_warped = warp(yos_img_03, resized_U23, resized_V23)
    diffImg2=yos_img_03_warped-yos_img_02
    d=np.zeros([h,w])
    diff2=cv2.normalize(diffImg2,d,0,255,cv2.NORM_MINMAX)

    image_pairs = np.concatenate((image_pair12, image_pair23), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-1.png"), image_pairs)
    diffs=np.concatenate((diff, diff2), axis=1)
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-2.png"), diffs)
    # Similarly, you can compute displacements for yos_img_02 and yos_img_03 (but no need to save images)
    
    # TODO: Repeat for DataSeq2 (save images)
    seq2_0 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '0.png'), 0) / 255.0
    seq2_0_g_pyr = gaussian_pyramid(seq2_0, 4)
    seq2_1 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '1.png'), 0) / 255.0
    seq2_1_g_pyr = gaussian_pyramid(seq2_1, 4)
    h,w=seq2_0.shape[:2]
    image_pair01,resized_U01, resized_V01 = UVImagePair(seq2_1_g_pyr[2], seq2_0_g_pyr[2],h,w,20)
    seq01_warped = warp(seq2_1, resized_U01, resized_V01)
    diffImg01=seq01_warped-seq2_0
    d=np.zeros([h,w])
    diff01=cv2.normalize(diffImg01,d,0,255,cv2.NORM_MINMAX)
    seq2_2 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '2.png'), 0) / 255.0
    seq2_2_g_pyr = gaussian_pyramid(seq2_2, 4)
    
    image_pair12,resized_U12, resized_V12 = UVImagePair(seq2_2_g_pyr[2], seq2_1_g_pyr[2],h,w,20)
    seq12_warped = warp(seq2_2, resized_U12, resized_V12)
    diffImg12=seq12_warped-seq2_1
    d=np.zeros([h,w])
    diff12=cv2.normalize(diffImg12,d,0,255,cv2.NORM_MINMAX)
    image_pairs = np.concatenate((image_pair01, image_pair12), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-3.png"), image_pairs)
    diffs=np.concatenate((diff01, diff12), axis=1)
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-4.png"), diffs)

    

    # 4a
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    U10, V10 = hierarchical_LK(Shift0, ShiftR10,2,11)  # TODO: implement this
    U20, V20 = hierarchical_LK(Shift0, ShiftR20,3,11)
    U40, V40 = hierarchical_LK(Shift0, ShiftR40,4,11)
    image_pair10 = combineUV(U10,V10)
    image_pair20 = combineUV(U20,V20)
    image_pair40 = combineUV(U40,V40)
    # TODO: Save displacement image pairs (U, V), stacked
    stack1=np.concatenate((image_pair10, image_pair20), axis=0)
    stack2=np.concatenate((stack1, image_pair40), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-1.png"), stack2)
    # Hint: You can use np.concatenate()
    ShiftR10_warped = warp(ShiftR10, U10, V10)
    ShiftR20_warped = warp(ShiftR20, U20, V20)
    ShiftR40_warped = warp(ShiftR40, U40, V40)
    diffImg10=ShiftR10_warped-Shift0
    h,w=Shift0.shape[:2]
    d=np.zeros([h,w])
    diff10=cv2.normalize(diffImg10,d,0,255,cv2.NORM_MINMAX)
    diffImg20=ShiftR20_warped-Shift0
    d=np.zeros([h,w])
    diff20=cv2.normalize(diffImg20,d,0,255,cv2.NORM_MINMAX)
    diffImg40=ShiftR40_warped-Shift0
    d=np.zeros([h,w])
    diff40=cv2.normalize(diffImg40,d,0,255,cv2.NORM_MINMAX)
    stack1=np.concatenate((diff10, diff20), axis=0)
    stack2=np.concatenate((stack1, diff40), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-2.png"), stack2)
    # TODO: Save difference between each warped image and original image (Shift0), stacked
    
    # 4b
    # TODO: Repeat for DataSeq1 (use yos_img_01.png as the original)
    U12, V12 = hierarchical_LK(yos_img_01, yos_img_02,4,39)
    U13, V13 = hierarchical_LK(yos_img_01, yos_img_03,4,39)
    U23, V23 = hierarchical_LK(yos_img_02, yos_img_03,4,39)
    image_pair12 = combineUV(U12,V12)
    image_pair13 = combineUV(U13,V13)
    image_pair23 = combineUV(U23,V23)
    stack1=np.concatenate((image_pair12, image_pair13), axis=0)
    stack2=np.concatenate((stack1, image_pair23), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-b-1.png"), stack2)
    yos2_warped = warp(yos_img_02, U12, V12)
    h2,w2=np.shape(yos2_warped)
    diffImg12=yos2_warped-yos_img_01[0:h2,0:w2]
    yos3_warped = warp(yos_img_03, U13, V13)
    h3,w3=np.shape(yos3_warped)
    diffImg13=yos3_warped-yos_img_01[0:h3,0:w3]
    yos3_warped = warp(yos_img_03, U23, V23)
    h4,w4=np.shape(yos3_warped)
    diffImg23=yos3_warped-yos_img_02[0:h4,0:w4]
    h,w=yos_img_01.shape[:2]
    d=np.zeros([h,w])
    diff12=cv2.normalize(diffImg12,d,0,255,cv2.NORM_MINMAX)
    d=np.zeros([h,w])
    diff13=cv2.normalize(diffImg13,d,0,255,cv2.NORM_MINMAX)
    d=np.zeros([h,w])
    diff23=cv2.normalize(diffImg23,d,0,255,cv2.NORM_MINMAX)
    stack1=np.concatenate((diff12,diff13), axis=0)
    stack2=np.concatenate((stack1, diff23), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-b-2.png"), stack2)

    
    # 4c
    # TODO: Repeat for DataSeq1 (use 0.png as the original)    
    U01, V01 = hierarchical_LK(seq2_0, seq2_1,4,20)
    U02, V02 = hierarchical_LK(seq2_0, seq2_2,4,20)
    U12, V12 = hierarchical_LK(seq2_1, seq2_2,4,20)
    image_pair12 = combineUV(U12,V12)
    image_pair01 = combineUV(U01,V01)
    image_pair02 = combineUV(U02,V02)
    stack1=np.concatenate((image_pair01, image_pair02), axis=0)
    stack2=np.concatenate((stack1, image_pair12), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-c-1.png"), stack2)
    seq1_warped = warp(seq2_1, U01, V01)
    h2,w2=np.shape(seq1_warped)
    diffImg01=seq1_warped-seq2_0[0:h2,0:w2]
    seq2_warped = warp(seq2_2, U02, V02)
    h3,w3=np.shape(seq2_warped)
    diffImg02=seq2_warped-seq2_0[0:h3,0:w3]
    seq12_warped = warp(seq2_2, U12, V12)
    h4,w4=np.shape(seq12_warped)
    diffImg12=seq12_warped-seq2_1[0:h4,0:w4]
    h,w=seq2_0.shape[:2]
    d=np.zeros([h,w])
    diff01=cv2.normalize(diffImg01,d,0,255,cv2.NORM_MINMAX)
    d=np.zeros([h,w])
    diff12=cv2.normalize(diffImg12,d,0,255,cv2.NORM_MINMAX)
    d=np.zeros([h,w])
    diff02=cv2.normalize(diffImg02,d,0,255,cv2.NORM_MINMAX)
    stack1=np.concatenate((diff01,diff02), axis=0)
    stack2=np.concatenate((stack1, diff12), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-c-2.png"), stack2)

    #5
    jungle0 = cv2.imread(os.path.join(input_dir, 'Juggle', '0.png'), 0) / 255.0
    jungle1 = cv2.imread(os.path.join(input_dir, 'Juggle', '1.png'), 0) / 255.0
    jungle2 = cv2.imread(os.path.join(input_dir, 'Juggle', '2.png'), 0) / 255.0
    U01, V01 = hierarchical_LK(jungle0, jungle1,4,20)
    U02, V02 = hierarchical_LK(jungle0, jungle2,4,20)
    image_pair01 = combineUV(U01,V01)
    image_pair02 = combineUV(U02,V02)
    stack1=np.concatenate((image_pair01, image_pair02), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-5-a-1.png"), stack1)
    jungle1_warped = warp(jungle1, U01, V01)
    h2,w2=np.shape(jungle1_warped)
    diffImg01=jungle1_warped-jungle0[0:h2,0:w2]
    jungle2_warped = warp(jungle2, U02, V02)
    h3,w3=np.shape(jungle2_warped)
    diffImg02=jungle2_warped-jungle0[0:h3,0:w3]
    h,w=jungle0.shape[:2]
    d=np.zeros([h,w])
    diff01=cv2.normalize(diffImg01,d,0,255,cv2.NORM_MINMAX)
    d=np.zeros([h,w])
    diff02=cv2.normalize(diffImg02,d,0,255,cv2.NORM_MINMAX)
    stack1=np.concatenate((diff01,diff02), axis=1)
    cv2.imwrite(os.path.join(output_dir, "ps6-5-a-2.png"), stack1)

if __name__ == "__main__":
    main()
