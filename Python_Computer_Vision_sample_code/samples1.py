"""Problem Set 5: Harris, SIFT, RANSAC."""

import numpy as np
import math
import cv2
import random

import os
# it takes about 18 seconds to run this program
# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
def gradientX(image):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    sobel_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)  
    h1,w1=image.shape[:2]
    dst=np.zeros([h1,w1])
    Ix=cv2.normalize(sobel_x,dst,-1.0,1.0,cv2.NORM_MINMAX)
    return Ix


def gradientY(image):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    sobel_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    h1,w1=image.shape[:2]
    dst=np.zeros([h1,w1])
    Iy=cv2.normalize(sobel_y,dst,-1.0,1.0,cv2.NORM_MINMAX) 
    return Iy


def make_image_pair(image1, image2):
    """Adjoin two images side-by-side to make a single new image.

    Parameters
    ----------
        image1: first image, could be grayscale or color (BGR)
        image2: second image, same type as first

    Returns
    -------
        image_pair: combination of both images, side-by-side, same type
    """

    # TODO: Your code here
    image_pair = np.concatenate((image1, image2), axis=1)
    return image_pair


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris reponse map using given image gradients.

    Parameters
    ----------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
        Iy: image gradient in Y direction, same size and type as Ix
        kernel: 2D windowing kernel with weights, typically square
        alpha: Harris detector parameter multiplied with square of trace

    Returns
    -------
        R: Harris response map, same size as inputs, floating-point
    """

    # TODO: Your code here
    # Note: Define any other parameters you need locally or as keyword arguments
    Ix1 = cv2.GaussianBlur(Ix,(3,3),4)
    Iy1 = cv2.GaussianBlur(Iy,(3,3),4)
    
    Ix2_ave=cv2.filter2D(Ix1**2,-1,kernel)
    Iy2_ave=cv2.filter2D(Iy1**2,-1,kernel)
    IxIy_ave=cv2.filter2D(Ix1*Iy1,-1,kernel)
    h1,w1=Ix.shape[:2]
    R=np.zeros([h1,w1])
    for i in range (0,h1):
        for j in range (0, w1):
            M=np.array([[Ix2_ave[i,j], IxIy_ave[i,j]],[IxIy_ave[i,j],Iy2_ave[i,j]]])
            detM=Ix2_ave[i,j]*Iy2_ave[i,j]-IxIy_ave[i,j]*IxIy_ave[i,j]
            traceM=Ix2_ave[i,j]+Iy2_ave[i,j]
            R[i,j]=detM-alpha*traceM**2
    
    a=np.array(R)
    
    return R


def find_corners(R, threshold, radius):
    """Find corners in given response map.

    Parameters
    ----------
        R: floating-point response map, e.g. output from the Harris detector
        threshold: response values less than this should not be considered plausible corners
        radius: radius of circular region for non-maximal suppression (could be half the side of square instead)

    Returns
    -------
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates
    """
    r,c=np.shape(R)
    corners=np.zeros([r,c])
    R_extend=np.zeros([r+2*radius, c+2*radius])
    R_extend[radius:r+radius,radius:c+radius]=R
    
    # TODO: Your code here
    for i in range (radius,r+radius):
        for j in range (radius,c+radius):
            if R_extend[i,j]==np.amax(R_extend[i-radius:i+radius+1, j-radius:j+radius+1])and R_extend[i,j]>threshold:
                corners[i-radius,j-radius]=R[i-radius,j-radius]
   
    return corners


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates

    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """

    # TODO: Your code here
    r,c=np.shape(corners)
    h,w=image.shape[:2]
    n=0
    image_out=image
    for i in range (0, r):
        for j in range (0, c):
            if corners[i,j]!=0: 
                n=n+1
                cv2.circle(image_out, (j,i), 3, (0,255,0), -1)
    
    return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix: image gradient in X direction
        Iy: image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle: gradient angle image, each value in degrees [0, 359)
    """

    # TODO: Your code here
    # Note: +ve X axis points to the right (0 degrees), +ve Y axis points down (90 degrees)
    Ix_arr=np.array(Ix)
    Iy_arr=np.array(Iy)
    r,c=np.shape(Ix_arr)
    angle=np.zeros([r,c])
    angle[:,:]=np.arctan2(Iy_arr[:,:],Ix_arr[:,:])
    for i in range(0,r):
        for j in range (0,c):
            if angle[i,j]>=0:
                angle[i,j]=(angle[i,j]/np.pi)*180
            else:
                angle[i,j]=((angle[i,j]+2*np.pi)/np.pi)*180
    
    return angle


def get_keypoints(points, R, angle, _size, _octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.

    Parameters
    ----------
        points: interest points (e.g. corners), as a sequence (list) of (x, y) coordinates
        R: floating-point response map, e.g. output from the Harris detector
        angle: gradient angle (orientation) image, each value in degrees [0, 359)
        _size: fixed _size parameter to pass to cv2.KeyPoint() for all points
        _octave: fixed _octave parameter to pass to cv2.KeyPoint() for all points

    Returns
    -------
        keypoints: a sequence (list) of cv2.KeyPoint objects
    """

    # TODO: Your code here
    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
  
    r,c=np.shape(points)
    keypoints = []
    for i in range(0,r):
        for j in range (0,c):
            if points[i,j]!=0:
                keypoints.append(cv2.KeyPoint(j,i, _size, angle[i,j],R[i,j], _octave))
    return keypoints


def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.

    Parameters
    ----------
        keypoints: a sequence (list) of cv2.KeyPoint objects

    Returns
    -------
        descriptors: 2D NumPy array of shape (len(keypoints), 128)
    """
    
    # TODO: Your code here
    # Note: You can use OpenCV's SIFT.compute() method to extract descriptors, or write your own!
    sift=cv2.ORB_create()
    keys, descriptors=sift.compute(image,keypoints)
    
    return keys, descriptors


def match_descriptors(desc1, desc2):
    """Match feature descriptors obtained from two images.

    Parameters
    ----------
        desc1: descriptors from image 1, as returned by SIFT.compute()
        desc2: descriptors from image 2, same format as desc1

    Returns
    -------
        matches: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """

    # TODO: Your code here
    bfm=cv2.BFMatcher()
    match=bfm.match(desc1,desc2)
    # Note: You can use OpenCV's descriptor matchers, or roll your own!
   
    return match


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.

    Parameters
    ----------
        image1: first image
        image2: second image, same type as first
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out: image1 and image2 joined side-by-side with matching lines; color image (BGR), uint8, values in [0, 255]
    """

    # TODO: Your code here
    temp=make_image_pair(image1,image2)
    h1,w1=image1.shape[:2]
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        colors = ((0x00, 0xFF, 0xFF),
		(0x00, 0x00, 0xFF),
		(0x00, 0xFF, 0x00),
		(0xFF, 0x00, 0x00), 
		(0xFF, 0x00, 0xFF), 
		(0x2D, 0x52, 0xA0),
		(0x80, 0x00, 0x80),
		(0x00, 0x8C, 0xFF),
		(0x00, 0x00, 0x80), 
		(0xFF, 0xFF, 0x00))
        cv2.circle(temp, (int(x1),int(y1)), 3, (255, 0, 0), 1)   
        cv2.circle(temp, (int(x2)+w1,int(y2)), 3, (255, 0, 0), 1)
        cv2.line(temp, (int(x1), int(y1)),(int(x2)+w1, int(y2)),random.choice(colors),1)
    image_out=temp        
    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own :)
    return image_out


def compute_translation_RANSAC(kp1, kp2, matches, tolerance):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        translation: translation/offset vector <x, y>, NumPy array of shape (2, 1)
        good_matches: consensus set of matches that agree with this translation
    """

    # TODO: Your code here
    
    translation= []
    t=[]
    
    good_matches=[]
    good={}
    L=len(matches)
    max=0
    n=np.zeros(L)
   
    for i in range (0,L):
        
        
        img1_idx = matches[i].queryIdx
        img2_idx = matches[i].trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        Dx=x2-x1
        Dy=y2-y1
        t.append([Dx,Dy])
       
        good[i]=[matches[i]]
        for j in range (0,L):
            
            img1_idxj = matches[j].queryIdx
            img2_idxj = matches[j].trainIdx
            (x1j,y1j) = kp1[img1_idxj].pt
            (x2j,y2j) = kp2[img2_idxj].pt
            x2_com=x1j+Dx
            y2_com=y1j+Dy
            d=np.sqrt((x2j-x2_com)**2+(y2j-y2_com)**2)
            
            if d<=tolerance:
                good[i].append(matches[j])
                
                n[i]=n[i]+1
           
        if n[i]>max:
            max=n[i]
            good_matches=good[i]
            translation[:]=[]
            translation.append(t[i][0])
            translation.append(t[i][1])
    
    print('length of good match for transA and transB is ', len(good_matches)-1)
   
    return translation, good_matches      


def compute_similarity_RANSAC(kp1, kp2, matches, tolerance):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    # TODO: Your code here
    transform= np.zeros([2,3])
    t=[]
    
    good_matches=[]
    good={}
    L=len(matches)
    max=0
    n=np.zeros(L)
   
    for i in range (0,L-1):
   
        img1_idx = matches[i].queryIdx
        img2_idx = matches[i].trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        img1_idx = matches[i+1].queryIdx
        img2_idx = matches[i+1].trainIdx
        (x3,y3) = kp1[img1_idx].pt
        (x4,y4) = kp2[img2_idx].pt
        A=np.array([[x1, -y1, 1, 0],[y1, x1, 0, 1], [x3, -y3, 1, 0], [y3, x3, 0, 1]])
        b=([[x2],[y2],[x4],[y4]])
        vector=np.dot(np.linalg.inv(A),b)
        transV=np.array([[vector[0][0], -vector[1][0], vector[2][0]], [vector[1][0], vector[0][0], vector[3][0]]])
        t.append(transV)
       
        good[i]=[matches[i]]
        for j in range (0,L):
            
            img1_idxj = matches[j].queryIdx
            img2_idxj = matches[j].trainIdx
            (x1j,y1j) = kp1[img1_idxj].pt
            (x2j,y2j) = kp2[img2_idxj].pt
            simAxis=np.array([[x1j],[y1j],[1]])
            axis_com=np.dot(transV, simAxis)
            
            d=np.sqrt((x2j-axis_com[0][0])**2+(y2j-axis_com[1][0])**2)
            
            if d<=tolerance:
                good[i].append(matches[j])
                
                n[i]=n[i]+1
           
        if n[i]>max:
            max=n[i]
            good_matches=good[i]
            transform=t[i]
            
   
    print('length of good match for simA and simB is ', len(good_matches)-1)
   
    return transform, good_matches
def compute_affine_RANSAC(kp1, kp2, matches, tolerance):
    

    transform= np.zeros([2,3])
    t=[]    
    good_matches=[]
    good={}
    L=len(matches)
    max=0
    n=np.zeros(L)
   
    for i in range (0,L-10):
   
        img1_idx = matches[i].queryIdx
        img2_idx = matches[i].trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        img1_idx = matches[i+5].queryIdx
        img2_idx = matches[i+5].trainIdx
        (x3,y3) = kp1[img1_idx].pt
        (x4,y4) = kp2[img2_idx].pt
        img1_idx = matches[i+10].queryIdx
        img2_idx = matches[i+10].trainIdx
        (x5,y5) = kp1[img1_idx].pt
        (x6,y6) = kp2[img2_idx].pt
        A=np.array([[x1, y1, 1, 0, 0, 0],[0, 0, 0, x1, y1, 1], [x3, y3, 1, 0, 0, 0],[0, 0, 0, x3, y3, 1],[x5, y5, 1, 0, 0, 0],[0, 0, 0, x5, y5, 1]])
        b=([[x2],[y2],[x4],[y4],[x6],[y6]])
        vector=np.dot(np.linalg.inv(A),b)
        
        transV=np.reshape(vector,(2,3))
        t.append(transV)
       
        good[i]=[matches[i]]
        for j in range (0,L):
            
            img1_idxj = matches[j].queryIdx
            img2_idxj = matches[j].trainIdx
            (x1j,y1j) = kp1[img1_idxj].pt
            (x2j,y2j) = kp2[img2_idxj].pt
            simAxis=np.array([[x1j],[y1j],[1]])
            axis_com=np.dot(transV, simAxis)
            
            d=np.sqrt((x2j-axis_com[0][0])**2+(y2j-axis_com[1][0])**2)
            
            if d<=tolerance:
                good[i].append(matches[j])
                
                n[i]=n[i]+1
           
        if n[i]>max:
            max=n[i]
            good_matches=good[i]
            transform=t[i]
            
   
    print('length of good match for affine simA and simB is ', len(good_matches)-1)
   
    return transform, good_matches

# Driver code
def main():
    # Note: Comment out parts of this code as necessary
 
    # 1a
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    transA_Ix = gradientX(transA)  # TODO: implement this
    transA_Iy = gradientY(transA)  # TODO: implement this
    transA_pair = make_image_pair(transA_Ix, transA_Iy)  # TODO: implement this
    h,w=transA_pair.shape[:2]
    dst=np.zeros([h,w])
    transA_pair=cv2.normalize(transA_pair,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-1.png"), transA_pair)  # Note: you may have to scale/type-cast image before writing
    
    # TODO: Similarly for simA.jpg
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    simA_Ix = gradientX(simA)  # TODO: implement this
    simA_Iy = gradientY(simA)  # TODO: implement this
    simA_pair = make_image_pair(simA_Ix, simA_Iy)  # TODO: implement this
    h,w=simA_pair.shape[:2]
    dst=np.zeros([h,w])
    simA_pair=cv2.normalize(simA_pair,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-2.png"), simA_pair)  

    # 1b
    transA_R = harris_response(transA_Ix, transA_Iy, np.ones((3, 3), dtype=np.float_) / 9.0, 0.04)  # TODO: implement this, tweak parameters for best response
    # TODO: Scale/type-cast response map and write to file
    h,w=transA_R.shape[:2]
    dst=np.zeros([h,w])
    transA_R=cv2.normalize(transA_R,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-1.png"), transA_R)  

    # TODO: Similarly for transB, simA and simB (you can write a utility function for grouping operations on each image)
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    transB_Ix = gradientX(transB)  # TODO: implement this
    transB_Iy = gradientY(transB)  # TODO: implement this
    transB_R = harris_response(transB_Ix, transB_Iy, np.ones((3, 3), dtype=np.float_) / 9.0, 0.04)
    h,w=transB_R.shape[:2]
    dst=np.zeros([h,w])
    transB_R=cv2.normalize(transB_R,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-2.png"), transB_R)

    simA_R = harris_response(simA_Ix, simA_Iy, np.ones((3, 3), dtype=np.float_) / 9.0, 0.04)
    h,w=simA_R.shape[:2]
    dst=np.zeros([h,w])
    simA_R=cv2.normalize(simA_R,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-3.png"), simA_R)

    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    simB_Ix = gradientX(simB)  # TODO: implement this
    simB_Iy = gradientY(simB)  # TODO: implement this
    simB_R = harris_response(simB_Ix, simB_Iy, np.ones((3, 3), dtype=np.float_) / 9.0, 0.04)
    h,w=simB_R.shape[:2]
    dst=np.zeros([h,w])
    simB_R=cv2.normalize(simB_R,dst,0,255,cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-4.png"), simB_R)


    # 1c
    transA_corners = find_corners(transA_R, 120, 3)  # TODO: implement this, tweak parameters till you get good corners
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"))
    transA_out = draw_corners(transA, transA_corners)  # TODO: implement this
    # TODO: Write image to file
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-1.png"), transA_out)
    # TODO: Similarly for transB, simA and simB (write a utility function if you want)
    transB_corners = find_corners(transB_R, 90, 3)
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"))
    transB_out = draw_corners(transB, transB_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-2.png"), transB_out)

    simA_corners = find_corners(simA_R, 100, 3)
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"))
    simA_out = draw_corners(simA, simA_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-3.png"), simA_out)

    simB_corners = find_corners(simB_R, 110, 3)
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"))
    simB_out = draw_corners(simB, simB_corners)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-4.png"), simB_out)

    # 2a
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"))
    transA_angle = gradient_angle(transA_Ix, transA_Iy)  # TODO: implement this
   
    transA_kp = get_keypoints(transA_corners, transA_R, transA_angle, _size=10.0, _octave=0)  # TODO: implement this, update parameters
    
    # TODO: Draw keypoints on transA
 
    out_imgA = cv2.drawKeypoints(transA, transA_kp, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # TODO: Similarly, find keypoints for transB and draw them
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"))
    transB_angle = gradient_angle(transB_Ix, transB_Iy)
    transB_kp = get_keypoints(transB_corners, transB_R, transB_angle, _size=10.0, _octave=0)
    out_imgB = cv2.drawKeypoints(transB, transB_kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    temp=make_image_pair(out_imgA,out_imgB)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-1.png"), temp)
    # TODO: Combine transA and transB images (with keypoints drawn) using make_image_pair() and write to file
    
    # TODO: Ditto for (simA, simB) pair
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"))
    simA_angle = gradient_angle(simA_Ix, simA_Iy)  # TODO: implement this
    simA_kp = get_keypoints(simA_corners, simA_R, simA_angle, _size=10.0, _octave=0)  # TODO: implement this, update parameters
    
    # TODO: Draw keypoints on transA
 
    out_imgA = cv2.drawKeypoints(simA, simA_kp, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # TODO: Similarly, find keypoints for transB and draw them
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"))
    simB_angle = gradient_angle(simB_Ix, simB_Iy)
    simB_kp = get_keypoints(simB_corners, simB_R, simB_angle, _size=10.0, _octave=0)
    out_imgB = cv2.drawKeypoints(simB, simB_kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    temp=make_image_pair(out_imgA,out_imgB)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-2.png"), temp)

    # 2b
    
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"),cv2.IMREAD_GRAYSCALE)
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"),cv2.IMREAD_GRAYSCALE)
    transA_kpN,transA_desc = get_descriptors(transA, transA_kp)  # TODO: implement this
    # TODO: Similarly get transB_desc
    transB_kpN, transB_desc = get_descriptors(transB, transB_kp)                
    # TODO: Find matches: trans_matches = match_descriptors(transA_desc, transB_desc)
    trans_matches = match_descriptors(transA_desc, transB_desc)
    
    # TODO: Draw matches and write to file: draw_matches(transA, transB, transA_kp, transB_kp, trans_matches)
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"))
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"))
    imgOut=draw_matches(transA, transB, transA_kpN, transB_kpN, trans_matches)
   
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-1.png"), imgOut)
    # TODO: Ditto for (simA, simB) pair (may have to vary some parameters along the way?)
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"))
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"))
    simA_kpN,simA_desc = get_descriptors(simA, simA_kp)  # TODO: implement this
    # TODO: Similarly get transB_desc
    simB_kpN,simB_desc = get_descriptors(simB, simB_kp)                
    # TODO: Find matches: trans_matches = match_descriptors(transA_desc, transB_desc)
    sim_matches = match_descriptors(simA_desc, simB_desc)
    # TODO: Draw matches and write to file: draw_matches(transA, transB, transA_kp, transB_kp, trans_matches)
    imgOut=draw_matches(simA, simB, simA_kpN, simB_kpN, sim_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-2.png"), imgOut)
    
    # 3a
    # TODO: Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
    translation, good_matches=compute_translation_RANSAC(transA_kpN, transB_kpN, trans_matches, 20)
    print('translation vector for transA  transB is', ' ', translation)
    imgOut=draw_matches(transA, transB, transA_kpN, transB_kpN, good_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-a-1.png"), imgOut)
    
    
    # 3b
    # TODO: Compute similarity transform for (simA, simB) pair, draw biggest consensus set
    transform, good_matches2=compute_similarity_RANSAC(simA_kpN, simB_kpN, sim_matches, 20)
    print('transform is ',  transform)
    imgOut=draw_matches(simA, simB, simA_kpN, simB_kpN, good_matches2)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-b-1.png"), imgOut)

    # Extra credit: 3c, 3d, 3e
    affine_transform, good_matches3=compute_affine_RANSAC(simA_kpN, simB_kpN, sim_matches, 10)
    print('affine transform is ',  affine_transform)
    imgOut=draw_matches(simA, simB, simA_kpN, simB_kpN, good_matches3)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-c-1.png"), imgOut)

    #3d
    
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"),cv2.IMREAD_GRAYSCALE)
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"),cv2.IMREAD_GRAYSCALE)
    h,w=simB.shape[:2]
    simB_arr=np.array(simB)
    r,c=np.shape(simB_arr)
    warpedB=np.zeros([int(1.4*h),int(1.2*w)])
    M=np.append(transform,[[0,0,1]],axis=0)
    M_inv=np.linalg.inv(M)
    X,Y=np.mgrid[0:w,0:h]
    x_A=M_inv[0][0]*X+M_inv[0][1]*Y+M_inv[0][2]
    y_A=M_inv[1][0]*X+M_inv[1][1]*Y+M_inv[1][2]
    xmin=np.ceil((np.absolute(np.amin(x_A))))
    ymin=np.ceil((np.absolute(np.amin(y_A))))
    for y in range (0,h):
        for x in range(0,w):
            warpedB[np.round(y_A[x,y]+ymin),np.round(x_A[x,y]+xmin)]=simB_arr[y][x]
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-1.png"), warpedB)

    
    
    simA_arr=np.array(simA)
    h,w=np.shape(warpedB)
    ha,wa=np.shape(simA_arr)
    img_color=np.zeros([h,w,3])
    img_color[ymin:ymin+ha,xmin:xmin+wa,2]=simA_arr
    img_color[0:h,0:w,1]=warpedB
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-2.png"), img_color)

    #3e:
    h,w=simB.shape[:2]
    simB_arr=np.array(simB)
    r,c=np.shape(simB_arr)
    warpedB=np.zeros([int(1.3*h),int(1.2*w)])
    M=np.append(affine_transform,[[0,0,1]],axis=0)
    M_inv=np.linalg.inv(M)
    
    X,Y=np.mgrid[0:w,0:h]
    x_A=M_inv[0][0]*X+M_inv[0][1]*Y+M_inv[0][2]
    y_A=M_inv[1][0]*X+M_inv[1][1]*Y+M_inv[1][2]
    xmin=np.ceil((np.absolute(np.amin(x_A))))
    ymin=np.ceil((np.absolute(np.amin(y_A))))
    for y in range (0,h):
        for x in range(0,w):
            warpedB[np.round(y_A[x,y]+ymin),np.round(x_A[x,y]+xmin)]=simB_arr[y][x]
    cv2.imwrite(os.path.join(output_dir, "ps5-3-e-1.png"), warpedB)

    
    
    simA_arr=np.array(simA)
    h,w=np.shape(warpedB)
    ha,wa=np.shape(simA_arr)
    img_color=np.zeros([h,w,3])
    img_color[ymin:ymin+ha,xmin:xmin+wa,2]=simA_arr
    img_color[0:h,0:w,1]=warpedB
    cv2.imwrite(os.path.join(output_dir, "ps5-3-e-2.png"), img_color)


if __name__ == "__main__":
    main()
