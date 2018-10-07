"""Problem Set 8: Motion History Images."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class MotionHistoryBuilder(object):
    """Builds a motion history image (MHI) from sequential video frames."""

    def __init__(self, frame, **kwargs):
        """Initialize motion history builder object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            kwargs: additional keyword arguments needed by builder, e.g. theta, tau
        """
        # TODO: Your code here - initialize variables, read keyword arguments (if any)
        self.frame=frame
        self.mhi = np.zeros(frame.shape[:2], dtype=np.float_)  # e.g. motion history image
        self.theta=kwargs.get('theta',10)
        self.tau=kwargs.get('tau', 30)
    def process(self, frame):
        """Process a frame of video, return binary image indicating motion areas.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]

        Returns
        -------
            motion_image: binary image (type: bool or uint8), values: 0 (static) or 1 (moving)
        """
        # TODO: Your code here - compute binary motion image, update MHI
        frame0=self.frame
        frame0= cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame0=frame0.astype(np.float32)
        frame0_MBlur=cv2.medianBlur(frame0, 5)
        frame0_Blur= cv2.GaussianBlur(frame0_MBlur,(5,5),5)
        
        frame1= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1=frame1.astype(np.float32)
        frame1_MBlur=cv2.medianBlur(frame1, 5)
        frame1_Blur=cv2.GaussianBlur(frame1_MBlur,(5,5),5)
        
        r,c=frame0.shape[:2]
        thetaArr=np.ones([r,c])*self.theta
        diffArr=np.zeros([r,c])
        diffArr=np.absolute(frame1_Blur-frame0_Blur)
        motion_image = np.greater_equal(diffArr, thetaArr)
        motion_image.astype(np.float32)
        self.frame = frame

        
        motion_image1=motion_image*self.tau
        boolean=np.equal(motion_image1, np.zeros([r,c]))
        onesArr=np.ones([r,c])
        temp=np.maximum(self.mhi-onesArr,np.zeros([r,c]))
        temp=temp*boolean
        self.mhi=motion_image1+temp
        
        #print("11--", np.amax(motion_image), np.amin(motion_image))
        return motion_image  # note: make sure you return a binary image with 0s and 1s
        
    def get_MHI(self):
        """Return motion history image computed so far.

        Returns
        -------
            mhi: float motion history image, values in [0.0, 1.0]
        """
        # TODO: Make sure MHI is updated in process(), perform any final steps here (e.g. normalize to [0, 1])
        # Note: This method may not be called for every frame (typically, only once)
        mhi=self.mhi/self.tau
        return mhi


class Moments(object):
    """Spatial moments of an image - unscaled and scaled."""

    def __init__(self, image):
        """Compute spatial moments on given image.

        Parameters
        ----------
            image: single-channel image, uint8 or float
        """
        # TODO: Your code here - compute all desired moments here (recommended)
        self.central_moments = None  # array: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]
        self.scaled_moments = None   # array: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]
        # Note: Make sure computed moments are in correct order
        M00=np.sum(image)
        rm,cm=np.shape(image)
        X_vector=np.linspace(0,cm-1, cm)
        X_array=np.tile(X_vector, (rm,1))
        Y=np.linspace(0,rm-1, rm)
        Y_vector=np.reshape(Y,(rm,1))
        Y_array=np.tile(Y_vector, (1,cm))
        M10=np.sum(image*X_array)
        M01=np.sum(image*Y_array)
        x_ave=M10/M00
        y_ave=M01/M00
        X_diff=X_array-x_ave
        X_diff2=X_diff**2
        X_diff1=X_diff**1
        X_diff0=X_diff**0
        X_diff3=X_diff**3
        Y_diff=Y_array-y_ave
        Y_diff0=Y_diff**0
        Y_diff1=Y_diff**1
        Y_diff2=Y_diff**2
        Y_diff3=Y_diff**3
        Mu20=np.sum(X_diff2*Y_diff0*image)
        Mu11=np.sum(X_diff1*Y_diff1*image)
        Mu02=np.sum(X_diff0*Y_diff2*image)
        Mu21=np.sum(X_diff2*Y_diff1*image)
        Mu12=np.sum(X_diff1*Y_diff2*image)
        Mu22=np.sum(X_diff2*Y_diff2*image)
        Mu30=np.sum(X_diff3*Y_diff0*image)
        Mu03=np.sum(X_diff0*Y_diff3*image)
        self.central_moments=np.array([Mu20,Mu11,Mu02,Mu21,Mu12,Mu22,Mu30,Mu03])

        Mu00=np.sum(X_diff0*Y_diff0*image)
        Mu00_20=Mu00**(1+(2+0)/2)
        Mu20_scal=Mu20/Mu00_20
        Mu00_11=Mu00**(1+(1+1)/2)
        Mu11_scal=Mu11/Mu00_11
        Mu00_02=Mu00**(1+(2+0)/2)
        Mu02_scal=Mu02/Mu00_02
        Mu00_21=Mu00**(1+(2+1)/2)
        Mu21_scal=Mu21/Mu00_21
        Mu00_12=Mu00**(1+(2+1)/2)
        Mu12_scal=Mu12/Mu00_12
        Mu00_22=Mu00**(1+(2+2)/2)
        Mu22_scal=Mu22/Mu00_22
        Mu00_30=Mu00**(1+(3+0)/2)
        Mu30_scal=Mu30/Mu00_30
        Mu00_03=Mu00**(1+(2+0)/2)
        Mu03_scal=Mu03/Mu00_03
        self.scaled_moments=np.array([Mu20_scal,Mu11_scal,Mu02_scal,Mu21_scal,Mu12_scal,Mu22_scal,Mu30_scal,Mu03_scal])
        
        
    def get_central_moments(self):
        """Return central moments as NumPy array.

        Order: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]

        Returns
        -------
            self.central_moments: float array of central moments
        """
        return self.central_moments

    def get_scaled_moments(self):
        """Return scaled central moments as NumPy array.

        Order: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]

        Returns
        -------
            self.scaled_moments: float array of scaled central moments
        """
        return self.scaled_moments  # note: make sure moments are in correct order


def compute_feature_difference(a_features, b_features):
    """Compute feature difference between two videos.

    Parameters
    ----------
        a_features: feaures from one video, MHI & MEI moments in a 16-element 1D array
        b_features: like a_features, from other video
   
    Returns
    -------
        diff: a single float value, difference between the two feature vectors
    """
    # TODO: Your code here - return feature difference using an appropriate measure
    # Tip: Scale/weight difference values to get better results as moment magnitudes differ
    diff=np.sqrt(np.sum((a_features-b_features)**2))
    return diff


# Driver/helper code
def build_motion_history_image(builder_class, video_filename, save_frames={}, mhi_frame=None, mhi_filename=None, **kwargs):
    """Instantiate and run a motion history builder on a given video, return MHI.

    Creates an object of type builder_class, passing in initial video frame,
    and any additional keyword arguments.

    Parameters
    ----------
        builder_class: motion history builder class to instantiate
        video_filename: path to input video file
        save_frames: output binary motion images to save {<frame number>: <filename>}
        mhi_frame: which frame to obtain the motion history image at
        mhi_filename: output filename to save motion history image
        kwargs: arbitrary keyword arguments passed on to constructor

    Returns
    -------
        mhi: float motion history image generated by builder, values in [0.0, 1.0]
    """

    # Open video file
    video = cv2.VideoCapture(video_filename)
    print("Video: {} ({}x{}, {:.2f} fps, {} frames)".format(
        video_filename,
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        video.get(cv2.CAP_PROP_FPS),
        int(video.get(cv2.CAP_PROP_FRAME_COUNT))))

    # Initialize objects
    mhi_builder = None
    mhi = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Initialize motion history builder (one-time only)
            if mhi_builder is None:
                mhi_builder = builder_class(frame, **kwargs)

            # Process frame
            motion_image = mhi_builder.process(frame)  # TODO: implement this!

            # Save output, if indicated
            if frame_num in save_frames:
                cv2.imwrite(save_frames[frame_num], np.uint8(motion_image * 255))  # scale [0, 1] => [0, 255]
            
            # Grab MHI, if indicated
            if frame_num == mhi_frame:
                mhi = mhi_builder.get_MHI()
                print("MHI frame: {}".format(mhi_frame))
                break  # uncomment for early stop

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

    # If not obtained earlier, get MHI now
    if mhi is None:
        mhi = mhi_builder.get_MHI()

    # Save MHI, if filename is given
    if mhi_filename is not None:
        cv2.imwrite(mhi_filename, np.uint8(mhi * 255))  # scale [0, 1] => [0, 255]

    return mhi


def match_features(a_features_dict, b_features_dict, n_actions):
    """Compare features, tally matches for each action pair to produce a confusion matrix.

    Note: Skips comparison for keys that are identical in the two dicts.

    Parameters
    ----------
        a_features_dict: one set of features, as a dict with key: (<action>, <participant>, <trial>)
        b_features_dict: another set of features like a_features
        n_actions: number of distinct actions present in the feature sets

    Returns
    -------
        confusion_matrix: table of matches found, n_actions by n_actions
    """
    
    confusion_matrix = np.zeros((n_actions, n_actions), dtype=np.float_)
    for a_key, a_features in a_features_dict.items():
        min_diff = np.inf
        best_match = None
        for b_key, b_features in b_features_dict.items():
            if a_key == b_key:
                continue  # don't compare with yourself!
            diff = compute_feature_difference(a_features, b_features)  # TODO: implement this!
            if diff < min_diff:
                min_diff = diff
                best_match = b_key
        if best_match is not None:
            #print("{} matches {}, diff: {}".format(a_key, best_match, min_diff))  # [debug]
            confusion_matrix[a_key[0] - 1, best_match[0] - 1] += 1  # note: 1-based to 0-based indexing

    confusion_matrix /= confusion_matrix.sum(axis=1)[:, np.newaxis] # normalize confusion_matrix along each row
    return confusion_matrix


def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
        os.path.join(input_dir, "PS8A1P1T1.avi"),  # input video
        save_frames={
            10: os.path.join(output_dir, 'ps8-1-a-1.png'),
            20: os.path.join(output_dir, 'ps8-1-a-2.png'),
            30: os.path.join(output_dir, 'ps8-1-a-3.png')
        }, theta=6.0, tau=30)  # output motion images to save, mapped to filenames
    # TODO: Specify any other keyword args that your motion history builder expects, e.g. theta, tau

    # 1b
    build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
        os.path.join(input_dir, "PS8A1P1T1.avi"),  # TODO: choose sequence (person, trial) for action A1
        mhi_frame=95,  # TODO: pick a good frame to obtain MHI at, i.e. when action just ends
        mhi_filename=os.path.join(output_dir, 'ps8-1-b-1.png'),theta=12.0, tau=35)
    # TODO: Specify any other keyword args that your motion history builder expects, e.g. theta, tau

    # TODO: Similarly for actions A2 & A3
    build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
        os.path.join(input_dir, "PS8A2P1T1.avi"),  # TODO: choose sequence (person, trial) for action A1
        mhi_frame=62,  # TODO: pick a good frame to obtain MHI at, i.e. when action just ends
        mhi_filename=os.path.join(output_dir, 'ps8-1-b-2.png'),theta=12.0, tau=35)
    build_motion_history_image(MotionHistoryBuilder,  # motion history builder class
        os.path.join(input_dir, "PS8A3P1T1.avi"),  # TODO: choose sequence (person, trial) for action A1
        mhi_frame=95,  # TODO: pick a good frame to obtain MHI at, i.e. when action just ends
        mhi_filename=os.path.join(output_dir, 'ps8-1-b-3.png'),theta=12.0, tau=35)

    # 2a
    # Compute MHI and MEI features (unscaled and scaled central moments) for each video
    central_moment_features = {}  # 16 features (8 MHI, 8 MEI) as one vector, key: (<action>, <participant>, <trial>)
    scaled_moment_features = {}  # similarly, scaled central moments
    
    default_params = dict(mhi_frame=60)  # params for build_motion_history(), overriden by custom_params for specified videos
    # Note: To specify custom parameters for a video, add to the dict below:
    #   (<action>, <participant>, <trial>): dict(<param1>=<value1>, <param2>=<value2>, ...)
    custom_params = {
        (1, 1, 1): dict(mhi_frame=98,tau=68,theta=12),
        (1, 1, 2): dict(mhi_frame=85,tau=59,theta=12),  # PS8A1P1T3.avi
        (1, 1, 3): dict(mhi_frame=100,tau=70,theta=12),  # PS8A1P2T3.avi
        (1, 2, 1): dict(mhi_frame=68,tau=47,theta=12),
        (1, 2, 2): dict(mhi_frame=58,tau=40,theta=12),
        (1, 2, 3): dict(mhi_frame=62,tau=43,theta=12),
        (1, 3, 1): dict(mhi_frame=76,tau=53,theta=12),
        (1, 3, 2): dict(mhi_frame=73,tau=51,theta=12),  
        (1, 3, 3): dict(mhi_frame=70,tau=49,theta=12),  
        (2, 1, 1): dict(mhi_frame=64,tau=45,theta=12),
        (2, 1, 2): dict(mhi_frame=75,tau=52,theta=12),
        (2, 1, 3): dict(mhi_frame=76,tau=52,theta=12),
        (2, 2, 1): dict(mhi_frame=55,tau=38,theta=12),
        (2, 2, 2): dict(mhi_frame=58,tau=40,theta=12),
        (2, 2, 3): dict(mhi_frame=66,tau=46,theta=12),
        (2, 3, 1): dict(mhi_frame=71,tau=49,theta=12),
        (2, 3, 2): dict(mhi_frame=69,tau=48,theta=12),
        (2, 3, 3): dict(mhi_frame=73,tau=51,theta=12),
        (3, 1, 1): dict(mhi_frame=103,tau=72,theta=12),
        (3, 1, 2): dict(mhi_frame=91,tau=64,theta=12),
        (3, 1, 3): dict(mhi_frame=86,tau=60,theta=12),
        (3, 2, 1): dict(mhi_frame=68,tau=47,theta=12),
        (3, 2, 2): dict(mhi_frame=76,tau=53,theta=12),
        (3, 2, 3): dict(mhi_frame=72,tau=50,theta=12),
        (3, 3, 1): dict(mhi_frame=70,tau=49,theta=12),
        (3, 3, 2): dict(mhi_frame=84,tau=59,theta=12),
        (3, 3, 3): dict(mhi_frame=81,tau=57,theta=12),
        
    }

    # Loop for each action, participant, trial
    n_actions = 3
    n_participants = 3
    n_trials = 3
    print("Computing features for each video...")
    for a in range(1, n_actions + 1):  # actions
        for p in range(1, n_participants + 1):  # participants
            for t in range(1, n_trials + 1):  # trials
                video_filename = os.path.join(input_dir, "PS8A{}P{}T{}.avi".format(a, p, t))
                mhi = build_motion_history_image(MotionHistoryBuilder, video_filename, **dict(default_params, **custom_params.get((a, p, t), {})))
                #cv2.imshow("MHI: PS8A{}P{}T{}.avi".format(a, p, t), mhi)  # [debug]
                #cv2.waitKey(1)  # uncomment if using imshow
                mei = np.uint8(mhi > 0)
                mhi_moments = Moments(mhi)
                mei_moments = Moments(mei)
                central_moment_features[(a, p, t)] = np.hstack((mhi_moments.get_central_moments(), mei_moments.get_central_moments()))
                scaled_moment_features[(a, p, t)] = np.hstack((mhi_moments.get_scaled_moments(), mei_moments.get_scaled_moments()))

    # Match features in a leave-one-out scheme (each video with all others)
    central_moments_confusion = match_features(central_moment_features, central_moment_features, n_actions)
    print("Confusion matrix (unscaled central moments):-")
    print(central_moments_confusion)

    # Similarly with scaled moments
    scaled_moments_confusion = match_features(scaled_moment_features, scaled_moment_features, n_actions)
    print("Confusion matrix (scaled central moments):-")
    print(scaled_moments_confusion)

    # 2b
    # Match features by testing one participant at a time (i.e. taking them out)
    # Note: Pick one between central_moment_features and scaled_moment_features
    features_P1 = {key: feature for key, feature in central_moment_features.items() if key[1] == 1}
    features_sans_P1 = {key: feature for key, feature in central_moment_features.items() if key[1] != 1}
    confusion_P1 = match_features(features_P1, features_sans_P1, n_actions)
    print("Confusion matrix for P1:-")
    print(confusion_P1)

    # TODO: Similarly for participants P2 & P3
    features_P2 = {key: feature for key, feature in central_moment_features.items() if key[1] == 2}
    features_sans_P2 = {key: feature for key, feature in central_moment_features.items() if key[1] != 2}
    confusion_P2 = match_features(features_P2, features_sans_P2, n_actions)
    print("Confusion matrix for P2:-")
    print(confusion_P2)
    features_P3 = {key: feature for key, feature in central_moment_features.items() if key[1] == 3}
    features_sans_P3 = {key: feature for key, feature in central_moment_features.items() if key[1] != 3}
    confusion_P3 = match_features(features_P3, features_sans_P3, n_actions)
    print("Confusion matrix for P3:-")
    print(confusion_P3)

    # Note: Feel free to modify this driver function, but do not modify the interface for other functions/methods!


if __name__ == "__main__":
    main()
