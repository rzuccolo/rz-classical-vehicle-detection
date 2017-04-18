import pickle
from os.path import join

# Where did you save the camera calibration results? pickle files
calibration_outputs_dir = 'output_images/camera_cal' 

# Filename used to save the camera calibration result (mtx,dist)
calibration_mtx_dist_filename = 'camera_cal_dist_pickle.p'

# Filename used to save the perspective transform matrices (M, Minv)
calibration_M_Minv_filename = 'perspective_trans_matrices.p'

# Where did you save the vehicle detection classifier? pickle files
vehicle_classifier_outputs_dir = 'training_dataset' 

# Filename used to save the vehicle classifier data
vehicle_classifier_filename = 'trained_dist_pickle.p'


def load_camera_mtx_dist_from_pickle():
    '''
    Read in the saved camera matrix and distortion coefficients
    These are the arrays we calculated using cv2.calibrateCamera()
    '''
    
    dist_pickle = pickle.load( open( join(calibration_outputs_dir, calibration_mtx_dist_filename), "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    return mtx, dist


def load_perspective_transform_from_pickle():
    '''
    Read in the saved perspective transformation matrices
    These are the arrays we calculated using cv2.getPerspectiveTransform()
    '''
    
    dist_pickle = pickle.load( open( join(calibration_outputs_dir, calibration_M_Minv_filename), "rb" ) )
    M = dist_pickle["M"]
    Minv = dist_pickle["Minv"]
    
    return M, Minv


def load_vehicle_detection_classifier_from_pickle():
    '''
    Read in the saved vehice detection classifier and other data
    These classifier is from linera SVM
    '''
    
    dist_pickle = pickle.load( open( join(vehicle_classifier_outputs_dir, vehicle_classifier_filename), "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["X_scaler"]
    orient = dist_pickle["orient"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]
    hog_channel = dist_pickle["hog_channel"]
    spatial_feat = dist_pickle["spatial_feat"]
    hist_feat = dist_pickle["hist_feat"]
    hog_feat = dist_pickle["hog_feat"]
    cell_per_block = dist_pickle["cell_per_block"]
    
    return svc, X_scaler, orient, spatial_size, hist_bins, color_space, hog_channel, spatial_feat, hist_feat, hog_feat, cell_per_block