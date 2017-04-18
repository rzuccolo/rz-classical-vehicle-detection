import numpy as np
import cv2
from skimage.feature import hog



# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    img: (M, N, H) ndarray
    bboxes: bounding box positions, [((x1, y1), (x2, y2)), ((,),(,)), ...]
    color: optional 3-tuple, for example, (0, 0, 255) for blue
    thick: optional integer parameter to define the box thickness
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to compute color histogram features 
def color_hist(img, nbins=32):
    '''
    img: (M, N, H) ndarray
    
    nbins: optional, bins is an int, it defines the number of equal-width bins in the given range
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    '''
    img: (M, N, H) ndarray
    
    size: optional 2-tuple, output image size
    '''
    # Use cv2.resize().ravel() to create the feature vector
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    # Return the feature vector
    return np.hstack((color1, color2, color3))


# Define a function to return HOG (Histogram of Oriented Gradient) features and visualization
def get_hog_features(img, orient=9, pix_per_cell=(8,8), cell_per_block=(8,8), 
                        vis=False, feature_vec=True):
    '''
    img: (M, N) ndarray,  single color channel or grayscaled image
    
    orient: optional, integer, and represents the number of orientation bins that the gradient information
            will be split up into in the histogram. Typical values are between 6 and 12 bins
            
    pix_per_cell: optional 2-tuple, cell size over which each gradient histogram is computed. This paramater
                  is passed as a 2-tuple so you could have different cell sizes in x and y, but cells are
                  commonly chosen to be square
                   
    cell_per_block: optional 2-tuple, and specifies the local area over which the histogram counts in a given
                    cell will be normalized. Block normalization is not necessarily required, but generally
                    leads to a more robust feature set
                    
    vis: flag tells the function to output a visualization of the HOG feature computation or not
    
    feature_vec: flag tells the function to unroll the feature array into a feature vector using same as
                 features.ravel() would do, which yields, in this case, to a one dimensional array
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=pix_per_cell,
                                  cells_per_block=cell_per_block, 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:         
        features = hog(img, orientations=orient, 
                   pixels_per_cell=pix_per_cell,
                   cells_per_block=cell_per_block, 
                   transform_sqrt=True, 
                   visualise=vis, feature_vector=feature_vec)
            
        return features

    
    
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', hist_bins=32,
                     spatial_size=(32, 32),
                     orient=9, pix_per_cell=(8,8), cell_per_block=(2,2), hog_channel=0,
                     hist_feat=True, spatial_feat=True, hog_feat=True):
    '''
    imgs:  list of image filenames
    
    color_space: optional for color feature extraction,'RGB', 'HSV', 'LUV', 'HLS', 'YUV' or 'YCrCb' 
    
    hist_bins: optional for color feature extraction, bins is an int, it defines the number of
               equal-width bins in the given range
               
    spatial_size: optional for spatial bining, 2-tuple, spatial binning output image size
    
    orient: optional for HOG feature extraction, integer, and represents the number of orientation bins
            that the gradient information will be split up into in the histogram. Typical values are
            between 6 and 12 bins
            
    pix_per_cell: optional 2-tuple, for HOG feature extraction, cell size over which each gradient histogram is computed.
                  This paramater is passed as a 2-tuple so you could have different cell sizes in x and y,
                  but cells are commonly chosen to be square
                  
    cell_per_block: optional 2-tuple, for HOG feature extraction, specifies the local area over which the
                    histogram counts in a given cell will be normalized. Block normalization is not 
                    necessarily required, but generally leads to a more robust feature set
                    
    hog_channel: optional for HOG feature extraction, which channel t apply HOG: 0, 1, 2 or "ALL"
    
    hist_feat: flag to apply or not color histogram feature extraction
    
    spatial_feat: flag to apply or not spatial binning
    
    hog_feat: flag to apply or not HOG feature extraction
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #print('len spatial_features in extract_features',len(spatial_features))
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            #print('len hist_features in extract_features',len(hist_features))
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                #print('len hog_features in extract_features',len(hog_features))
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Convert color spaces
def convert_color(img, conv='RGB2YCrCb'):
    '''
    img: (M, N, H) ndarray
    conv: 'RGB2YCrCb', 'RGB2LUV', 'RGB2HSV', 'RGB2HLS' or 'RGB2YUV'
    '''
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# Function that adds "heat" to a map for a list of bounding boxes
def add_heat(heatmap, bbox_list):
    '''
    heatmap: mask (1 channel) of original image
    bbox_list: bounding box positions, [((x1, y1), (x2, y2)), ((,),(,)), ...]
    '''
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


# imposing a threshold, to reject areas affected by false positives  
def apply_threshold(heatmap, threshold):
    '''
    heatmap: mask (1 channel) of original image
    threshold: reject areas affected by false positives
    '''
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# Take labels image and put bounding boxes around the labeled regions
def draw_labeled_bboxes(img, labels):
    '''
    img: (M, N, H) ndarray
    labels: 2-tuple, where the first item is an array the size of the heatmap input image,
            and the second element is the number of labels (cars) found
    '''
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        cv2.putText(img,"Vehicle: " + "{:0.0f}".format(car_number), org=(bbox[0][0],bbox[0][1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    # Return the image
    return img  