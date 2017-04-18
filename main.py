from collections import deque
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import cProfile

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

# Program local libraries
from line import Line
from load_parameters import load_camera_mtx_dist_from_pickle as load_mtx_dist
from load_parameters import load_perspective_transform_from_pickle as load_M_Minv
from load_parameters import load_vehicle_detection_classifier_from_pickle as load_vehicle_classifier
from warp_transformer import thresholding
from vehicle_detection_helpers import *



def sliding_window(binary_warped):
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype(np.uint8)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), color=(0,255,0), thickness=2) # Green
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), color=(0,255,0), thickness=2) # Green
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #print(left_fit) # to measure tolerances
    
    # Stash away polynomials
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img[ploty.astype('int'),left_fitx.astype('int')] = [0, 255, 255]
    out_img[ploty.astype('int'),right_fitx.astype('int')] = [0, 255, 255]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

    # Calculate radii of curvature in meters
    y_eval = np.max(ploty)  # Where radius of curvature is measured
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Stash away the curvatures  
    left_line.radius_of_curvature = left_curverad  
    right_line.radius_of_curvature = right_curverad
    
    return left_fit, right_fit, left_curverad, right_curverad, out_img

    
    
def non_sliding(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
        & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
        & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
    except:
        return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
    
    else:
        # Check difference in fit coefficients between last and new fits  
        left_line.diffs = left_line.current_fit - left_fit
        right_line.diffs = right_line.current_fit - right_fit
        if (left_line.diffs[0]>0.001 or left_line.diffs[1]>0.4 or left_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
        #print(left_line.diffs)
        if (right_line.diffs[0]>0.001 or right_line.diffs[1]>0.4 or right_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
        #print(right_line.diffs)
        
        # Stash away polynomials
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Calculate radii of curvature in meters
        y_eval = np.max(ploty)  # Where radius of curvature is measured
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])     

        # Stash away the curvatures  
        left_line.radius_of_curvature = left_curverad  
        right_line.radius_of_curvature = right_curverad

        return left_fit, right_fit, left_curverad, right_curverad, None
    
     
    
    
def draw_lane(undistorted, binary_warped, left_fit, right_fit, left_curverad, right_curverad):
    
    # Create an image to draw the lines on
    warped_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]   
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    midpoint = np.int(undistorted.shape[1]/2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warped, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undistorted.shape[1], undistorted.shape[0])
    unwarped = cv2.warpPerspective(color_warped, Minv, img_size, flags=cv2.INTER_LINEAR)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, unwarped, 0.3, 0)
    radius = np.mean([left_curverad, right_curverad])

    # Add radius and offset calculations to top of video
    cv2.putText(result,"L. Lane Radius: " + "{:0.2f}".format(left_curverad/1000) + 'km', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"R. Lane Radius: " + "{:0.2f}".format(right_curverad/1000) + 'km', org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"C. Position: " + "{:0.2f}".format(offset) + 'm', org=(50,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)

    return result


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler,
              orient, pix_per_cell, cell_per_block,
              spatial_size,
              hist_bins,
              window):
    '''
    img: (M, N, H) ndarray
    
    ystart: y pixel coordinate at top of ROI
    
    ystop: y pixel coordinate at bottom of ROI
    
    scale: this scale factor will divide the ROI dimensions. Use >=1, where >1 will decreased dimensions
    
    svc: support vector classifier, example: LinearSVC
    
    X_scaler: normalized feature vectors
    
    orient: integer, and represents the number of orientation bins
            that the gradient information will be split up into in the histogram. Typical values are
            between 6 and 12 bins
            
    pix_per_cell: 2-tuple, cell size over which each gradient histogram is computed.
                  This paramater is passed as a 2-tuple so you could have different cell sizes in x and y,
                  but cells are commonly chosen to be square
                  
    cell_per_block: 2-tuple, specifies the local area over which the
                    histogram counts in a given cell will be normalized. Block normalization is not 
                    necessarily required, but generally leads to a more robust feature set 
    
    spatial_size: for spatial bining, 2-tuple, spatial binning output image size
    
    hist_bins: for color feature extraction, bins is an int, it defines the number of
               equal-width bins in the given range
               
    window: 2-tuple, window searching size
    '''
    draw_img = np.copy(img)
   
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    #ch2 = ctrans_tosearch[:,:,1]
    #ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell[0])-1
    nyblocks = (ch1.shape[0] // pix_per_cell[1])-1 
    nfeat_per_block = orient*cell_per_block[0]*cell_per_block[1]
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = window
    nxblocks_per_window = (window[0] // pix_per_cell[0])-1 
    nyblocks_per_window = (window[1] // pix_per_cell[1])-1 
    cells_per_step = (2,2)  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nxblocks_per_window) // cells_per_step[0] + 1
    nysteps = (nyblocks - nyblocks_per_window) // cells_per_step[1] + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Initialize a list to append window positions to
    window_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step[1]
            xpos = xb*cells_per_step[0]
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
            #hog_feat2 = hog2[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
            #hog_feat3 = hog3[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
            #hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            hog_features = np.copy(hog_feat1)

            xleft = xpos*pix_per_cell[0]
            ytop = ypos*pix_per_cell[1]     

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window[1], xleft:xleft+window[0]], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))    
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features)).reshape(1, -1) )           
            
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_drawx = np.int(window[0]*scale)
                win_drawy = np.int(window[1]*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_drawx,ytop_draw+win_drawy+ystart),(0,0,255),6) 

                # Calculate window position
                startx = xbox_left
                endx = startx + win_drawx
                starty = ytop_draw + ystart
                endy = starty + win_drawy

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
                
    #return draw_img, window_list  
    return window_list  


# Multi Sliding Search
def multi_window(img):
    
    # Window 1 characteristics
    ystart = 400
    ystop = 700
    scale = 1.
    pix_per_cell=(40,30)
    window = (320,240)

    # Window searching
    box_list1 = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                           orient, pix_per_cell, cell_per_block,
                           spatial_size,
                           hist_bins,
                           window)
    
    # Window 2 characteristics
    ystart = 380
    ystop = 620
    scale = 1.
    pix_per_cell=(30,20)
    window = (240,160)

    # Window searching
    box_list2 = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                           orient, pix_per_cell, cell_per_block,
                           spatial_size,
                           hist_bins,
                           window)

    box_list1.extend(box_list2)    

    # Window 3 characteristics
    ystart = 380
    ystop = 536
    scale = 1.
    pix_per_cell=(20,13)
    window = (160,104)

    # Window searching
    box_list3 = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                           orient, pix_per_cell, cell_per_block,
                           spatial_size,
                           hist_bins,
                           window)

    box_list1.extend(box_list3)

    # Window 4 characteristics
    ystart = 400
    ystop = 490
    scale = 1.
    pix_per_cell=(10,9)
    window = (80,72)

    # Window searching
    box_list4 = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                           orient, pix_per_cell, cell_per_block,
                           spatial_size,
                           hist_bins,
                           window)

    box_list1.extend(box_list4)

    # Return bounding boxes
    return box_list1



# this will smooth the frames
# by DAVID A. VENTIMIGLIA
# http://davidaventimiglia.com/advanced_lane_lines.html
# https://github.com/dventimi/CarND-Advanced-Lane-Lines
def get_processor(nbins=10):
    bins = nbins
    l_params = deque(maxlen=bins)
    r_params = deque(maxlen=bins)
    l_radius = deque(maxlen=bins)
    r_radius = deque(maxlen=bins)
    weights = np.arange(1,bins+1)/bins  
    
    heat_map = deque(np.array([np.zeros(img_size).astype(np.float)]), maxlen=bins)

    def process_image(img):
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        binary_warped, binary_output2 = thresholding(undistorted, M)
        

        if len(l_params)==0:
            left_fit, right_fit, left_curverad, right_curverad, _ = sliding_window(binary_warped)
    
        else:
            left_fit, right_fit, left_curverad, right_curverad, _ = non_sliding(binary_warped,
                                                                    np.average(l_params,0,weights[-len(l_params):]),
                                                                    np.average(r_params,0,weights[-len(l_params):]))
            
            
        l_params.append(left_fit)
        r_params.append(right_fit)
        l_radius.append(left_curverad)
        r_radius.append(right_curverad)
        annotated_image1 = draw_lane(undistorted,
                                    binary_warped,
                                    np.average(l_params,0,weights[-len(l_params):]),
                                    np.average(r_params,0,weights[-len(l_params):]),
                                    np.average(l_radius,0,weights[-len(l_params):]),
                                    np.average(r_radius,0,weights[-len(l_params):]))
        
        # Window searching
        box_list = multi_window(img)
        
        # Heat mapping
        heat = np.zeros(img_size).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)
        
        # discard frame and use latest if nothing is identified
        heat_test = apply_threshold(heat,heat_threshold)
        if np.all(heat_test<=heat_threshold):
            heat_map.extend(np.array([heat_map[-1]]))
        else:
            heat_map.extend(np.array([heat]))
        
        # Apply threshold
        heat_map_avg = np.average(heat_map,0,weights[-len(heat_map):])
        heat_map_avg = apply_threshold(heat_map_avg,heat_threshold)
        
        # Final annotated image
        # Visualize the heatmap when displaying    
        heat_map_clip = np.clip(heat_map_avg, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heat_map_clip)
        cv2.putText(annotated_image1,"Close Vehicles: " + "{:0.0f}".format(labels[1]), org=(50,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
        annotated_image = draw_labeled_bboxes(np.copy(annotated_image1), labels)
        
        
        
        return annotated_image
    return process_image





if __name__ == '__main__':   
    
    # Load camera data
    mtx, dist = load_mtx_dist()
    M, Minv = load_M_Minv()
    
    # Initialize track objects to help evaluate good or bad frames
    left_line = Line()
    right_line = Line() 
    
    # Load vehicle detection classifier    
    svc, X_scaler, orient, spatial_size, hist_bins, color_space, hog_channel, spatial_feat, hist_feat, hog_feat, cell_per_block = load_vehicle_classifier()
    
    # Vehicle detection parameters
    img_size = (720, 1280)
    heat_threshold = 1
    
    # Generate annotated video using average of frames to smooth
    movie_output = 'output_images/annotated_project_video_avg.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    driving_clip = clip1.fl_image(get_processor(15))
    #driving_clip.write_videofile(movie_output, audio=False)
    
    # Run and measure performance
    #cProfile.run('driving_clip.write_videofile(movie_output, audio=False)', 'restats')    
    pr = cProfile.Profile()
    pr.enable()

    driving_clip.write_videofile(movie_output, audio=False)

    pr.disable()
    pr.print_stats(sort='time')
