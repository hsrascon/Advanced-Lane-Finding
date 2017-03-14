
## Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners.png "Detect Corners"
[image2]: ./output_images/undistortion1.png "Undistorted"
[image3]: ./output_images/unwarp.png "Unwarp"
[image4]: ./output_images/test_undistortion1.png "Unidstortion Example"
[image5]: ./output_images/threshold1.png "Threshold Example"
[image6]: ./output_images/trapezoid.png "Trapezoid Area"
[image7]: ./output_images/unwrap_trapezoid.png "UnWarp Trapezoid Area"
[image8]: ./output_images/perspective_transform1.png "Perspective Transform Example"
[image9]: ./output_images/windows1.png "Sliding Windows Example"
[image10]: ./output_images/windows_lanes1.png "Windows and Lanes"
[image11]: ./output_images/lanes_on_original1.png "UnWarp Lanes"
[image12]: ./output_images/text_all_lanes1.png "Final Result"
[video1]: ./output2_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in the first code cell of the IPython notebook located in `./Advanced_Lane_Lines.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
Here is a test image that has corners drawn on it:

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

The last thing I did was perform a perspective transform on the image to warp it. Here is the result: 

![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. The original image is shown on the left and the unidistorted image is shown on the right below:
![alt text][image4]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The gradients in the X and Y directions were calculated for each image. For the color thresholds, the V channel from the HSV color space was and the S channel from the HLS color space was used. The thresholding functions are in the seventh code cell in `./Advanced_Lane_Lines.ipynb`.  Here's an example of my output for this step:

![alt text][image5]

The thresholded binary image was created using this code found in the eighth code cell:

```
# Process image and generate binary pixel of interests 
    preprocess_img = np.zeros_like(test_img[:,:,0])
    gradx = abs_sobel_thresh(test_img, orient='x', sobel_kernel=3, thresh=(15,255))
    grady = abs_sobel_thresh(test_img, orient='y', sobel_kernel=3, thresh=(35,255))
    c_binary = color_threshold(test_img, sthresh=(100,255), vthresh=(50,255))
    preprocess_img[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

```
The threshold ranges were determined after many tries using trial and error. These threshold ranges gave me satisfying results. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_perspective()`, which appears in the 11th code cell of the IPython notebook.  The `transform_perspective()` function takes as inputs an image (`result`), as well as source points (`src`) and image size(`img_size`).  I first defined a perspective transform area, in this case a trapezoid using the function `transform_area()` found in the ninth code cell. The function has four tuning parameters that adjust the size of the trapezoid including the width and height. These parameters helped me determine the source points for me to use. I chose the source and destination points in the following manner:

```
src = np.float32([[img_size[0]*(.5-mid_width/2), img_size[1]*height_pct],                                    
                  [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct],                                             
                  [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim], 
                  [img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim]])

dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                 [img_size[0]-offset, img_size[1]], 
                 [offset, img_size[1]]])

```
This resulted in the following source and destination points:

| Source          | Destination   | 
|:--------------: |:-------------:| 
| 579.20, 460.08  | 230.40, 0     | 
| 700.80, 460.08  | 1049.60, 0    |
| 1017.60, 673.20 | 1049.60, 720  |
| 262.40, 673.20  | 230.40, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The following two images describe this process: 

![alt text][image6]
![alt text][image7]

The perspective transform of a thresholded binary image looks like this: 

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The next step after the perspective transformation was to detect lane-line pixels and fit their positions with a polynomial. I ended up implementing the sliding Windows method using 1D convolutions. My code in this section was heavily inspired by Udacity Content Developer Aaron Brown's video tutorial explaining this technique. The video can be found in this link [Video Tutorial](https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be&utm_medium=email&utm_campaign=2017-02-09_carnd_february_digest&utm_source=blueshift&utm_content=2017-02-09_carnd_monthlydigest&bsft_eid=692c214a-da20-4af0-a55a-5b590711f4ef&bsft_clkid=9a84f48a-848d-4b79-9593-7430fc8cb2c4&bsft_uid=229d21be-62d8-4201-aa1d-657de08d3285&bsft_mid=51d569b7-e638-4432-805d-c8912873acae). In the method, vertical pixel values are summed up in the image and can be visually represented by two peaks in a histogram. The two peaks would represent the left and right lane. Using a window with a set size, the centers of these peaks can be found. The window first starts at the bottom of the image and moves up the image to trying to find areas with a high pixel count. As the window is moving up it is also moving left and right trying to identify an area of high pixel count that could be part of a lane. This sliding window technique can be visualized below:

![alt text][image9]

To keep track of the windows for the left and right lanes in each image, I used the class `tracker()` found in the 14th code cell. The class also contains the function `find_window_centroids()` that is used to find the centroids of the windows in order to fit a line corresponding to a lane. 
The class is shown below: 

```
# Class to keep track of windows for left and right lanes in each image
class tracker():
    # Specify all unsigned variables when starting a new instance
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 30):
        # Store all the past (left, right) center set values used for smoothing the output in a list
        self.recent_centers = []
    
        # The window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = Mywindow_width
        
        # The window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # Breaks the image into vertical levels
        self.window_height = Mywindow_height
        
        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = Mymargin
        
        self.ym_per_pix = My_ym # meters per pixel in vertical axis
        
        self.xm_per_pix = My_xm # meters per pixel in horizontal axis
        
        self.smooth_factor = Mysmooth_factor
        
    # The main tracking function for finding and storing lane segment positions
    def find_window_centroids(self, warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin 
        
        window_centroids = [] # Store the (left, right) window centroid positions per level 
        window = np.ones(window_width) # Create window template that will be used for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # Then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2

        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # Convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0] - (level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
             
            # Find the best left centroid by using past left center as reference
            # Use the window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index-offset
            # Find the best right centroid by using past right center as a reference 
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(max(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))
        
        self.recent_centers.append(window_centroids)
        
        # Return averaged values of the line centers, helps to keep the markers from jumping around too much
        return np.mean(self.recent_centers[-self.smooth_factor:], axis=0)    

```

The centroids of the windows are found in order to fit a second order polynomial to them to form a line. The function that fits the left and right lanes is `fit_lanes()` and is found in the 17th code cell. The code that forms this function is shown below: 

```
# Function to fit left and right lanes
def fit_lanes(window_centroids, window_height, window_width): 
    # Points used to find the left and right lanes
    rightx = []
    leftx = []
    
    # Go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        # Add center value found in frame to the list of lane points per left,right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
    
    # fit the lane boundaries to the left, right center positions found 
    yvals = range(200, img_size[1])
    
    res_yvals = np.arange(img_size[1] - (window_height/2), 0, -window_height)
    
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
   
    return right_lane, left_lane, left_fitx, right_fitx, res_yvals, leftx, rightx, yvals

```

I then overlayed the fitted lines on top of the windows. This is shown below:

![alt text][image10]

I also overlayed each line on top of the original image by reversing the perspective transform as shown below:  

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature for both the left and right lanes. The x and y points from the fitted polynomial for the lanes were in pixels and had to be converted to meters. Once the points are converted to meters, a second order polynomial fit can be performed to calculate the radius of curvature. I followed the method discussed in the course lessons and in this [link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The radius of curvature calculation is performed in the 21st code cell by the function `radius()`. Below is the code showing the calculation:

```
# Function to calculate the radius of curvature
def radius(curve_centers, res_yvals, leftx, rightx, yvals):
    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dimension
    
    curve_fit_l = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad_l = ((1 + (2*curve_fit_l[0] * yvals[-1] * ym_per_pix + curve_fit_l[1])**2)**1.5) / np.absolute(2*curve_fit_l[0])
    
    curve_fit_r = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(rightx, np.float32) * xm_per_pix, 2)
    curverad_r = ((1 + (2*curve_fit_r[0] * yvals[-1] * ym_per_pix + curve_fit_r[1])**2)**1.5) / np.absolute(2*curve_fit_r[0])
    
    return curverad_r, curverad_l, xm_per_pix

```
The radius of curvature calculation was done assuming the lane is about 30 meters long and 3.7 meters wide as suggested in the course lessons. 

The position of the car relative to the center was done assuming that the camera is at the center of the car. The average of the x-intercepts from the two fitted polynomials was calculated and then subtracted from the center of the image to get the position from the center. If the value for the position of the car is greater than the center value then the car was to be on the left from the center. Otherwise, it would be on the right from the center. This procedure was performed in the 21st code cell in the function `center_offset()` and is detailed below: 

```
# Function to calculate the offset of the car relative to the center of the lane 
def center_offset(left_fitx, right_fitx, img_size, xm_per_pix):
    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-img_size[0]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    
    return side_pos, center_diff

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In the end, I warped the left(red) and right(blue) fitted lines back onto the original image. I also added an inner lane(green) and overlayed it on top of the original image. I also added text to the image that shows the curvature of radius for both lanes and the offset of the car from the center. I implemented this step in 23rd and 24th code cells. Here is an example of my result on a test image:

![alt text][image12]

---

### Resubmission

For my resubmission I ended up following the reviewer's suggestion of implementing sanity checks to reject unusable results and replace them with a result from prior frames. I had one instance in my previous submission when my the highlighted lane lines are exhibit strong flickering, which made the car be outside of the identified lane. The first step I took to fix this problem was to create a class called `line()` that kept track of the previous set of fitted values for both the left and right lanes. This class is found in the first code cell under the "Resubmission" heading in my IPython notebook. To reject unusable results and replace them with a result from a prior frame I took inspiration from Vivek Yadav's Medium post where he describes how to remove any error from incorrectly detected lanes. It can be found in this [link](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa#.wkctsm3ho). My technique involves finding the error between the current coefficient and the previous coefficient values assigned to the highest degree term of the fitted second degree polynomials used for both the left and right lanes. If this calculated error for both lanes is larger than a threshold value then the current fitted value is discarded and the previous fitted value is used. If the calculated error is below the threshold value then the new fitted value is 5% of the current fitted value and 95% of the last fitted value. By applying this method the flickering instance is now gone. I also changed the displayed decimal digits on the video for the radius of curvature and offset of the car to 2 digits to help with the visual appearance. 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://drive.google.com/file/d/0B-vUaJ15H_sHZ0FSbFZhWmhOR28/view?usp=sharing)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I ended spending a lot of time getting reasonable thresholded binary images. The approach I took was using trial and error to try different combinations of thresholding and different values for the minimum and maximum thresholded values. For this project, it was important to have good thresholded binary images that isolated the pixels representing the lane lines effectively. The sliding windows step depended on the quality of the thresholded binary images. Having good thresholded binary images allowed the windows to follow the lanes better. I also spent a lot of time tuning the parameters for the window sliding technique so that that the windows would follow the lanes better. My pipeline will likely fail if there are different shadows/lighting in the images and if there are objects really close to the lane markings. Using a different image size or a different camera position would also make my pipeline fail. To make my pipeline more robust I could use a better color filtering technique to highlight the yellow and white lane markers. 


```python

```
