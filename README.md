# CLEAR-code
This is the official Python implementation of our paper "CLEAR: A Novel Gap-filling Method for Optical Remote Sensing Images Combining Class-based Linear Regression and Iterative Residual Compensation"

CLEAR is a simple, accurate, and efficient remote sensing image gap-filling method that utilizes cloud-free or low-cloud-cover images from the time series to fill the missing values (e.g., clouds and cloud shadows) in the target image.

CLEAR does not require training and is highly flexible, allowing for rapid deployment in your own applications.

The figure below shows the comparison results of CLEAR and other methods on a simulated cloudy image in an agricutural area. One can observe from the figure that image processed by CLEAR has the highest similarity to the real image and exhibits the lowest error.

![本地图片](Omaha-low-4.png)

## Data
Data used in our paper: https://drive.google.com/drive/folders/1e5gDRZQacl8i6lITvRRUugTwQ1OwF60Y?usp=drive_link  
Each JPEG 2000 file contains stacked time-series images with a data type of uint16 and a value range of 0-10000. Dividing by 10000 yields the surface reflectance.

**If you are using your own data, please modify line 216 in main_CLEAR_fill_simulated.py and line 276 in main_CLEAR_fill_time_series.py to match your data type.**  

## Contact
**If you have any questions about our code or would like to use CLEAR in your own research, please feel free to contact me via email.**  
Author: Houcai GUO, PhD student at the University of Trento, Italy  
E-mail: houcai.guo@unitn.it; guohoucai@qq.com
