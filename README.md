# CLEAR-code
This is the official Python implementation of our paper "CLEAR: A Novel Gap-filling Method for Optical Remote Sensing Images Combining Class-based Linear Regression and Iterative Residual Compensation"

## Data
Data used in our paper: https://drive.google.com/drive/folders/1e5gDRZQacl8i6lITvRRUugTwQ1OwF60Y?usp=drive_link  
Each JPEG 2000 file contains stacked time-series images with a data type of uint16 and a value range of 0-10000. Dividing by 10000 yields the surface reflectance.

**If you are using your own data, please modify line 216 in main_CLEAR_fill_simulated.py and line 276 in main_CLEAR_fill_time_series.py to match your data type.**  

## Contact
**If you have any questions about our code or would like to use CLEAR in your own research, please feel free to contact me via email.**  
Author: Houcai GUO, PhD student at the University of Trento, Italy  
E-mail: houcai.guo@unitn.it; guohoucai@qq.com
