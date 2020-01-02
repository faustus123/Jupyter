#!/usr/bin/env python
# coding: utf-8

# # JLab ML Challenge 4 Test 1
# 
# This is a pared down copy of the original notebook. This is used to figure out how to feed multiple images into predict in one call. The hope it to better utilize the GPU by sending a batch of images.

#
# This takes about 12 hours to run. The images 
# created at the end are scaled such that 0-255
# corresponds to the ranges below for viewability.
# These numbers are derived from the data itself
# but not stored anywhere so I copy them here.
#
# delta_bad: min=-2.38419e-08  max=2.38419e-08
# delta_good: min=-5.39949e-09  max=6.96018e-09
# delta_nodata: min=-2.82967e-11  max=2.46473e-11
#
#
# In[ ]:


# Suppress a bunch of warnings when loading keras 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib
import time
import PIL

from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt

MODEL_FILE = '/home/davidl/work2/2019.12.17.MLChallenge4/MLChallenge3/CDC_occupancy-1569856232_566866.h5'
TARGET_FILE='/home/davidl/work2/2019.12.17.MLChallenge4/MLChallenge3/TARGET_IMG.png'

# Image dimensions model expects. Target image will be upsampled to this
img_width  = 800
img_height = 600

# Load model
print("Loading Model: " + MODEL_FILE)
model=load_model(MODEL_FILE)
print("Model Loaded")


# Open original image
img_orig = PIL.Image.open(TARGET_FILE)

# Define procedure for getting model prediction when a single
# pixel is modified
def GetModPred( ix=0, iy=0, mod=0):
    global img_orig
    
    (R, G, B) = img_orig.getpixel((ix, iy))
    RR = max(min(R+mod, 255), 0)
    GG = max(min(G+mod, 255), 0)
    BB = max(min(B+mod, 255), 0)
    rr = max(min(R-mod, 255), 0)
    gg = max(min(G-mod, 255), 0)
    bb = max(min(B-mod, 255), 0)

    # Red + mod
    img_orig.putpixel((ix, iy), (RR,G,B) )
    img_dup_red = img_orig.resize( (img_width, img_height) )
    a_redp = np.array(img_dup_red)/255.0
    a_redp = np.expand_dims(a_redp, axis=0)

    # Green + mod
    img_orig.putpixel((ix, iy), (R,GG,B) )
    img_dup_green = img_orig.resize( (img_width, img_height) )
    a_greenp = np.array(img_dup_green)/255.0
    a_greenp = np.expand_dims(a_greenp, axis=0)

    # Blue + mod
    img_orig.putpixel((ix, iy), (R,G,BB) )
    img_dup_blue = img_orig.resize( (img_width, img_height) )
    a_bluep = np.array(img_dup_blue)/255.0
    a_bluep = np.expand_dims(a_bluep, axis=0)

    # Red - mod
    img_orig.putpixel((ix, iy), (rr,G,B) )
    img_dup_red = img_orig.resize( (img_width, img_height) )
    a_redm = np.array(img_dup_red)/255.0
    a_redm = np.expand_dims(a_redm, axis=0)

    # Green - mod
    img_orig.putpixel((ix, iy), (R,gg,B) )
    img_dup_green = img_orig.resize( (img_width, img_height) )
    a_greenm = np.array(img_dup_green)/255.0
    a_greenm = np.expand_dims(a_greenm, axis=0)

    # Blue - mod
    img_orig.putpixel((ix, iy), (R,G,bb) )
    img_dup_blue = img_orig.resize( (img_width, img_height) )
    a_bluem = np.array(img_dup_blue)/255.0
    a_bluem = np.expand_dims(a_bluem, axis=0)

    # Get model predictions
    images = np.vstack([a_redp, a_greenp, a_bluep, a_redm, a_greenm, a_bluem])
    pred = model.predict(images)

    # Restore original color
    img_orig.putpixel((ix, iy), (R,G,B) )

    # Calculate derivative
    dpredR = (pred[0] - pred[3])/(RR - rr)
    dpredG = (pred[1] - pred[4])/(GG - gg)
    dpredB = (pred[2] - pred[5])/(BB - bb)
    dpred = np.array([dpredR, dpredG, dpredB])
	 
    return dpred

# Create arrays to hold the results of the single pixel tests
# so we can create images of them later to visualize the
# dependence.
# First index is bad,good,nodata. Last index is R,G,B
delta_bad    = np.zeros((3, img_orig.height, img_orig.width, 3))
delta_good   = np.zeros((3, img_orig.height, img_orig.width, 3))
delta_nodata = np.zeros((3, img_orig.height, img_orig.width, 3))

# Infer Model using original image
#pred_orig = GetModPred()[0]
#print( pred_orig )

# Loop over pixels, modifying each color component and checking how inference differs
print('Starting diff calculation ...')
tstart = time.time()
for iy in range(0, img_orig.height):
    tstart_row = time.time()
    for ix in range(0, img_orig.width):
        dpred = GetModPred( ix, iy, 5)
        
        # Red
        delta_bad[0][iy][ix][0]    = dpred[0,0]
        delta_good[0][iy][ix][0]   = dpred[0,1]
        delta_nodata[0][iy][ix][0] = dpred[0,2]
        
        # Green
        delta_bad[1][iy][ix][1]    = dpred[1,0]
        delta_good[1][iy][ix][1]   = dpred[1,1]
        delta_nodata[1][iy][ix][1] = dpred[1,2]
        
        # Blue
        delta_bad[2][iy][ix][2]    = dpred[2,0]
        delta_good[2][iy][ix][2]   = dpred[2,1]
        delta_nodata[2][iy][ix][2] = dpred[2,2]
        #break

    # Calculate time to process row and estimate remaining time
    now = time.time()
    tdiff_row = now - tstart_row
    tdiff_per_row = (now - tstart)/float(iy+1)
    if img_orig.height > (iy+1):
        tleft =  (img_orig.height - iy -1)*tdiff_per_row
        print('Time per row: %fs  -- eta: %fmin' % (tdiff_row, tleft/60.0))
    #break

# Normalize so we have RGB values in the 0-255 range
# Note that this loses track of where the "zero diff"
# color is.

print('delta_bad: min=%g  max=%g' % (delta_bad.min(), delta_bad.max()) )
print('delta_good: min=%g  max=%g' % (delta_good.min(), delta_good.max()) )
print('delta_nodata: min=%g  max=%g' % (delta_nodata.min(), delta_nodata.max()) )

delta_bad    = delta_bad    + delta_bad.min()
delta_good   = delta_good   + delta_good.min()
delta_nodata = delta_nodata + delta_nodata.min()

delta_bad    = delta_bad*255.0/delta_bad.max()
delta_good   = delta_good*255.0/delta_good.max()
delta_nodata = delta_nodata*255.0/delta_nodata.max()

# Make images of differences and save them
labels = ['bad','good','nodata']
colors = ['R','G','B']
for i in range(0,3): # Loop over colors
    dbad = delta_bad[i,:,:,:]
    dgood = delta_good[i,:,:,:]
    dnodata = delta_nodata[i,:,:,:]

    img_diff_bad = PIL.Image.fromarray( dbad.astype(np.uint8) )
    img_diff_good = PIL.Image.fromarray( dgood.astype(np.uint8) )
    img_diff_nodata = PIL.Image.fromarray( dnodata.astype(np.uint8) )
    img_diff_bad.save('diff_bad%s.png' % colors[i])
    img_diff_good.save('diff_good%s.png' % colors[i])
    img_diff_nodata.save('diff_nodata%s.png' % colors[i])

