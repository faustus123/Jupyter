#!/usr/bin/env python
# coding: utf-8

# # JLab ML Challenge 2
# 
# This notebook is an attempt at JLab ML Challenge 2. This was actually done some time after the challenge ended. My goal though is to take what was learned from the winning entries and implement it so as to better understand RNNs. This is the first step in applying this technology to actual data. 
# 
# In this section I recreate Andru Quiroga's model from his winning entry, but modify it a bit. Andru used an RNN, but fixed the number of time steps to be 7. In other words, he always used the last 7 detectors hit and ignored any other detectors when making his prediction. For the model defined below, I leave the number of time steps as undefined so that any number of them can be provided to the model and all will be used to predict the next step. This is done by specifying the input_shape of the first LSTM layer to have a shape of *(None,7)* instead of *(7,7)*. Note that the second *7* in the input_shape is for the input features. In this case, the 6 parameter state vector (*x,y,z,px,py,pz*) plus the z coordinate of the next detector plane that we are projecting to.
# 
# The other change relative to Andru's model is that the last layer uses *TimeDistributed* instead of a single Dense layer. What this does is output a separate *Dense* layer for each time step. This allows us to obtain the predicted state vector for every time step (i.e. detector plane). This is very useful in training since it allows every timestep to contribute to the training. This point is worth expanding on a little. What this means is that I can train on only 24 hit tracks, but the weights will be adjusted to try and fit every plane, not just the last one. The final model can then be fed any number of detector planes (e.g. 14) and it will give a prediction for the 15th plane.
# 
# In order to support the *TimeDistributed* layer, the **return_sequences** parameter of the last LSTM layer needed to be set to true. In Andru's model this was set to false and only the last time step passed to the final *Dense* layer.

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import gzip
import numpy as np
import math

import tensorflow as tf
import tensorflow.keras as keras

from keras.models import Model, Sequential
from keras.layers import Dense, TimeDistributed, Input, Lambda, LSTM

model = Sequential()
model.add( LSTM(units=500, return_sequences=True, input_shape=(None,7)) )  #  None -> undfined number of time steps
model.add( LSTM(units=500, return_sequences=True) )
model.add( LSTM(units=500, return_sequences=True) )
model.add( TimeDistributed(Dense(units=5)) )

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()


# ## Read in training data
# 
# In this section I read in the training data. The training file is in csv format with 150 values per line. These correspond to the 25 state vectors, each with 6 parameters. The first state vector is at the target position while all others are at detector planes. For this challenge, the target position is considered a detector plane itself. For detector plane (except the last) a 7-parameter feature list is created using the 6 parameter state vector plus the z corrdinate of the next detector plane. For the labels, we use the 5 parameters of the state vector (z coordinate is excluded) at the next plane. Thus, there are 24 *time steps* with 24 *labels* for each row of the input file.
# 
# Note that not every sample in the input file is used here since I was having problems with the server jupyter process dying and suspected it was due to memory issues.

# In[2]:


import csv

print('Reading input data ...')

with open('MLchallenge2_training.csv') as csv_file:
    csv_file.readline() # discard header line
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    x_all = []
    y_all = []
    for row in csv_reader:
        
        # Copy each state vector (6 values) into individual list with a 7th value being
        # the z of the next detector plane. At the same time, create 5 parameter label.
        features = []
        labels = []
        for i in range(0, 24):
            idx = i*6
            features.append(row[idx:idx+6]+[row[idx+6+2]])
            idx += 6
            labels.append(row[idx:idx+2]+row[idx+3:idx+6])
        x_all.append(features)
        y_all.append(labels)
        
        # Limit how many samples we use
        if len(y_all) >=50000: break

TRAIN_FRACTION = 0.90
idx = int(len(x_all)*TRAIN_FRACTION)
x_train = np.array(x_all[0:idx])
y_train = np.array(y_all[0:idx])
x_test  = np.array(x_all[idx:])
y_test  = np.array(y_all[idx:])

# Not sure if this allows memory to be freed, but maybe ...
x_all = []
y_all = []

print('Training samples: ' + str(len(x_train)) + ' (' + str(len(y_train)) +')')
print(' Testing samples: ' + str(len(x_test )) + ' (' + str(len(y_test )) +')')


# ## Data generator
# 
# In this section I set up the data generator. This is really not the right way to do this. I should either have combined reading the file with the generator so that I didn't have to limit the samples in the previous section, *OR* I should not use a generator at all and just pass the data into Keras as arrays. This is basically an artifact from my first attempt at this when the generator was much more complicated.

# In[3]:


BATCH_SIZE = 10

def my_generator(x_samples, y_samples):
    global BATCH_SIZE

    BATCH_NSTEPS = 7
    BATCH_INDEX  = 0
    NSAMPLES = len(x_samples)

    print('my_generator: ' + str(len(x_samples)) + ' samples')
    
    while True:
        x = []
        y = []
        for i in range(0, BATCH_SIZE):
            x.append( x_samples[BATCH_INDEX] )
            y.append( y_samples[BATCH_INDEX] )
            BATCH_INDEX = (BATCH_INDEX+1)%NSAMPLES
        
        x_np = np.array(x)
        y_np = np.array(y)
        yield x_np, y_np

train_generator = my_generator(x_train, y_train)
valid_generator = my_generator(x_test, y_test)


# ## Fit the model
# 
# ***BE CAREFUL RUNNING THIS CELL!***
# 
# In this section we fit the model. Using Jupyter to do this is actually really slow due to the hardware allocated to the notebook. Running this on one of the sciml190X machines with multiple Titan RX GPUs sped this up by a factor of about 10-20. Because I don't want this to accidentally overwrite the model file, I only have it fit and save the model if *FIT_MODEL* is set to True. I leave it set to False so I can run all of the cells here without overwriting the model file.
# 
# To run this on one of the sciml190X machines, I exported this notebook as an executable script. Here are the details for reproduceability. Pleae note that these are based on magic that Will Phelps figured out and shared with me:
# 
# 1. **File->Export Notebook As...->Export Notebook To Executable Script**<br>
# Transfer the produced script (*2019.12.31.MLChallenge2.py*) to the working directory on the CUE. In my case: */home/davidl/Jupyter/2019.12.31.MLChallenge2*
# 
# 2. **Setup conda environment. Only do this once. Skip to section 3 if you've already done this:**<br>
# 2a. ssh ifarm1802<br>
# 2b. ln -s /work/halld2/home/davidl/builds/CONDA_ENV ~/.conda<br>
# 2c. bash<br>
# 2d. source /etc/profile.d/modules.sh<br>
# 2e. module use /apps/modulefiles<br>
# 2f. module load anaconda2/4.5.12<br>
# 2g. conda create -n tf-gpu tensorflow-gpu=1.14 cudatoolkit=10.0 keras numpy=1.16.4<br>
# 2h. conda activate tf-gpu<br>
# 2i. pip install Pillow<br>
# 2j. conda install ipython matplotlib<br>
# 
# 3. **Allocate interactive session on one of the scml190X nodes**<br>
# 3a. salloc --gres gpu:TitanRTX:2 --partition gpu --nodes 1<br>
# 3b. srun --pty bash<br>
# 3c. source /etc/profile.d/modules.sh<br>
# 3d. module use /apps/modulefiles<br>
# 3e. module load anaconda2/4.5.12<br>
# 3f. conda activate tf-gpu<br>
# 
# 4. **Go to working directory and run script**<br>
# 4a. cd /home/davidl/Jupyter/2019.12.31.MLChallenge2<br>
# 4b. ipython3 2019.12.31.MLChallenge2.py<br>

# In[4]:



FIT_MODEL = False
EPOCHS = 20
STEP_SIZE_TRAIN = len(x_train)/BATCH_SIZE
STEP_SIZE_VALID = len(x_test)/BATCH_SIZE
model_fname = 'MyModel.h5'

if FIT_MODEL:
    history = model.fit_generator(
      generator            = train_generator
      ,steps_per_epoch     = STEP_SIZE_TRAIN
      ,validation_data     = valid_generator
      ,validation_steps    = STEP_SIZE_VALID
      ,epochs              = EPOCHS
      ,use_multiprocessing = False
    )

    model.save(model_fname)
    print('Model saved to: ' + model_fname)


# ## Check the model using the Test data set
# 
# In this section I load the model and feed it some tracks from the Test data set. In the Test set, each row of the inputs file will have a diffrent number of entries depending on how many detector planes are present. This will format the inputs into the 7 parameter features the model expects with as many time steps as detector planes.
# 
# I then loop through the outputs file which contains 5 parameters per row corresponding to the expected output from the model.

# In[7]:


import csv
import time
import numpy as np
from tensorflow.keras.models import load_model

print('Loading model: ' + model_fname)
model = load_model(model_fname)
model.summary()

inputs_fname = 'MLchallenge2_testing_inputs.csv'
outputs_fname = 'MLchallenge2_testing_outputs.csv'

# Read in all inputs
with open(inputs_fname) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    x = []
    for row in csv_reader:
        
        # Copy each state vector (6 values) into individual list with a 7th value being
        # the z of the next detector plane. At the same time, create 5 parameter label.
        features = []
        Ndets = int((len(row)-1)/6)
        for i in range(0, Ndets):
            idx = i*6
            idx_next_z = idx+6+2
            if idx_next_z >= len(row): idx_next_z = len(row)-1
            features.append(row[idx:idx+6]+[row[idx_next_z]])
        x.append(features)

        # Limit how many samples we use
        if len(x) >=5000: break

# Read in all output targets
with open(outputs_fname) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    y = []
    for row in csv_reader:
        y.append(row)

        # Limit how many samples we use
        if len(y) >=len(x): break

tstart = time.time()
for i in range(0, len(x)):
    model.reset_states()
    pred = model.predict([[x[i]]])

tend = time.time()
tdiff = tend - tstart

print(' Number tracks: %d' % len(x))
print('    Total time: %3.3f sec' % tdiff)
print('Time per track: %3.3f msec' % (tdiff*1000.0/len(x)))


# In[ ]:




