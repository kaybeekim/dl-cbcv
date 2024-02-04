print('hyperparam.py is loaded.')

import pandas as pd 
import numpy as np
import torch
import math
from datetime import date, datetime, timedelta  # for working with dates and times


# ### TO DO:
# if problem == 'STL' and target == 'acq':
#     pass
# elif problem == 'STL' and target == 'rptorder_p_c':
#     pass
# elif problem == 'STL' and target == 'aov':
#     pass
# elif problem == 'STL' and target == 'sales':
#     pass
# elif problem == 'MTL' and embedding == True:
#     pass
# elif problem == 'MTL' and embedding == False:
#     pass
# else:
#     raise ValueError('Invalid problem and target combination')



### Hyperparameters for ROPC AOV in STL ======================================================

### train:test data period
TRAIN_START='2017-01-01'  # when train period starts
TEST_START='2019-04-01' # when test period starts
TEST_END='2020-02-29' # when test period ends
VAL_LOSS=True # whether to use validation loss or not
VAL_START='2019-01-01' # when validation period starts

### optimization hyperparameters
N_EPOCHS=100                     # number of epochs
LEARNING_RATE=5e-4              # learning rate
BATCH_SIZE=64                   # batch size
WEIGHT_DECAY=1e-7                # weight decay (overfitting large: 1e=3~4, if data set is large and reduce overfitting: 1e-6~8)
PATIENCE=5                      # patience for early stopping
MINDELTA=1e-5                   # min delta for early stopping
GRADCLIP=1                      # gradient clipping
REDUCE_LR=False                 # whether to reduce learning rate or not
GPUS = '0'                      # which gpu to use (if multiple GPUs available)
NUM_WORKERS = 0                 # number of workers for data loading

### Transformer model hyperparameters
D_MODEL=32                      # model dimension
N_HEAD=4                        # number of heads
D_FEEDFORWARD=64               # feedforward dimension in Transformer
D_FEEDFORWARD_TASK=64          # feedforward dimension in Transformer for task-specific layers
N_ENCODER_LAYERS=2              # number of encoder layers in Transformer
N_DECODER_LAYERS=2              # number of decoder layers in Transformer
ACTIVATION='relu'               # activation function
DROPOUT=0.0                     # dropout rate

### encoder decoder validation length
INPUT_CHUNK_LENGTH=10           # input sequence length
OUTPUT_CHUNK_LENGTH=1           # output sequence length (single horizon prediction)

### compute dates for train:validation:test data period
TRAIN_START_with_offset=(pd.to_datetime(TRAIN_START) - timedelta(days=7*INPUT_CHUNK_LENGTH)).strftime('%Y-%m-%d')
VAL_START_with_offset=(pd.to_datetime(VAL_START) - timedelta(days=7*INPUT_CHUNK_LENGTH)).strftime('%Y-%m-%d')
TEST_START_with_offset=(pd.to_datetime(TEST_START) - timedelta(days=7*INPUT_CHUNK_LENGTH)).strftime('%Y-%m-%d')

TRAIN_END=(pd.to_datetime(VAL_START) - timedelta(days=1)).strftime('%Y-%m-%d')
VAL_END=(pd.to_datetime(TEST_START) - timedelta(days=1)).strftime('%Y-%m-%d')
TEST_END_EXTEND=(pd.to_datetime(TEST_END) + timedelta(days=91)).strftime('%Y-%m-%d') # + additional one quarter (for computational issue)

### embedding dimension
MERCHANT_EMB_DIM = 32
WEEK_EMB_DIM = round(math.sqrt(53))
COHORT_MONTH_EMB_DIM = round(math.sqrt(12))
COHORT_EMB_NUM = 100
GROUP_CENSORED_EMB_DIM = round(math.sqrt(COHORT_EMB_NUM))


### MTL weights
w_acq = 4
w_ropc = 4
w_aov = 1
w_spend = 1

### Hyperparameters for Acq, Sales in STL ======================================================


# ### train:test data period
# TRAIN_START='2017-01-01'  # when train period starts
# TEST_START='2019-04-01' # when test period starts
# TEST_END='2020-02-29' # when test period ends
# VAL_LOSS=True # whether to use validation loss or not
# VAL_START='2019-01-01' # when validation period starts

# ### optimization hyperparameters
# N_EPOCHS=90                     # number of epochs
# LEARNING_RATE=1e-3              # learning rate
# BATCH_SIZE=32                   # batch size
# WEIGHT_DECAY=0.0                # weight decay
# PATIENCE=5                      # patience for early stopping
# MINDELTA=1e-3                   # min delta for early stopping
# GRADCLIP=1                      # gradient clipping
# REDUCE_LR=False                 # whether to reduce learning rate or not
# GPUS = '0'                      # which gpu to use (if multiple GPUs available)
# NUM_WORKERS = 0                 # number of workers for data loading

# ### Transformer model hyperparameters
# D_MODEL=32                      # model dimension
# N_HEAD=8                        # number of heads
# D_FEEDFORWARD=512               # feedforward dimension in Transformer
# N_ENCODER_LAYERS=3              # number of encoder layers in Transformer
# N_DECODER_LAYERS=3              # number of decoder layers in Transformer
# ACTIVATION='relu'               # activation function
# DROPOUT=0.0                     # dropout rate

# ### encoder decoder validation length
# INPUT_CHUNK_LENGTH=10           # input sequence length
# OUTPUT_CHUNK_LENGTH=1           # output sequence length (single horizon prediction)





## hyperparameter tuning: MTLwith3tasks one company over shopify, grubhub, planet fitness
LEARNING_RATE_MTL_ONE1=4.7e-05
D_MODEL_MTL_ONE1=32                     
D_FEEDFORWARD_MTL_ONE1=32             
D_FEEDFORWARD_TASK_MTL_ONE1=64          
N_ENCODER_LAYERS_MTL_ONE1=1            
N_DECODER_LAYERS_MTL_ONE1=1            
DROPOUT_MTL_ONE1=0.01                    

LEARNING_RATE_MTL_ONE2=0.002
D_MODEL_MTL_ONE2=16                     
D_FEEDFORWARD_MTL_ONE2=1024             
D_FEEDFORWARD_TASK_MTL_ONE2=64          
N_ENCODER_LAYERS_MTL_ONE2=1            
N_DECODER_LAYERS_MTL_ONE2=3            
DROPOUT_MTL_ONE2=0.14 

## hyperparameter tuning: MTLwith4tasks one company over itunes, pizza hut, home depot, target4 sales objective only
LEARNING_RATE_MTL4_ONE=0.0001
D_MODEL_MTL4_ONE=12
D_FEEDFORWARD_MTL4_ONE=32
D_FEEDFORWARD_TASK_MTL4_ONE=512
N_ENCODER_LAYERS_MTL4_ONE=2
N_DECODER_LAYERS_MTL4_ONE=2
DROPOUT_MTL4_ONE=0.055



LEARNING_RATE_MTL_EMB1=0.004
D_MODEL_MTL_EMB1=16                     
D_FEEDFORWARD_MTL_EMB1=64             
D_FEEDFORWARD_TASK_MTL_EMB1=32          
N_ENCODER_LAYERS_MTL_EMB1=3           
N_DECODER_LAYERS_MTL_EMB1=1            
DROPOUT_MTL_EMB1=0.1

LEARNING_RATE_MTL_EMB2=0.0003
D_MODEL_MTL_EMB2=32
D_FEEDFORWARD_MTL_EMB2=64
D_FEEDFORWARD_TASK_MTL_EMB2=64
N_ENCODER_LAYERS_MTL_EMB2=3
N_DECODER_LAYERS_MTL_EMB2=1
DROPOUT_MTL_EMB2=0.35



LEARNING_RATE_STL_ROPC_EMB1=0.0004
D_MODEL_STL_ROPC_EMB1=16
D_FEEDFORWARD_STL_ROPC_EMB1=256
N_ENCODER_LAYERS_STL_ROPC_EMB1=1
N_DECODER_LAYERS_STL_ROPC_EMB1=2
DROPOUT_STL_ROPC_EMB1=0.02






