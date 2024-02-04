print('config.py is loaded.')

import os
import pandas as pd 
import numpy as np
import torch

def get_environ_info(device, random_seed=0, verbose=True):
    if verbose:
        print('\npandas version:', pd.__version__)
        print('pytorch version:', torch.__version__)

    print('\nDevice:', device)
    if device.type == 'cuda': # if gpu is available
        print('CUDA version:', torch.version.cuda)
        print('GPU spec:', torch.cuda.get_device_name(0))
        print('\nMemory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # Check if more than one GPU is available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs are available.")

    ## ensure reproducibility of random number generation
    random_seed = random_seed
    torch.manual_seed(random_seed) # in torch
    torch.cuda.manual_seed_all(random_seed) # across all gpu devices
    np.random.seed(random_seed) # in numpy
    print(f"Random seed: {random_seed} for torch and numpy.")

    print('\nCurrent working directory:', os.getcwd())



def create_save_folders(save_mode, save_folder):
    if save_mode:
        save_model = f'{save_folder}/model'
        save_epoch = f'{save_folder}/epoch'
        save_plot = f'{save_folder}/plot'
        save_predict = f'{save_folder}/outcome'
        save_actual = f'{save_folder}/actual'
        for folder in [save_folder, save_model, save_epoch, save_plot, save_predict, save_actual]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        return save_model, save_epoch, save_plot, save_predict, save_actual
    else:
        return None, None, None, None, None



### one company mode ==================
## select one company from below 10 companies
TOY_MERCHANT_NAME_LIST = ['102_shopify', '104_planet_fitness', '113_grubhub', '122_autozone', '129_geico', '169_robinhood', '172_goodwill', '177_digit_co', '193_instacart', '381_wayfair']



def get_target_variable_name(prediction_goal):
    if prediction_goal == 'repeat_order_per_customer' or prediction_goal == 'ropc':
        return 'rpt_orders_per_cust' # name of variable in dataset
    elif prediction_goal == 'average_order_value' or prediction_goal == 'aov':
        return 'aov' # name of variable in dataset
    elif prediction_goal == 'acquisition' or prediction_goal == 'acq':
        return 'cohort_size' # name of variable in dataset
    elif prediction_goal == 'sales' or prediction_goal == 'spend':
        return 'spend'
    elif prediction_goal == 'mtl_4tasks':
        return ['cohort_size_pseudo', 'rpt_orders_per_cust', 'aov', 'rpt_spend_pseudo']
    elif prediction_goal == 'mtl_3tasks':
        return ['cohort_size_pseudo', 'rpt_orders_per_cust', 'aov']
    else:
        raise ValueError('PREDICTION_GOAL is not valid. See libs.hyperparam.py for valid options')



def get_covariate_variable_name(df, USE_EMBEDDING=False, COHORT_EMBEDDING=False, DUMMY_VAR=False):
    COVARIATE_FEATURE_NAMES = ['holidays_1w_ahead', 
                               'linear_trend', 'quadratic_trend', 
                               'tenure_scaled', 'quad_tenure_scaled']
    if DUMMY_VAR:
        COVARIATE_FEATURE_NAMES += ['week_' + str(i+1) for i in range(53)] 
    else:
        COVARIATE_FEATURE_NAMES += ['week_int']

    if COHORT_EMBEDDING and DUMMY_VAR:
        COVARIATE_FEATURE_NAMES += ['cohort_m' + str(i+1) for i in range(12)] 
        COVARIATE_FEATURE_NAMES += [col for col in df.columns if 'cohort_' in col]
    elif COHORT_EMBEDDING and not DUMMY_VAR:
        COVARIATE_FEATURE_NAMES = COVARIATE_FEATURE_NAMES + ['cohort_month_int', 'group_censored_int']
    else:
        COVARIATE_FEATURE_NAMES = COVARIATE_FEATURE_NAMES + ['group_int_scaled', 'quad_group_int_scaled']

    if USE_EMBEDDING:
        COVARIATE_FEATURE_NAMES = COVARIATE_FEATURE_NAMES + ['merchant_emb_int']
        
    assert all([col in df.columns for col in COVARIATE_FEATURE_NAMES]), \
        'COVARIATE_FEATURE_NAMES is not valid.'

    covariate_name_to_index = {name: index for index, name in enumerate(COVARIATE_FEATURE_NAMES)}
    
    return COVARIATE_FEATURE_NAMES, covariate_name_to_index
