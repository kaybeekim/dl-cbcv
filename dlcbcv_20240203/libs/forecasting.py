print('forecasting.py is loaded.')

import numpy as np
import pandas as pd
import torch
from libs.customized_dataset import CrossSectionalTimeSeriesDataset



def prepare_TimeSeriesDataset(main_value_test_np, main_cov_test, TimeSeriesDataset, INPUT_CHUNK_LENGTH):
    test_datasets = []
    
    # over cohorts
    for i in range(len(main_value_test_np)):        
        # pick cohort i and output dictionary of target, covariate, gt samples
        test_dataset = TimeSeriesDataset(main_value_test_np[i:(i+1)],
                                         main_cov_test[i:(i+1)], INPUT_CHUNK_LENGTH)
        test_datasets.append(test_dataset) 
        
    return test_datasets



def predict_each_cohort(model, device, test_dataset, current_seq):    
    # initialize empty list to store predictions for each cohort
    pred_seq_acq = []
    pred_seq_rptorder_p_c = []
    pred_seq_aov = []
    
    # over samples in the cohort
    for j in range(len(test_dataset)):        
        # concatenate current sequence (from j-1th sample) with covariate
        test_input = torch.cat((current_seq, test_dataset[j]['covariate'].to(device)), dim=-1) # all gpu
        
        # reshape from (input_chunk_length, input_dim) to (1, input_chunk_length, input_dim)
        test_input = test_input.unsqueeze(0).float().to(device) # to gpu
        
        # predict next value
        acq_hat, rptorder_p_c_hat, aov_hat = model(test_input) # gpu

        # detach tensors (still on device) and append detached tensors to lists (if they are tensors)
        pred_seq_acq.append(acq_hat.detach().cpu().numpy()) # all cpu
        pred_seq_rptorder_p_c.append(rptorder_p_c_hat.detach().cpu().numpy())
        pred_seq_aov.append(aov_hat.detach().cpu().numpy())

        # concatenate and convert to tensor (all on device)
        # Assuming acq_hat_detached, rptorder_p_c_hat_detached, and aov_hat_detached are 1D tensors with a single element
        new_entry = torch.tensor([[acq_hat.detach().item(), rptorder_p_c_hat.detach().item(), aov_hat.detach().item()]], device=device)
        current_seq = torch.cat((current_seq[1:], new_entry), dim=0) # all gpu

    return pred_seq_acq, pred_seq_rptorder_p_c, pred_seq_aov      




def rolling_forecast(test_datasets, model, device, TASK_FEATURE_NAMES, verbose=False):
    # NOTE: For rolling forecast, we only need 
    # first sample of each cohort (test_dataset[0]['target']), and 
    # all covariates (test_dataset[j]['covariate'])
    
    # initialize empty list to store predictions for entire cohorts
    main_value_test_pred_1 = []
    main_value_test_pred_2 = []
    main_value_test_pred_3 = []
    
    i=0
    total = len(test_datasets)
    for test_dataset in test_datasets:
        # initialize current sequence with the first input_chunk_length values of the cohort
        current_seq = test_dataset[0]['target'][:,:3].to(device) # regardless of sales binding or not, first 3 tasks are used
        
        pred_1, pred_2, pred_3 = predict_each_cohort(model, device, test_dataset, current_seq)

        main_value_test_pred_1.append(np.ravel(pred_1))
        main_value_test_pred_2.append(np.ravel(pred_2))
        main_value_test_pred_3.append(np.ravel(pred_3))
        
        i+=1
        if verbose and i%100==0:
            print(f'{i}/{total} cohorts are predicted.')

    main_value_test_pred = {
        TASK_FEATURE_NAMES[0]: main_value_test_pred_1,
        TASK_FEATURE_NAMES[1]: main_value_test_pred_2,
        TASK_FEATURE_NAMES[2]: main_value_test_pred_3
    }
    
    return main_value_test_pred




## TO DO: refactor this:

def get_sales_recovered_data(predicted_main_acq_raw, actual_main_raw, predicted_main_raw, TEST_START,
                             USE_EMBEDDING=False, MERCHANT_NAMES_EMB_INT=None):
    
    actual_main_lst = []
    predicted_main_lst = []
    
    if USE_EMBEDDING:
        assert MERCHANT_NAMES_EMB_INT is not None, 'MERCHANT_NAMES_EMB_INT is not provided.'
        merchant_name_lst = list(MERCHANT_NAMES_EMB_INT.keys())
        for merchant_name in merchant_name_lst:
            predicted_main_acq0 = predicted_main_acq_raw[predicted_main_acq_raw['merchant_name']==merchant_name]
            actual_main0 = actual_main_raw[actual_main_raw['merchant_name']==merchant_name]
            predicted_main0 = predicted_main_raw[predicted_main_raw['merchant_name']==merchant_name]

            ## to avoid cohort_size key error in repeated merging
            if 'cohort_size' in predicted_main0.columns: 
                predicted_main0.drop(columns=['cohort_size'], inplace=True)
                    
            predicted_main_acq=predicted_main_acq0.copy()
            actual_main=actual_main0.copy()
            predicted_main=predicted_main0.copy()
            
            ## prepare acquisition data
            acq_predicted_main = predicted_main_acq
            acq_predicted_main.columns = ['merchant_name', 'group','cohort_size']
            acq_predicted_main['group'] = acq_predicted_main['group'].astype(str) # change datetime64[ns] to object

            acq_actual_main = actual_main[['group','cohort_size']].drop_duplicates().reset_index(drop=True)
            acq_actual_main = acq_actual_main[acq_actual_main['group'] < TEST_START]

            # combine acq_predicted_main and acq_actual_main by rows
            acq_predicted_main = pd.concat([acq_actual_main, acq_predicted_main], axis=0)
            acq_predicted_main['merchant_name']=merchant_name

            predicted_main = predicted_main.merge(acq_predicted_main, on=['merchant_name','group'], how='left')
            predicted_main['merchant_name']=merchant_name

            # create initial order column. if group == time, then 1, else 0
            actual_main['initial_order_per_cust'] = np.where(actual_main['group']==actual_main['time'], 1, 0) 
            predicted_main['initial_order_per_cust'] = np.where(predicted_main['group']==predicted_main['time'], 1, 0) 

            actual_main['sales'] = actual_main['cohort_size'] * actual_main['aov'] * (actual_main['rpt_orders_per_cust'] + actual_main['initial_order_per_cust'] )
            predicted_main['sales'] = predicted_main['cohort_size'] * predicted_main['aov'] * (predicted_main['rpt_orders_per_cust'] + predicted_main['initial_order_per_cust'])
            
            actual_main_lst.append(actual_main)
            predicted_main_lst.append(predicted_main)
        
        actual_main_concat = pd.concat(actual_main_lst, axis=0, ignore_index=True)
        predicted_main_concat = pd.concat(predicted_main_lst, axis=0, ignore_index=True)
        return actual_main_concat, predicted_main_concat

    else:
        
        predicted_main_acq0 = predicted_main_acq_raw
        actual_main0 = actual_main_raw
        predicted_main0 = predicted_main_raw
        
        ## to avoid cohort_size key error in repeated merging
        if 'cohort_size' in predicted_main0.columns: 
            predicted_main0.drop(columns=['cohort_size'], inplace=True)
                
        predicted_main_acq=predicted_main_acq0.copy()
        actual_main=actual_main0.copy()
        predicted_main=predicted_main0.copy()
        
        ## prepare acquisition data
        acq_predicted_main = predicted_main_acq
        acq_predicted_main.columns = ['group','cohort_size']
        acq_predicted_main['group'] = acq_predicted_main['group'].astype(str) # change datetime64[ns] to object

        acq_actual_main = actual_main[['group','cohort_size']].drop_duplicates().reset_index(drop=True)
        acq_actual_main = acq_actual_main[acq_actual_main['group'] < TEST_START]

        # combine acq_predicted_main and acq_actual_main by rows
        acq_predicted_main = pd.concat([acq_actual_main, acq_predicted_main], axis=0)

        predicted_main = predicted_main.merge(acq_predicted_main, on='group', how='left')

        # create initial order column. if group == time, then 1, else 0
        actual_main['initial_order_per_cust'] = np.where(actual_main['group']==actual_main['time'], 1, 0) 
        predicted_main['initial_order_per_cust'] = np.where(predicted_main['group']==predicted_main['time'], 1, 0) 

        actual_main['sales'] = actual_main['cohort_size'] * actual_main['aov'] * (actual_main['rpt_orders_per_cust'] + actual_main['initial_order_per_cust'] )
        predicted_main['sales'] = predicted_main['cohort_size'] * predicted_main['aov'] * (predicted_main['rpt_orders_per_cust'] + predicted_main['initial_order_per_cust'])

        return actual_main, predicted_main



def rolling_forecast_stl(test_datasets, model, device, TARGET_TASK):
    # initialize empty list to store predictions for entire cohorts
    main_value_test_pred_lst = []
    
    for test_dataset in test_datasets:
        current_seq = test_dataset[0]['target'].to(device) # initialize current sequence with the first input_chunk_length values of the cohort
        pred_seq = [] # initialize empty list to store predictions for each cohort  
        
        # over samples in the cohort
        for j in range(len(test_dataset)):        
            # concatenate current sequence (from j-1th sample) with covariate
            test_input = torch.cat((current_seq, test_dataset[j]['covariate'].to(device)), dim=-1) # all gpu
            # reshape from (input_chunk_length, input_dim) to (1, input_chunk_length, input_dim)
            test_input = test_input.unsqueeze(0).float().to(device) # to gpu
            # predict next value
            pred_point = model(test_input) # gpu
            # detach tensors (still on device) and append detached tensors to lists (if they are tensors)
            pred_seq.append(pred_point.detach().cpu().numpy()) # all cpu
            # concatenate and convert to tensor (all on device)
            current_seq = torch.cat((current_seq[1:], pred_point), dim=0) # all gpu
            
        main_value_test_pred_lst.append(np.ravel(pred_seq))
        
    main_value_test_pred = {TARGET_TASK: main_value_test_pred_lst}
    return main_value_test_pred




