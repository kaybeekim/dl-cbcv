
print('util.py is loaded.')

## when loading for-loop, uncomment all 'continue' command

import os
import glob



def get_file_lists(READ_DIR):    
    filepaths = glob.glob(f'{READ_DIR}/*')

    ## trim down the file string (based one file naming structure)
    prefix_len = len(f'{READ_DIR}/')
    suffix_len = len('_cohort.csv')

    ## get idx - parent merchant - merchant filename
    idx_parentmerchant_merchant = [filepath[prefix_len:][:-suffix_len] for filepath in filepaths]
    idx_parentmerchant_merchant.sort()

    ## get parent merchant - merchant filename
    parentmerchant_merchant = ['_'.join(s.split('_')[1:]) for s in idx_parentmerchant_merchant]
    
    return filepaths, idx_parentmerchant_merchant, parentmerchant_merchant



def get_company_file(READ_DIR, MERCHANT_NAME):
    filepaths = glob.glob(os.path.join(READ_DIR, f'*{MERCHANT_NAME}*')) # get all file paths that contain the company name
    
    if len(filepaths) != 1:
        print('number of company files searched:', len(filepaths)) # company names can be duplicated
        print('we pick the first file onf this list')
    filepath = filepaths[0] if filepaths else None # select the first file if multiple files are found

    if filepath:
        print('filepath:', filepath)
        return filepath
    else:
        print('No files found for the given merchant name.')
        return None



def change_raw_df_column_names(raw_df, MERCHANT_NAME, TRAIN_START, TEST_START, 
                               group_identifier, time_identifier, 
                               acquisition_identifier,
                               order_identifier, 
                               spend_identifier):
    '''
    if for-loop script for company specific training, nudge this:
    if raw_df == None:
        continue
    '''
    raw_df = raw_df.rename(columns={group_identifier: 'group', 
                                    time_identifier: 'time',
                                    acquisition_identifier: 'cohort_size',
                                    order_identifier: 'orders',
                                    spend_identifier: 'spend'})        
  
    if len(raw_df[raw_df['time'] < TRAIN_START])==0:
        print('NOTE:', MERCHANT_NAME, 'censored period is empty thus dropped')
        return None
    
    if len(raw_df[raw_df['time'] >= TEST_START])==0:
        print('NOTE:', MERCHANT_NAME, 'test period is empty thus dropped')
        return None
    
    if len(raw_df[raw_df['cohort_size'] == 0]) > 0:
        print('NOTE:', MERCHANT_NAME, 'there exists zero acquisition size')

    if raw_df['time'].max().strftime('%Y-%m-%d') > raw_df['group'].max().strftime('%Y-%m-%d'):
        print('end of group:', raw_df['group'].max().strftime('%Y-%m-%d'))
        print('end of time:', raw_df['time'].max().strftime('%Y-%m-%d'))
        print('WARNING: End of group < end of time -- ')
        raw_df = raw_df[raw_df.time <= raw_df.group.max()]
    
    # Ensure to convert the 'time' column to datetime format (missing this processing can throw an error in the next step)
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    
    return raw_df    
    


import calendar

def get_week_start(raw_df, TRAIN_START, TEST_START, TEST_END):
    week_start = raw_df['time'][0].weekday() # check: here 6 is Sunday
    FREQ = f'w-{calendar.day_name[week_start][:3].upper()}'
    print(f'week start day is {week_start}, {FREQ}, {calendar.day_name[week_start]}')
    
    ## calculate length of train:test period
    observed_period = len(raw_df[['time']][(raw_df['time']>=TRAIN_START) & (raw_df['time']<TEST_START)].drop_duplicates())
    unobserved_period = len(raw_df[['time']][(raw_df['time']>=TEST_START) & (raw_df['time']<=TEST_END)].drop_duplicates())
    # observed_period_unleftcensored = len(raw_df[['time']][(raw_df['time']<=TRAIN_END)].drop_duplicates())
    # unobserved_period_unrightcensored = len(raw_df[['time']][(raw_df['time']>=TEST_START)].drop_duplicates())

    print('observed (train start to test end) period (weeks) '+ str(observed_period))
    print('unobserved (test start to test end) period (weeks) '+ str(unobserved_period))

    # this will be used for generating zero padding, calendar covariates etc
    return FREQ, week_start


### TO DO LATER: 
## acquisition or sales - time series data do not require acq_week - week panel format thus can take time aggregated format



import pandas as pd
import numpy as np

def generate_behaviorfeatures(raw_df):
    # sanity check if all the minimum required variables in data
    if not all(col in raw_df.columns for col in ['group', 'time', 'orders', 'spend', 'rpt_spend', 'cohort_size']):
        raise ValueError(f"{col} does not exist in data")
    
    raw_df = raw_df.sort_values(by=['group', 'time']).reset_index(drop=True)
    
    # create tenure
    raw_df['tenure'] = ((raw_df['time'] - raw_df['group']).dt.days / 7).astype(int)
    
    # create initial order
    raw_df['initial_order'] = np.where(raw_df['tenure'] == 0, raw_df['cohort_size'], 0)
    raw_df['initial_order_per_cust'] = raw_df['initial_order'] / raw_df['cohort_size'] # can have null if cohort_size is zero
    
    # create rpt_orders
    raw_df['rpt_orders'] = raw_df['orders'] - raw_df['initial_order']
    raw_df['rpt_orders_per_cust'] = raw_df['rpt_orders'] / raw_df['cohort_size'] # can have null if cohort_size is zero

    # create initial spend
    raw_df['initial_spend'] = raw_df['spend'] - raw_df['rpt_spend']    
    raw_df['initial_spend_per_cust'] = raw_df['initial_spend'] / raw_df['cohort_size'] # can have null if cohort_size is zero   
    
    # create rpt_spend
    raw_df['rpt_spend_per_cust'] = raw_df['rpt_spend'] / raw_df['cohort_size'] # can have null if cohort_size is zero

    # create aov
    raw_df['aov'] = raw_df['spend'] / raw_df['orders'] # can have null if orders is zero
    raw_df['initial_aov'] = raw_df['initial_spend'] / raw_df['initial_order'] # can have null if cohort_size is zero
    raw_df['rpt_aov'] = raw_df['rpt_spend'] / raw_df['rpt_orders'] # can have null if cohort_size is zero

    # fill missing values in AOV = 0/0 ( 0 spend and 0 orders)
    # propagate non-null values forward within each cohort (as always initial aov for each cohort is non-null)
    raw_df[['aov', 'initial_aov', 'rpt_aov']] = raw_df[['aov', 'initial_aov', 'rpt_aov']].fillna(method = 'ffill')

    # if cohort_size is zero, these are imputed as zero
    raw_df['initial_order_per_cust'] = np.where(raw_df['cohort_size'] == 0, 0, raw_df['initial_order_per_cust'])
    raw_df['rpt_orders_per_cust'] = np.where(raw_df['cohort_size'] == 0, 0, raw_df['rpt_orders_per_cust'])
    raw_df['initial_spend_per_cust'] = np.where(raw_df['cohort_size'] == 0, 0, raw_df['initial_spend_per_cust'])
    raw_df['rpt_spend_per_cust'] = np.where(raw_df['cohort_size'] == 0, 0, raw_df['rpt_spend_per_cust'])
    raw_df['initial_aov'] = np.where(raw_df['cohort_size'] == 0, 0, raw_df['initial_aov'])

    # IMPORTANT: generate cohort size along week (for training MTL)
    acq_df = raw_df[raw_df['tenure']==0][['group','cohort_size']]
    acq_df.columns = ['time','cohort_size_pseudo']
    raw_df = raw_df.merge(acq_df, on='time', how='left')

    # IMPORTANT: generate fake sales for regularization term
    raw_df['spend_pseudo'] = raw_df['cohort_size_pseudo'] * raw_df['aov'] * ( raw_df['rpt_orders_per_cust'] + np.where(raw_df['tenure'] == 0, 1, 0) )
    raw_df['rpt_spend_pseudo'] = raw_df['cohort_size_pseudo'] * raw_df['aov'] * raw_df['rpt_orders_per_cust']

    # check if there is any zero values in N_week_cohort
    if not (raw_df['cohort_size'] == 0).any():

        assert np.allclose(raw_df['initial_order']/raw_df['cohort_size_pseudo'], np.where(raw_df['tenure'] == 0, 1, 0), atol=1e-6), \
            "initial_order per customer is not accurately derived"

        assert np.allclose((raw_df['rpt_orders_per_cust'] + raw_df['initial_order']/raw_df['cohort_size']), 
                        raw_df['orders']/raw_df['cohort_size'], atol=1e-6), \
            "order per customer is not accurately derived"
    else:
        print("There are zero values in cohort_size.")

    # quick check for data integrity
    sales_recovered = raw_df['cohort_size'] * raw_df['aov'] *\
        (raw_df['rpt_orders_per_cust'] + np.where(raw_df['tenure'] == 0, 1, 0))

    sales_recovered = sales_recovered.fillna(0) # e.g. 135_ross

    try:
        assert np.allclose(sales_recovered, raw_df['spend'], atol=1e-4)
    except AssertionError:
        print("The original cohort-week-spend and CBCV based recovered spend are not element-wise equal within 1e-6 tolerance.")
        print("sales_recovered:", sum(sales_recovered))
        print("raw_df['spend']:", sum(raw_df['spend']))
        raise

    cols = ['group', 'time', 'cohort_size', 'tenure',
            'orders','rpt_orders_per_cust','aov','spend']  # the columns to move
    rest_of_cols = [col for col in raw_df.columns if col not in cols]  # the rest of the columns

    # rearrange column order
    raw_df = raw_df[cols + rest_of_cols]
    
    return raw_df



## TO DO: 
## separate parts of generating covariates (group_int, attention_mask, tenure etc) from zero padding

def zero_padding(raw_df, targets, company_meta, feature_additional, INPUT_CHUNK_LENGTH, FREQ,
                 use_merchant_embedding = False, merchant_name = None, verbose = True):
    if not use_merchant_embedding:
        df_unpadded = raw_df[['group', 'time'] + targets]
    else:
        assert merchant_name is not None
        df_unpadded = raw_df[[merchant_name, 'group', 'time'] + targets]

    # Define a function to add additional rows with 0 values for the 'value' column in a INPUT_CHUNK_LENGTH weeks look back window
    def add_rows(group_seq, numzeros=INPUT_CHUNK_LENGTH + 1, use_merchant_embedding=use_merchant_embedding):
        min_date = group_seq['time'].min()
        look_back_weeks = pd.date_range(end=min_date, periods=numzeros, freq=FREQ)[:-1]
        target_dict = {target: 0. for target in targets}
        
        if not use_merchant_embedding:
            new_rows = pd.DataFrame({'group': group_seq['group'].iloc[0], 
                                    'time': look_back_weeks,
                                    **target_dict})
        else:
            new_rows = pd.DataFrame({merchant_name: group_seq[merchant_name].iloc[0],
                                     'group': group_seq['group'].iloc[0], 
                                     'time': look_back_weeks,
                                     **target_dict})
        return pd.concat([new_rows, group_seq])

    # Apply the function to each group and concatenate the results
    if not use_merchant_embedding:
        df_padded = df_unpadded.groupby('group').apply(add_rows).reset_index(drop=True)
        # merge additional features
        df_padded = pd.merge(df_padded,
                            raw_df[['group','time'] + feature_additional],
                            on=['group','time'], how='left')    
        # merge company meta
        for col in company_meta:
            df_padded[col] = raw_df[col].iloc[0]

    else:
        df_padded = df_unpadded.groupby([merchant_name, 'group']).apply(add_rows).reset_index(drop=True)
        # merge additional features
        df_padded = pd.merge(df_padded,
                            raw_df[[merchant_name, 'group','time'] + feature_additional],
                            on=[merchant_name, 'group','time'], how='left')

        # quick check for data integrity
        assert df_padded.shape[0] - df_unpadded.shape[0] == df_padded[[merchant_name,'group']].drop_duplicates().shape[0] * INPUT_CHUNK_LENGTH, \
            "The length of the padded DataFrame should be equal to the length of the original DataFrame plus the expected number of zero pads."

        # merge company meta
        company_meta = raw_df[company_meta].drop_duplicates()
        df_padded = pd.merge(df_padded, company_meta, on=[merchant_name], how='left')

    if verbose:
        print(f'before: {len(df_unpadded)} rows')
        print(f'after: {len(df_padded)} rows')

    return df_padded



# ## These to are identical:

# ## option A)
# df_padded = util.zero_padding(raw_df, [TARGET_TASK], company_meta, feature_additional, INPUT_CHUNK_LENGTH, FREQ,
#                                use_merchant_embedding=True, merchant_name='merchant_name')

# ## option B)
# df_padded_lst = []
# for merchant_name in MERCHANT_NAMES_EMB_IDX.keys():
#     subdf = raw_df[raw_df['merchant_name'] == merchant_name]
#     subdf_padded = util.zero_padding(subdf, [TARGET_TASK], company_meta, feature_additional, INPUT_CHUNK_LENGTH, FREQ)
#     df_padded_lst.append(subdf_padded)

# df_padded = pd.concat(df_padded_lst)














from sklearn.preprocessing import MinMaxScaler
import holidays
from datetime import timedelta
import numpy as np

def generate_calendartime_features(df_padded, FREQ, week_start, dummy_ver=False):
    
    def generate_week_index(df_padded):
        start = (pd.to_datetime(df_padded['time'].min())).strftime('%Y-%m-%d')
        end = (pd.to_datetime(df_padded['time'].max())).strftime('%Y-%m-%d')
        week_index = pd.date_range(start=start, end=end, freq=FREQ)
        
        ## shift one week forward so that use future covariates to predict future sales
        start_shifted = (pd.to_datetime(df_padded['time'].min()) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        end_shifted = (pd.to_datetime(df_padded['time'].max()) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')        
        week_index_shifted = pd.date_range(start=start_shifted, end=end_shifted, freq=FREQ)
        print(len(week_index_shifted), 'weeks in our zero padded raw data')
        
        return week_index, week_index_shifted

    def generate_holiday_dummy(week_index, week_index_shifted):
        year_scope = range(week_index[0].year, week_index[-1].year)
        country_holidays = holidays.CountryHoliday('US', years=year_scope)
        country_holidays_series = pd.Series(pd.to_datetime(list(country_holidays.keys())))
        
        country_holidays_week = country_holidays_series - pd.to_timedelta(country_holidays_series.dt.dayofweek - week_start, unit='D')
        
        # Subtract 7 days from the week if the day of the week is less than week_start
        country_holidays_week = np.where(country_holidays_series.dt.dayofweek < week_start, country_holidays_week - timedelta(days=7), country_holidays_week)
        country_holidays_week = pd.Series(pd.to_datetime(country_holidays_week).unique()).sort_values().reset_index(drop=True)
        holiday_series = pd.Series(week_index_shifted).apply(lambda x: x in pd.Series(country_holidays_week).values)
        holiday_series = np.array(holiday_series*1).reshape(-1,1)
        print(f'{np.sum(holiday_series)} holiday weeks are identified\n')
        
        return holiday_series, country_holidays

    week_index, week_index_shifted = generate_week_index(df_padded)
    holiday_series, country_holidays = generate_holiday_dummy(week_index, week_index_shifted)

    # calendar covariates for 1 week ahead
    year_series = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(week_index_shifted.year).reshape(-1,1))
    month_series = pd.get_dummies(week_index_shifted.month).values
    
    if dummy_ver:
        weekofyear_series = pd.get_dummies(week_index.isocalendar().week).values
    else:
        weekofyear_series = (week_index_shifted.isocalendar().week - 1).values.reshape(-1,1)

    # time varying covariates
    linear_trend = np.linspace(start = 0, stop = 1, num = len(week_index_shifted), dtype=np.float32).reshape(-1,1)
    quadratic_trend = linear_trend**2

    # concatenate all covariates of interests
    covariates = np.concatenate((weekofyear_series, holiday_series, linear_trend, quadratic_trend), axis = 1)
    covariates_pd = pd.DataFrame(covariates, index=week_index,
                                 columns=['week_' + str(i+1) for i in range(weekofyear_series.shape[1])] +\
                                     ['holidays_1w_ahead', 'linear_trend', 'quadratic_trend'])
    
    if not dummy_ver:
        covariates_pd.rename(columns={'week_1': 'week_int'}, inplace=True)
        covariates_pd[['week_int']] = covariates_pd[['week_int']].astype(int)
        
    covariates_pd['time'] = week_index
    df_padded_w_cov = pd.merge(df_padded, covariates_pd, on=['time'], how='left')
    df_padded_w_cov[['holidays_1w_ahead','linear_trend','quadratic_trend']] = df_padded_w_cov[['holidays_1w_ahead','linear_trend','quadratic_trend']].astype(int)

    return df_padded_w_cov, country_holidays



## TO DO:
## censored cohort embedding 2016 uniformly or first 3 months for each merchant

def generate_cohort_features(df_padded, targets, COHORT_EMBEDDING=False, dummy_ver=False, COHORT_EMB_NUM=None):
    # create cohort tenure, cohort integer, attention mask
    df_padded['tenure'] = ((df_padded['time'] - df_padded['group']).dt.days / 7).astype(int)
    df_padded['group_int'], _ = pd.factorize(df_padded['group'])
    df_padded['attention_mask'] = (df_padded['time'] < df_padded['group']).astype(int)
    
    # group-time varying covariates
    tenure_series = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df_padded['tenure']).reshape(-1,1))
    tenure_series = tenure_series*np.array(1-df_padded['attention_mask']).reshape(-1,1)
    quadratic_tenure_series = tenure_series**2
    
    # attach group related columns
    if COHORT_EMBEDDING:
        assert COHORT_EMB_NUM is not None
        # group varying covariates : group seasonality
        df_padded['cohort_month_int'] = pd.to_datetime(df_padded['group']).dt.month
        if dummy_ver:
            cohort_month_dummies = pd.get_dummies(df_padded['cohort_month_int'], prefix='cohort_m')
            df_padded = pd.concat([df_padded, cohort_month_dummies], axis=1)

        # group varying covariates : censored cohort one hot encoding
        first_n_groups = df_padded['group'].drop_duplicates().iloc[COHORT_EMB_NUM]
        df_padded['group_censored'] = df_padded['group'].apply(lambda x: first_n_groups.strftime("%Y-%m-%d") if x >= pd.to_datetime(first_n_groups) else x.strftime("%Y-%m-%d"))
        df_padded['group_censored_int'], _ = pd.factorize(df_padded['group_censored'])
        if dummy_ver:
            cohort_group_dummies = pd.get_dummies(df_padded['group_leftcensored'])
            cohort_group_dummies.columns = [f'cohort_{col}' for col in cohort_group_dummies.columns]
            df_padded = pd.concat([df_padded, cohort_group_dummies], axis=1)
    else:
        # group varying covariates : group quadratic trend
        group_int_series = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df_padded['group_int']).reshape(-1,1))
        quadratic_group_int_series = group_int_series**2
        df_padded["group_int_scaled"] = group_int_series
        df_padded["quad_group_int_scaled"] = quadratic_group_int_series

    df_padded["tenure_scaled"] = tenure_series
    df_padded["quad_tenure_scaled"] = quadratic_tenure_series

    # change back to string format
    df_padded["group"] = df_padded["group"].dt.strftime("%Y-%m-%d")
    
    cols = ['group', 'time', 'tenure'] + targets  # the columns to move
    rest_of_cols = [col for col in df_padded.columns if col not in cols]  # the rest of the columns
    # rearrange column order
    df_padded = df_padded[cols + rest_of_cols]
    
    # change back to string format
    df_padded["time"] = df_padded["time"].dt.strftime("%Y-%m-%d")
        
    return df_padded



def check_holidays(df_padded_w_cov, country_holidays):
    ## checking holiday calendar
    for key in sorted(country_holidays.keys()):
        value = country_holidays[key]
        print(f'{key}: {value}')

    print("print(df_padded_w_cov[['time','holidays_1w_ahead']][df_padded_w_cov['holidays_1w_ahead']==1].drop_duplicates())")
    print(df_padded_w_cov[['time','holidays_1w_ahead']][df_padded_w_cov['holidays_1w_ahead']==1].drop_duplicates())

    # ensure holidays_1w_ahead are really 1 week ahead (t+1), and others are t
    print(df_padded_w_cov[['group','time','cohort_size_pseudo','rpt_orders_per_cust','aov','spend',
                           'week_1','tenure_scaled','holidays_1w_ahead']][df_padded_w_cov['time']>='2016-01-01'])
    print(df_padded_w_cov[['group','time','holidays_1w_ahead','cohort_size_pseudo','rpt_orders_per_cust','aov','spend',
                           'week_1','tenure_scaled']][df_padded_w_cov['holidays_1w_ahead']==1])



import inspect

def split_dataframe(df, TRAIN_START, VAL_START, TEST_START, VAL_START_with_offset, TEST_START_with_offset, VAL_LOSS,
                    verbose=True):
    censored_df = df[(df['group'] < TRAIN_START)]
    main_df = df[(df['group'] >= TRAIN_START)]

    if VAL_LOSS:
        censored_df_train = censored_df[(censored_df['time'] < VAL_START)]
        censored_df_valid = censored_df[(censored_df['time'] >= VAL_START_with_offset) &
                                        (censored_df['time'] < TEST_START)]

        main_df_train = main_df[(main_df['group'] < VAL_START) &
                                (main_df['time'] < VAL_START)]
        main_df_valid = main_df[(main_df['group'] < TEST_START) &
                                (main_df['time'] >= VAL_START_with_offset) &
                                (main_df['time'] < TEST_START)]
    else:
        censored_df_train = censored_df[(censored_df['time'] < TEST_START)]
        main_df_train = main_df[(main_df['group'] < TEST_START) &
                                (main_df['time'] < TEST_START)]        
        censored_df_valid = pd.DataFrame() # just emopty dataframe
        main_df_valid = pd.DataFrame()

    censored_df_test = censored_df[censored_df['time'] >= TEST_START_with_offset]
    main_df_test = main_df[main_df['time'] >= TEST_START_with_offset]
    
    def group_length(df, VAL_LOSS):
        if VAL_LOSS:
            # length of sequence for each group
            group_lengths = df.groupby('group').size().values        
            
            # Get the name of df
            df_name = [k for k, v in inspect.currentframe().f_back.f_locals.items() if v is df][0]
        
            print(f'{df_name}: cohorts {len(group_lengths)} * sequence length [{np.max(group_lengths)} to {np.min(group_lengths)}]')
        else:
            pass

    if verbose:
        # check the size of each data (height=num of cohorts * upper width to lower width=length of time)
        group_length(censored_df_train, True) # A
        group_length(censored_df_valid, VAL_LOSS) # B
        group_length(censored_df_test, True) # C
        group_length(main_df_train, True) # D
        group_length(main_df_valid, VAL_LOSS) # E
        group_length(main_df_test, True) # F

    return main_df, main_df_train, main_df_valid, main_df_test, censored_df, censored_df_train, censored_df_valid, censored_df_test





def df_to_numpy(df, TASKS, COVARIATES, use_merchant_embedding=False, merchant_name=None):    
    # group by
    if not use_merchant_embedding:
        df_groupby = df.groupby(['group']) # string in dict_keys
    else:
        assert merchant_name is not None
        df_groupby = df.groupby([merchant_name, 'group']) # tuple in dict_keys

    # Groups
    group_sequence = list(df_groupby.groups.keys()) # list length (num_groups)

    # Extracting time sequences for each group into a list of numpy arrays
    time_sequences = [group['time'].values for _, group in df_groupby] # list length (num_groups) * np shape (seq_len,)

    # Transform covariates into a list of numpy arrays, grouped by the 'group' column
    cov_sequences = [group[COVARIATES].values for _, group in df_groupby] # list length (num_groups) * np shape (seq_len, num_covariates)

    # Transform each column into a list of numpy arrays, grouped by the 'group' column
    value_sequences_dict = {} # dictionary to store the value sequences
    for col in TASKS:
        value_sequences_dict[col] = df_groupby.apply(lambda x: x[col].to_numpy()).tolist() # dictionary (targets) * list length (num_groups) * np shape (seq_len,)

    return {
        'group_seq': group_sequence,
        'time_seq': time_sequences,
        'value_seq_dict': value_sequences_dict,
        'cov_seq': cov_sequences
    }





from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer

def scale_arrays(arrays, scaler=None):
    # Step 1: Concatenate all arrays into a single array
    concatenated_array = np.concatenate(arrays).reshape(-1, 1)  # Reshape to 2D array with shape [X, 1]

    # Step 2: Apply Scaler
    if scaler is None:
        ## option 1 common scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # ## option 2 robust to outliers
        # scaler = RobustScaler(quantile_range=(25.0, 75.0))
        
        scaler.fit(concatenated_array)

    scaled_array = scaler.transform(concatenated_array)

    # Step 3: Split the scaled array back into individual arrays of original lengths
    split_indices = np.cumsum([len(arr) for arr in arrays])[:-1]  # Exclude the last sum
    scaled_arrays = np.split(scaled_array, split_indices)

    # Convert 2D scaled arrays back to 1D for each group
    scaled_arrays = [arr.flatten() for arr in scaled_arrays]

    return scaler, scaled_arrays



def df_to_scaled_numpy(df, TASKS, COVARIATES, scalers_inherited=None,
                       use_merchant_embedding=False, merchant_name=None):

    # step 1: transform df to splitted numpy arrays
    np_dict = df_to_numpy(df, TASKS, COVARIATES, use_merchant_embedding=use_merchant_embedding,
                          merchant_name=merchant_name)
    group_sequences = np_dict['group_seq']
    time_sequences = np_dict['time_seq']
    value_sequences = np_dict['value_seq_dict'] 
    cov_sequences = np_dict['cov_seq']
    
    # step 2: scale each task
    scaled_value_sequences = {} # dictionary to store the value sequences
    if scalers_inherited is None:    
        scalers = {} # dictionary to store the scalers
        for col in TASKS:
            scaler, scaled_value_sequences[col] = scale_arrays(value_sequences[col])
            scalers[col] = scaler
    else:
        scalers = scalers_inherited
        for col in TASKS:
            _, scaled_value_sequences[col] = scale_arrays(value_sequences[col], scalers[col])

        # combine scaled_value_sequences into a singlie ist of numpy arrays
        # combined_scaled_value_sequences = [np.stack([scaled_value_sequences[col] for col in TASK], axis=1) for i in range(len(group_sequences))]

        # # reverse scaling
        # combined_value_sequences = [np.stack([scalers[col].inverse_transform(combined_scaled_value_sequences[i][:,j].reshape(-1,1)).flatten()
        #                                       for j, col in enumerate(TASK)], axis=1)
        #                             for i in range(len(group_sequences))]

    return {
        'group_seq': group_sequences,
        'time_seq': time_sequences,
        'scaler': scalers,
        'scaled_value_seq_dict': scaled_value_sequences,
        'cov_seq': cov_sequences
    }


# ## TO DO: scaler check
# def scaler_info(scaler_dic, key):
#     print(scaler_dic[key].min_)
#     print(scaler_dic[key].data_max_)
#     print(scaler_dic[key].data_range_)
#     print(scaler_dic[key].scale_, '\n')

# scaler_info(censored_train_scaler, 'cohort_size_pseudo')
# scaler_info(main_train_scaler, 'cohort_size_pseudo')
# scaler_info(censored_train_scaler, 'rpt_orders_per_cust')
# scaler_info(main_train_scaler, 'rpt_orders_per_cust')
# scaler_info(censored_train_scaler, 'aov')
# scaler_info(main_train_scaler, 'aov')



# ## TO DO: scaled np data format check
# def print_shape(**kwargs):
#     for name, arg in kwargs.items():
#         if isinstance(arg, np.ndarray):
#             print(f'Shape of np array {name}: {arg.shape}')

#         elif isinstance(arg, list):
#             print(f'Length of list {name}: {len(arg)}')
#             if not isinstance(arg[0], str):
#                 print(f'Length of first element of list {name}: {arg[0].shape}')
#                 print(f'Length of last element of list {name}: {arg[-1].shape}')

#         elif isinstance(arg, dict):
#             print(f'Length of dic {name}: {len(arg)}')
#             print(f'Length of first element of first dict key of {name}: {arg[list(arg.keys())[0]][0].shape}')
#             print(f'Length of last element of first dict key of {name}: {arg[list(arg.keys())[0]][-1].shape}')
#     print('\n')

# print_shape(whole_group=whole_group, whole_value=whole_value, whole_time=whole_time, whole_cov=whole_cov)
# print_shape(censored_group=censored_group, censored_value_train=censored_value_train, censored_time_train=censored_time_train, censored_cov_train=censored_cov_train)
# print_shape(censored_value_valid=censored_value_valid, censored_time_valid=censored_time_valid, censored_cov_valid=censored_cov_valid)
# print_shape(censored_value_test=censored_value_test, censored_time_test=censored_time_test, censored_cov_test=censored_cov_test)
# print_shape(main_group_train=main_group_train, main_value_train=main_value_train, main_time_train=main_time_train, main_cov_train=main_cov_train)
# print_shape(main_group_valid=main_group_valid, main_value_valid=main_value_valid, main_time_valid=main_time_valid, main_cov_valid=main_cov_valid)
# print_shape(main_group_test=main_group_test, main_value_test=main_value_test, main_time_test=main_time_test, main_cov_test=main_cov_test)



def inverse_scale_np_to_dataframe(all_scalers, all_np_arrays, grouptimedf):

    ## group-time framework prepared
    inverse_scaled_df = pd.DataFrame({'group': grouptimedf['group'], 'time': grouptimedf['time']})

    ## for each task
    for col in list(all_np_arrays.keys()):
        np_arrays = all_np_arrays[col] # list shape: (num_groups, group specific seq_len)

        # Step 1: Concatenate all arrays into a single array
        concatenated_array = np.concatenate(np_arrays).reshape(-1, 1)  # np array shape: (num_groups * group specific seq_len, 1)

        # Step 2: Inverse transform with scaler
        if all_scalers is not None:           
            scaler = all_scalers[col]
            concatenated_array = scaler.inverse_transform(concatenated_array)

        # Step 3: Split the inverse scaled array back into individual arrays of original lengths
        split_indices = np.cumsum([len(arr) for arr in np_arrays])[:-1]  # e.g. 57,  114,  171, .. Exclude the last sum
        inverse_scaled_arrays = np.split(concatenated_array, split_indices)
        # Convert 2D inverse scaled arrays (.reshape(-1,1)) back to 1D for each group (.flatten())
        inverse_scaled_arrays = [arr.flatten() for arr in inverse_scaled_arrays]

        # Step 4: Convert the inverse scaled arrays back to a dataframe
        inverse_scaled_df[col] = np.concatenate(inverse_scaled_arrays)

    return inverse_scaled_df



def inverse_scale_np_to_dataframe_embedding(scaler_dict, group_lst, time_lst, scaled_value_dict, INPUT_CHUNK_LENGTH):

    df_lst = []
    for i in range(len(group_lst)):
        for t in time_lst[i][INPUT_CHUNK_LENGTH:]:
            df_lst.append({'merchant_name': group_lst[i][0],
                        'group': group_lst[i][1], 
                        'time': t,
                        })

    inverse_scaled_df = pd.DataFrame(df_lst)

    ## for each task
    for col in list(scaled_value_dict.keys()):

        scaled_value_lst = scaled_value_dict[col] # 

        # Step 1: Concatenate all arrays into a single array (from list shape: (num_groups, group specific seq_len) to (num_groups * group specific seq_len, 1))
        concatenated_array = np.concatenate(scaled_value_lst).reshape(-1, 1) 

        # Step 2: Inverse transform with scaler
        if scaler_dict is not None:           
            scaler = scaler_dict[col]
            concatenated_array = scaler.inverse_transform(concatenated_array)

        # Step 3: Split the inverse scaled array back into individual arrays of original lengths
        split_indices = np.cumsum([len(arr) for arr in scaled_value_lst])[:-1]  # e.g. 57,  114,  171, .. Exclude the last sum
        inverse_scaled_arrays = np.split(concatenated_array, split_indices)
        # Convert 2D inverse scaled arrays (.reshape(-1,1)) back to 1D for each group (.flatten())
        inverse_scaled_arrays = [arr.flatten() for arr in inverse_scaled_arrays]

        # Step 4: Convert the inverse scaled arrays back to a dataframe
        inverse_scaled_df[col] = np.concatenate(inverse_scaled_arrays)
        
    return inverse_scaled_df



def check_inverse_scale_np_to_dataframe(scalers, value_arrays, df, task_feature_names):
    original = df[ ['group', 'time'] + task_feature_names ]
    reversed = inverse_scale_np_to_dataframe(scalers, value_arrays, df[['group', 'time']])
    reversed.sort_values(['group', 'time'], inplace=True)

    ## check wether inverse_scale_np_to_dataframe recovers well
    for col in task_feature_names:
        assert np.allclose(original[col], reversed[col], atol=1e-6), \
            "The original df and reversed df are not element-wise equal within 1e-6 tolerance."



# # Only two scalers (one on censored cohort's train data, the other on main cohort's train data)
# util.check_inverse_scale_np_to_dataframe(censored_train_scaler, censored_value_train, censored_df_train, TASK_FEATURE_NAMES)
# util.check_inverse_scale_np_to_dataframe(censored_train_scaler, censored_value_test, censored_df_test, TASK_FEATURE_NAMES)
# util.check_inverse_scale_np_to_dataframe(censored_train_scaler, censored_value_valid, censored_df_valid, TASK_FEATURE_NAMES)
# util.check_inverse_scale_np_to_dataframe(main_train_scaler, main_value_train, main_df_train, TASK_FEATURE_NAMES)
# util.check_inverse_scale_np_to_dataframe(main_train_scaler, main_value_test, main_df_test, TASK_FEATURE_NAMES)
# util.check_inverse_scale_np_to_dataframe(main_train_scaler, main_value_valid, main_df_valid, TASK_FEATURE_NAMES)

# # # check manually
# # pd.DataFrame({'original': raw_df.N_week_cohort_alongweek[:10].values.reshape(-1,1).squeeze(),
# #               'scaled': censored_train_scaler['N_week_cohort_alongweek'].transform(raw_df.N_week_cohort_alongweek[:10].values.reshape(-1,1)).squeeze(),
# #               'inverse_scaled': censored_train_scaler['N_week_cohort_alongweek'].inverse_transform( censored_train_scaler['N_week_cohort_alongweek'].transform(raw_df.N_week_cohort_alongweek[:10].values.reshape(-1,1))).squeeze()})



## TO DO
# ## check training sample
# for i, sample in enumerate(train_dataset):
#   print(sample['target'], sample['gt'])
#   if i > 1:
#     break

# ## check the shape of each batch
# print(first_batch[0].shape) # target (batch_size, input_chunk_length, tgt_dim)
# print(first_batch[1].shape) # covariate (batch_size, input_chunk_length, cov_dim)
# print(first_batch[2].shape) # gt (batch_size, tgt_dim)
# print(torch.cat((first_batch[0], first_batch[1]), dim=-1).shape)





def get_min_max_values_from_scaler(main_train_scaler, TASK_FEATURE_NAMES, MERCHANT_NAMES_EMB_IDX, device):
    min_values = {target: {} for target in TASK_FEATURE_NAMES}
    max_values = {target: {} for target in TASK_FEATURE_NAMES}

    for merchant_idx in MERCHANT_NAMES_EMB_IDX.values():
        for target in targets:
            min_values[target][merchant_idx] = torch.tensor(main_train_scaler[merchant_idx][target].data_min_, dtype=torch.float32).to(device)
            max_values[target][merchant_idx] = torch.tensor(main_train_scaler[merchant_idx][target].data_max_, dtype=torch.float32).to(device)

    return min_values, max_values





### acquisition, sales time series data

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def split_acq_df_to_scaled_np(df, TARGET_TASK, COVARIATE_FEATURE_NAMES,
                          TRAIN_START_with_offset, TRAIN_END, TEST_START, TEST_START_with_offset,
                          VAL_LOSS, VAL_START_with_offset, VAL_END, verbose=True):
    ## transform pandas dataframe to to numpy array
    whole_np = np.array(df[TARGET_TASK]) # ground truth
    train_val_np = np.array(df[(df['time']>=TRAIN_START_with_offset) & (df['time'] < TEST_START)][TARGET_TASK])
    test_np = np.array(df[df['time'] >= TEST_START_with_offset][TARGET_TASK])

    ## if considering validation loss
    if VAL_LOSS:
        train_np = np.array(df[(df['time'] >= TRAIN_START_with_offset) & (df['time'] <= TRAIN_END)][TARGET_TASK]) 
        val_np = np.array(df[(df['time'] >= VAL_START_with_offset) & (df['time'] <= VAL_END)][TARGET_TASK])
        if verbose:
            print(f"Train len: {len(train_np)}, \
                \nValidation len: {len(val_np)}, \nTest len: {len(test_np)}, \nTotal len: {len(whole_np)}")
    else :
        train_np = train_val_np
        val_np = None
        if verbose:
            print(f"Train len (train period + input chunk length): {len(train_np)}, \
                \nTest len: {len(test_np)}, \nTotal len: {len(whole_np)}")

    ## scale data 
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    train_np_scaled = train_scaler.fit_transform(train_np.reshape(-1, 1)) # scaler fit by train data
    train_val_np_scaled = train_scaler.fit_transform(train_val_np.reshape(-1, 1))
    test_np_scaled = train_scaler.transform(test_np.reshape(-1,1))
    whole_np_scaled = train_scaler.transform(whole_np.reshape(-1,1))
    if VAL_LOSS:
        val_np_scaled = train_scaler.transform(val_np.reshape(-1,1))
    else:
        val_np_scaled = None

    ## build covariate np
    covariates_np = np.array(df[COVARIATE_FEATURE_NAMES])

    covariates_train_val_np = covariates_np[:len(train_val_np)]
    covariates_test_np = covariates_np[-len(test_np):]
    if verbose:
        print('covariate shape:', covariates_np.shape)

    if VAL_LOSS:
        covariates_train_np = covariates_train_val_np[:len(train_np)]
        covariates_val_np = covariates_train_val_np[-len(val_np):]
        if verbose:
            print('covariates_train shape:', covariates_train_np.shape)
            print('covariates_val shape:', covariates_val_np.shape)
    else:
        covariates_train_np = covariates_train_val_np
        covariates_val_np = None
        if verbose:
            print('covariates_train shape:', covariates_train_np.shape)
    
    if verbose:
        print('covariates_test shape:', covariates_test_np.shape)

    return train_scaler, train_np_scaled, val_np_scaled, train_val_np_scaled, test_np_scaled, \
        covariates_train_np, covariates_val_np, covariates_test_np
        
        
        
        
        
        
        
        
        

def read_files_generate_behaviorfeatures_get_embed_dict(READ_DIR, TRAIN_START, TEST_START, TEST_END,
                               group_identifier, time_identifier, 
                               acquisition_identifier,
                               order_identifier, 
                               spend_identifier):
    '''
    step 1: read all files in directory
    step 2: change column names
    step 3: drop df with no censored period or test period
    step 4: generate behavior features
    step 5: get embedding dictionary
    '''
    filepaths = glob.glob(f'{READ_DIR}/*.csv')
    filepaths.sort()

    df_list = []
    MERCHANT_NAME_LIST = []
    
    i = 0
    for filepath in filepaths:
        raw_df = pd.read_csv(filepath, parse_dates=[group_identifier,time_identifier])
        MERCHANT_NAME = filepath.split("/")[-1][:-len('_cohort.csv')]
        
        raw_df = change_raw_df_column_names(raw_df, MERCHANT_NAME, TRAIN_START, TEST_START,
                                                group_identifier='acq_week', time_identifier='week',
                                                acquisition_identifier='N_week_cohort',
                                                order_identifier = 'orders',
                                                spend_identifier = 'spend')
        if raw_df is None:
            continue
        
        raw_df = generate_behaviorfeatures(raw_df) # generate behavior features
        raw_df['merchant_emb_int'] = i # add merchant embedding index
        i += 1

        df_list.append(raw_df)
        MERCHANT_NAME_LIST.append(MERCHANT_NAME)

    ## get final df
    df = pd.concat(df_list, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    df['merchant_name'] = df[['merchant_index', 'parent_merchant', 'merchant']].astype(str).apply('_'.join, axis=1)   
    
    ## generate embedding dictionary
    MERCHANT_NAMES_EMB_INT = {merchant:idx for idx, merchant in enumerate(MERCHANT_NAME_LIST)} # e.g. {'wayfair': 0, 'shopify': 1, 'spotify': 2}
    print('Number of companies:', len(MERCHANT_NAMES_EMB_INT))

    return df, MERCHANT_NAMES_EMB_INT





def split_acq_df_to_scaled_np_pooled_ver(df, MERCHANT_NAMES_EMB_IDX, TARGET_TASK, COVARIATE_FEATURE_NAMES,
                          TRAIN_START_with_offset, TRAIN_END, TEST_START, TEST_START_with_offset,
                          VAL_LOSS, VAL_START_with_offset, VAL_END, verbose=True):
    ## scaler placeholder
    main_train_scaler_dict = {merchant_idx:None for merchant_idx in MERCHANT_NAMES_EMB_IDX.values()}

    ## train data as a list
    train_np_scaled_lst, val_np_scaled_lst, train_val_np_scaled_lst = [],[],[]
    covariates_train_np_lst, covariates_val_np_lst = [],[]

    ## test data as a dictionary
    test_np_scaled_dict = {merchant_idx:None for merchant_idx in MERCHANT_NAMES_EMB_IDX.values()}
    covariates_test_np_dict = {merchant_idx:None for merchant_idx in MERCHANT_NAMES_EMB_IDX.values()}

    for MERCHANT_NAME in list(MERCHANT_NAMES_EMB_IDX.keys()):
        MERCHANT_EMB_IDX = MERCHANT_NAMES_EMB_IDX[MERCHANT_NAME] # get merchant embedding index
        if verbose:
            print(f'{MERCHANT_NAME} ({MERCHANT_EMB_IDX}) starts!')
        sub_df = df[df['merchant_name']==MERCHANT_NAME].sort_values(by='time')

        train_scaler, train_np_scaled, val_np_scaled, train_val_np_scaled, test_np_scaled, \
            covariates_train_np, covariates_val_np, covariates_test_np = \
                split_acq_df_to_scaled_np(sub_df, TARGET_TASK, COVARIATE_FEATURE_NAMES,
                                            TRAIN_START_with_offset, TRAIN_END, TEST_START, TEST_START_with_offset,
                                            VAL_LOSS, VAL_START_with_offset, VAL_END, verbose=verbose)

        main_train_scaler_dict[MERCHANT_EMB_IDX] = train_scaler
        
        train_np_scaled_lst.append(train_np_scaled)
        val_np_scaled_lst.append(val_np_scaled)
        train_val_np_scaled_lst.append(train_val_np_scaled)
        covariates_train_np_lst.append(covariates_train_np)
        covariates_val_np_lst.append(covariates_val_np)

        test_np_scaled_dict[MERCHANT_EMB_IDX] = test_np_scaled
        covariates_test_np_dict[MERCHANT_EMB_IDX] = covariates_test_np
        
    return main_train_scaler_dict, train_np_scaled_lst, val_np_scaled_lst, train_val_np_scaled_lst, test_np_scaled_dict, \
        covariates_train_np_lst, covariates_val_np_lst, covariates_test_np_dict
    