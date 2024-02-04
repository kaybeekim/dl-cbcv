
print('model.py is loaded.')

import torch  # for building and training neural networks
from torch.utils.data import Dataset, DataLoader # for loading and managing datasets
import torch.nn as nn  # for building neural networks
import torch.nn.functional as F  # for implementing various activation functions
import torch.optim as optim  # for defining optimizers

import libs.hyperparam as hyperparam

import math # for math operations



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Calculate the positional encoding matrix
        pe = torch.zeros(max_len, d_model) # max length of the input sequence X dimension of the model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0) # unsqueezed to add batch dimension
        self.register_buffer("pe", pe) # saved as part of the state_dict and moved to the device

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    
    
    
    

class MTL_Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int, # input (target + covariates) dimension
        feature_dict: dict,
        d_model: int = hyperparam.D_MODEL, # input embedding dimension. number of features in the Transformer encoder/decoder inputs
        nhead: int = hyperparam.N_HEAD, # the number of heads in the multihead attention mechanism
        num_encoder_layers: int = hyperparam.N_ENCODER_LAYERS, # the number of layers in Transformer encoder
        num_decoder_layers: int = hyperparam.N_DECODER_LAYERS, # the number of layers in Transformer decoder
        d_feedforward: int = hyperparam.D_FEEDFORWARD, # the dimension of the feedforward network in Transformer encoder/decoder
        d_feedforward_task: int = hyperparam.D_FEEDFORWARD_TASK, # the dimension of the feedforward network in the task-specific decoder
        dropout: float = hyperparam.DROPOUT, # the dropout probability
        activation: str = hyperparam.ACTIVATION,  # the activation function
        input_chunk_length: int = hyperparam.INPUT_CHUNK_LENGTH, # sequence length
        output_chunk_length: int = hyperparam.OUTPUT_CHUNK_LENGTH, # =1, single horizon prediction
        num_merchant: int = None, # number of coompanies
    ):
        super().__init__()

       # set parameters
        self.input_dim = input_dim # input (target + covariates) dimension
        self.tgt_dim = input_dim - len(feature_dict.keys())
        self.feature_dict = feature_dict
        self.input_chunk_length = input_chunk_length # input sequence length
        self.output_chunk_length = output_chunk_length # output sequence length
        self.d_model = d_model
                
        total_input_dim = input_dim
        # Mapping from input size to d_model size
        if 'merchant_emb_int' in feature_dict.keys():
            assert num_merchant is not None
            self.merchant_emb = nn.Embedding(num_embeddings=num_merchant, embedding_dim=hyperparam.MERCHANT_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.MERCHANT_EMB_DIM - 1

        if 'week_int' in feature_dict.keys():
            self.week_emb = nn.Embedding(num_embeddings=52 + 1, embedding_dim=hyperparam.WEEK_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.WEEK_EMB_DIM - 1

        if 'cohort_month_int' in feature_dict.keys():
            self.cohort_month_emb = nn.Embedding(num_embeddings=12 + 1, embedding_dim=hyperparam.COHORT_MONTH_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.COHORT_MONTH_EMB_DIM - 1

        if 'group_censored_int' in feature_dict.keys():
            self.group_censored_emb = nn.Embedding(num_embeddings=hyperparam.COHORT_EMB_NUM + 1, embedding_dim=hyperparam.GROUP_CENSORED_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.GROUP_CENSORED_EMB_DIM - 1
                
        self.input_embedding = nn.Linear(total_input_dim, d_model) 

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(d_model, dropout, input_chunk_length)

        # initialize a transformer model using the nn.Transformer class
        self.transformer = nn.Transformer(
            d_model=d_model, # number of features in the encoder/decoder inputs
            nhead=nhead,  # the number of heads in the multihead attention mechanism
            num_encoder_layers=num_encoder_layers,  # the number of layers in the encoder
            num_decoder_layers=num_decoder_layers,  # the number of layers in the decoder
            dim_feedforward=d_feedforward,  # the dimension of the feedforward network
            dropout=dropout,  # the dropout probability
            activation=activation,  # the activation function
            batch_first=True,  # if True, (batch, seq, feature). if False, (seq, batch, feature).
        )

        def create_task_fc(d_model, d_feedforward_task, output_chunk_length):
            return nn.Sequential(
                nn.Linear(d_model, d_feedforward_task), nn.ReLU(), 
                nn.Linear(d_feedforward_task, d_feedforward_task//2), nn.ReLU(), 
                nn.Linear(d_feedforward_task//2, output_chunk_length))

        # Then you can create each task like this:
        self.task_1_fc = create_task_fc(d_model, d_feedforward_task, output_chunk_length)
        self.task_2_fc = create_task_fc(d_model, d_feedforward_task, output_chunk_length)
        self.task_3_fc = create_task_fc(d_model, d_feedforward_task, output_chunk_length)



    def forward(self, src_input):
        ## src: a sequence of features of shape (batch_size, input_chunk_length, input_dim)
        ## tgt: a sequence of features of shape (batch_size, output_chunk_length=1, input_dim)

        tgt_input = src_input[:,:,:self.tgt_dim]
        cov_input = src_input[:,:,self.tgt_dim:]

        # non-embedding features
        non_embed_covariate_keys = set(self.feature_dict.keys()) - set(['merchant_emb_int','week_int','cohort_month_int','group_censored_int'])
        remaining_feature_idx = [self.feature_dict[key] for key in non_embed_covariate_keys]
        remaining_feature_idx.sort()
        extended_src_input = torch.cat((tgt_input, cov_input[:,:,remaining_feature_idx]), dim=-1)

        # embedding        
        if 'merchant_emb_int' in self.feature_dict.keys():
            # Convert src_input to a LongTensor and send it to the same device as the model
            src_input_merchant = cov_input[:,:,self.feature_dict['merchant_emb_int']].long().to(self.merchant_emb.weight.device)
            merchant_embedding = self.merchant_emb(src_input_merchant)
            extended_src_input = torch.cat((extended_src_input, merchant_embedding), dim=-1)
        if 'week_int' in self.feature_dict.keys():
            src_input_week = cov_input[:,:,self.feature_dict['week_int']].long().to(self.week_emb.weight.device)
            week_embedding = self.week_emb(src_input_week)
            extended_src_input = torch.cat((extended_src_input, week_embedding), dim=-1)
        if 'cohort_month_int' in self.feature_dict.keys():
            src_input_cohort_month = cov_input[:,:,self.feature_dict['cohort_month_int']].long().to(self.cohort_month_emb.weight.device)
            cohort_month_embedding = self.cohort_month_emb(src_input_cohort_month)
            extended_src_input = torch.cat((extended_src_input, cohort_month_embedding), dim=-1)
        if 'group_censored_int' in self.feature_dict.keys():
            src_input_group_censored = cov_input[:,:,self.feature_dict['group_censored_int']].long().to(self.group_censored_emb.weight.device)
            group_censored_embedding = self.group_censored_emb(src_input_group_censored)
            extended_src_input = torch.cat((extended_src_input, group_censored_embedding), dim=-1)

        # Get the last value of src and use it as target input
        tgt = extended_src_input[:, -1, :] # shape: (batch_size, input_dim)
        tgt = tgt.unsqueeze(1) # unsqueezes it to match the shape of the target tensor. shape: (batch_size, 1, input_dim)
 
        # Apply the encoder layer and positional encoding to the source input
        src = self.input_embedding(extended_src_input) * math.sqrt(self.d_model) # common scaling factor
        src = self.positional_encoding(src) # (batch_size, input_chunk_length, d_model)

        # Apply the encoder layer and positional encoding to the target input
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt) # (batch_size, 1, d_model)
   
        # Pass the source and target input through the transformer
        x = self.transformer(src=src, tgt=tgt) # (batch_size, 1, d_model)

        # Pass the output through the decoder
        pred_1 = self.task_1_fc(x)[:, 0, :] # (batch_size, 1) # N_week_cohort_alongweek
        pred_2 = self.task_2_fc(x)[:, 0, :] # (batch_size, 1) # rpt_orders_per_cust
        pred_3 = self.task_3_fc(x)[:, 0, :] # (batch_size, 1) # aov
        
        return pred_1, pred_2, pred_3


    def reset_parameters(self):
        # Loop over all modules and reset parameters if the module has the `reset_parameters` method
        for module in self.modules():
            if module != self and hasattr(module, 'reset_parameters'):
                module.reset_parameters()









def calculate_MTLloss_3tasks(model, target_input, cov_input, ground_truth, 
                             individual_loss_criterion, weights, device, smape=False):
    # forward pass data through the model
    input_data = torch.cat((target_input, cov_input), dim=-1).to(device)
    predictions = model(input_data) # comes out as tuple of tensors: (batch_size, 1) each ; acq_hat, rptorderpc_hat, aov_hat

    predictions = torch.stack(predictions, dim=1).squeeze() # shape: (batch_size, 3)
    squared_errors = individual_loss_criterion(predictions, ground_truth.to(device))
    weighted_errors = squared_errors * weights
    
    if smape:
        smape_loss = smape_loss_criterion(predictions, ground_truth.to(device))
        return weighted_errors.mean(), smape_loss # mean of the batch
    else:
        return weighted_errors.mean()



def calculate_MTLloss_4tasks(model, target_input, cov_input, ground_truth, 
                             max_scaler_values, min_scaler_values, TASK_FEATURE_NAMES,
                             individual_loss_criterion, weights, device, smape=False):
    ## IMPORTANT to use only first 3 targets from 'target_input' (acq, ropc, aov)
    target_input_3tasks = target_input[:,:,:3]    
    # forward pass data through the model
    input_data = torch.cat((target_input_3tasks, cov_input), dim=-1).to(device)
    predictions = model(input_data) # comes out as tuple of tensors: (batch_size, 1) each ; acq_hat, rptorderpc_hat, aov_hat
    predictions = torch.stack(predictions, dim=1).squeeze() # shape: (batch_size, 3)

    ######## compute pred_4 sale ########            
    pred1_unscaled = predictions[:,0].reshape(-1,1) * (max_scaler_values[TASK_FEATURE_NAMES[0]] - min_scaler_values[TASK_FEATURE_NAMES[0]]) + min_scaler_values[TASK_FEATURE_NAMES[0]]
    pred2_unscaled = predictions[:,1].reshape(-1,1) * (max_scaler_values[TASK_FEATURE_NAMES[1]] - min_scaler_values[TASK_FEATURE_NAMES[1]]) + min_scaler_values[TASK_FEATURE_NAMES[1]]
    pred3_unscaled = predictions[:,2].reshape(-1,1) * (max_scaler_values[TASK_FEATURE_NAMES[2]] - min_scaler_values[TASK_FEATURE_NAMES[2]]) + min_scaler_values[TASK_FEATURE_NAMES[2]]
    pred4_unscaled = pred1_unscaled * pred2_unscaled * pred3_unscaled # pseudo spend = pseudo acq * rptorderpc * aov regularization binding
    pred_4 = (pred4_unscaled - min_scaler_values[TASK_FEATURE_NAMES[3]])/(max_scaler_values[TASK_FEATURE_NAMES[3]] - min_scaler_values[TASK_FEATURE_NAMES[3]])
    predictions_4tasks = torch.hstack([predictions, pred_4])

    squared_errors = individual_loss_criterion(predictions_4tasks, ground_truth.to(device))
    weighted_errors = squared_errors * weights
    
    if smape:
        smape_loss = smape_loss_criterion(predictions_4tasks, ground_truth.to(device))
        return weighted_errors.mean(), smape_loss # mean of the batch
    else:
        return weighted_errors.mean()





def smape_loss_criterion(y_pred, y_true):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) 
    """
    # Avoid division by zero
    epsilon = 1e-8
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
    
    # Calculate sMAPE
    loss = torch.abs(y_true - y_pred) / denominator
    
    # Handle the case where both y_true and y_pred are zero
    loss[denominator == epsilon] = 0.0
    
    return torch.mean(loss)

#




class STL_Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int, # input (target + covariates) dimension
        feature_dict: dict,
        d_model: int = hyperparam.D_MODEL, # input embedding dimension. number of features in the Transformer encoder/decoder inputs
        nhead: int = hyperparam.N_HEAD, # the number of heads in the multihead attention mechanism
        num_encoder_layers: int = hyperparam.N_ENCODER_LAYERS, # the number of layers in Transformer encoder
        num_decoder_layers: int = hyperparam.N_DECODER_LAYERS, # the number of layers in Transformer decoder
        d_feedforward: int = hyperparam.D_FEEDFORWARD, # the dimension of the feedforward network in Transformer encoder/decoder
        dropout: float = hyperparam.DROPOUT, # the dropout probability
        activation: str = hyperparam.ACTIVATION,  # the activation function
        input_chunk_length: int = hyperparam.INPUT_CHUNK_LENGTH, # sequence length
        output_chunk_length: int = hyperparam.OUTPUT_CHUNK_LENGTH, # =1, single horizon prediction
        num_merchant: int = None, # number of coompanies
    ):
        super().__init__()
        
        # set parameters
        self.input_dim = input_dim # input (target + covariates) dimension
        self.tgt_dim = input_dim - len(feature_dict.keys())
        self.feature_dict = feature_dict
        self.input_chunk_length = input_chunk_length # input sequence length
        self.output_chunk_length = output_chunk_length # output sequence length
        self.d_model = d_model
                
        total_input_dim = input_dim
        # Mapping from input size to d_model size
        if 'merchant_emb_int' in feature_dict.keys():
            assert num_merchant is not None
            self.merchant_emb = nn.Embedding(num_embeddings=num_merchant, embedding_dim=hyperparam.MERCHANT_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.MERCHANT_EMB_DIM - 1

        if 'week_int' in feature_dict.keys():
            self.week_emb = nn.Embedding(num_embeddings=53, embedding_dim=hyperparam.WEEK_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.WEEK_EMB_DIM - 1

        if 'cohort_month_int' in feature_dict.keys():
            self.cohort_month_emb = nn.Embedding(num_embeddings=13, embedding_dim=hyperparam.COHORT_MONTH_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.COHORT_MONTH_EMB_DIM - 1

        if 'group_censored_int' in feature_dict.keys():
            self.group_censored_emb = nn.Embedding(num_embeddings=101, embedding_dim=hyperparam.GROUP_CENSORED_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.GROUP_CENSORED_EMB_DIM - 1
                
        self.input_embedding = nn.Linear(total_input_dim, d_model) 

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(d_model, dropout, input_chunk_length)

        # initialize a transformer model using the nn.Transformer class
        self.transformer = nn.Transformer(
            d_model=d_model, # number of features in the encoder/decoder inputs
            nhead=nhead,  # the number of heads in the multihead attention mechanism
            num_encoder_layers=num_encoder_layers,  # the number of layers in the encoder
            num_decoder_layers=num_decoder_layers,  # the number of layers in the decoder
            dim_feedforward=d_feedforward,  # the dimension of the feedforward network
            dropout=dropout,  # the dropout probability
            activation=activation,  # the activation function
            batch_first=True,  # if True, (batch, seq, feature). if False, (seq, batch, feature).
        )

        # Mapping from d_model size to task size * output_chunk_length
        self.linear = nn.Linear(d_model, output_chunk_length)



    def forward(self, src_input):
        ## src: a sequence of features of shape (batch_size, input_chunk_length, input_dim)
        ## tgt: a sequence of features of shape (batch_size, output_chunk_length=1, input_dim)
        
        tgt_input = src_input[:,:,:self.tgt_dim]
        cov_input = src_input[:,:,self.tgt_dim:]

        # non-embedding features
        non_embed_covariate_keys = set(self.feature_dict.keys()) - set(['merchant_emb_int','week_int','cohort_month_int','group_censored_int'])
        remaining_feature_idx = [self.feature_dict[key] for key in non_embed_covariate_keys]
        remaining_feature_idx.sort()
        extended_src_input = torch.cat((tgt_input, cov_input[:,:,remaining_feature_idx]), dim=-1)

        # embedding        
        if 'merchant_emb_int' in self.feature_dict.keys():
            # Convert src_input to a LongTensor and send it to the same device as the model
            src_input_merchant = cov_input[:,:,self.feature_dict['merchant_emb_int']].long().to(self.merchant_emb.weight.device)
            merchant_embedding = self.merchant_emb(src_input_merchant)
            extended_src_input = torch.cat((extended_src_input, merchant_embedding), dim=-1)
        if 'week_int' in self.feature_dict.keys():
            src_input_week = cov_input[:,:,self.feature_dict['week_int']].long().to(self.week_emb.weight.device)
            week_embedding = self.week_emb(src_input_week)
            extended_src_input = torch.cat((extended_src_input, week_embedding), dim=-1)
        if 'cohort_month_int' in self.feature_dict.keys():
            src_input_cohort_month = cov_input[:,:,self.feature_dict['cohort_month_int']].long().to(self.cohort_month_emb.weight.device)
            cohort_month_embedding = self.cohort_month_emb(src_input_cohort_month)
            extended_src_input = torch.cat((extended_src_input, cohort_month_embedding), dim=-1)
        if 'group_censored_int' in self.feature_dict.keys():
            src_input_group_censored = cov_input[:,:,self.feature_dict['group_censored_int']].long().to(self.group_censored_emb.weight.device)
            group_censored_embedding = self.group_censored_emb(src_input_group_censored)
            extended_src_input = torch.cat((extended_src_input, group_censored_embedding), dim=-1)

        # Get the last value of src and use it as target input
        tgt = extended_src_input[:, -1, :] # shape: (batch_size, input_dim)
        tgt = tgt.unsqueeze(1) # unsqueezes it to match the shape of the target tensor. shape: (batch_size, 1, input_dim)

        # Apply the encoder layer and positional encoding to the source input
        src = self.input_embedding(extended_src_input) * math.sqrt(self.d_model)
        src = self.positional_encoding(src) # (batch_size, input_chunk_length, d_model)

        # Apply the encoder layer and positional encoding to the target input
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt) # (batch_size, 1, d_model)

        # Pass the source and target input through the transformer
        x = self.transformer(src=src, tgt=tgt) # (batch_size, 1, d_model)
        
        # Pass the output through the decoder
        out = self.linear(x) # (batch_size, 1, 1) 

        # Get the predictions
        predictions = out[:, 0, :] # (batch_size, 1)
        
        return predictions




















class STL_Transformer_old(nn.Module):
    def __init__(
        self,
        input_dim: int, # input (target + covariates) dimension
        d_model: int, # input embedding dimension. number of features in the Transformer encoder/decoder inputs
        nhead: int, # the number of heads in the multihead attention mechanism
        num_encoder_layers: int, # the number of layers in Transformer encoder
        num_decoder_layers: int, # the number of layers in Transformer decoder
        d_feedforward: int, # the dimension of the feedforward network in Transformer encoder/decoder
        dropout: float, # the dropout probability
        activation: str,  # the activation function
        input_chunk_length: int, # sequence length
        output_chunk_length: int, # =1, single horizon prediction
        use_merchant_embedding: bool = False, # embedding for merchant (pooled model)
        num_merchant: int = None, # number of coompanies
        merchant_emb_dim: int = None, # company embedding dimension
    ):
        super().__init__()
        
        # set parameters
        self.input_dim = input_dim # input (target + covariates) dimension
        self.input_chunk_length = input_chunk_length # input sequence length
        self.output_chunk_length = output_chunk_length # output sequence length
        self.d_model = d_model
        
        # Mapping from input size to d_model size
        self.use_merchant_embedding = use_merchant_embedding
        if self.use_merchant_embedding:
            assert num_merchant is not None and merchant_emb_dim is not None
            self.merchant_emb = nn.Embedding(num_embeddings=num_merchant, embedding_dim=merchant_emb_dim) # entity embedding
            self.input_embedding = nn.Linear(input_dim - 1 + merchant_emb_dim, d_model) # -1 because last feature is group embedding index
        else:
            self.input_embedding = nn.Linear(input_dim, d_model) # Mapping from input size to d_model size

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(d_model, dropout, input_chunk_length)

        # initialize a transformer model using the nn.Transformer class
        self.transformer = nn.Transformer(
            d_model=d_model, # number of features in the encoder/decoder inputs
            nhead=nhead,  # the number of heads in the multihead attention mechanism
            num_encoder_layers=num_encoder_layers,  # the number of layers in the encoder
            num_decoder_layers=num_decoder_layers,  # the number of layers in the decoder
            dim_feedforward=d_feedforward,  # the dimension of the feedforward network
            dropout=dropout,  # the dropout probability
            activation=activation,  # the activation function
            batch_first=True,  # if True, (batch, seq, feature). if False, (seq, batch, feature).
        )

        # Mapping from d_model size to task size * output_chunk_length
        self.linear = nn.Linear(d_model, output_chunk_length)



    def forward(self, src_input):
        ## src: a sequence of features of shape (batch_size, input_chunk_length, input_dim)
        ## tgt: a sequence of features of shape (batch_size, output_chunk_length=1, input_dim)
        
        # merchant embedding     
        if self.use_merchant_embedding:
            ###### merchant embedding #####       
            # Convert src_input to a LongTensor and send it to the same device as the model
            src_input_last_dim = src_input[:,:,-1].long().to(self.merchant_emb.weight.device)
            merchant_embedding = self.merchant_emb(src_input_last_dim)
            src_input = torch.cat((src_input[:,:,:-1], merchant_embedding), dim=-1)
        else:
            pass
        
        # Get the last value of src and use it as target input
        tgt = src_input[:, -1, :] # shape: (batch_size, input_dim)
        tgt = tgt.unsqueeze(1) # unsqueezes it to match the shape of the target tensor. shape: (batch_size, 1, input_dim)

        # Apply the encoder layer and positional encoding to the source input
        print('\n\n\nsrc_input.shape', src_input.shape)
        src = self.input_embedding(src_input) * math.sqrt(self.d_model)
        src = self.positional_encoding(src) # (batch_size, input_chunk_length, d_model)
        print('src.shape', src.shape)

        print('tgt.shape', tgt.shape)
        # Apply the encoder layer and positional encoding to the target input
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt) # (batch_size, 1, d_model)
        print('tgt.shape', tgt.shape)
        # Pass the source and target input through the transformer
        x = self.transformer(src=src, tgt=tgt) # (batch_size, 1, d_model)
        
        # Pass the output through the decoder
        out = self.linear(x) # (batch_size, 1, 1) 

        # Get the predictions
        predictions = out[:, 0, :] # (batch_size, 1)
        
        return predictions











class MTL_Transformer_old(nn.Module):
    def __init__(
        self,
        input_dim: int, # input (target + covariates) dimension
        feature_dict: dict,
        d_model: int = hyperparam.D_MODEL, # input embedding dimension. number of features in the Transformer encoder/decoder inputs
        nhead: int = hyperparam.N_HEAD, # the number of heads in the multihead attention mechanism
        num_encoder_layers: int = hyperparam.N_ENCODER_LAYERS, # the number of layers in Transformer encoder
        num_decoder_layers: int = hyperparam.N_DECODER_LAYERS, # the number of layers in Transformer decoder
        d_feedforward: int = hyperparam.D_FEEDFORWARD, # the dimension of the feedforward network in Transformer encoder/decoder
        d_feedforward_task: int = hyperparam.D_FEEDFORWARD_TASK, # the dimension of the feedforward network in the task-specific decoder
        dropout: float = hyperparam.DROPOUT, # the dropout probability
        activation: str = hyperparam.ACTIVATION,  # the activation function
        input_chunk_length: int = hyperparam.INPUT_CHUNK_LENGTH, # sequence length
        output_chunk_length: int = hyperparam.OUTPUT_CHUNK_LENGTH, # =1, single horizon prediction
        num_merchant: int = None, # number of coompanies
        use_single_task_specific_layer: bool = False,
    ):
        super().__init__()

        # set parameters
        self.input_dim = input_dim 
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.d_model = d_model
        self.use_merchant_embedding = use_merchant_embedding
        self.use_single_task_specific_layer = use_single_task_specific_layer
        
        if self.use_merchant_embedding:
            assert num_merchant is not None and merchant_emb_dim is not None
            self.merchant_emb = nn.Embedding(num_embeddings=num_merchant, embedding_dim=merchant_emb_dim)
            self.input_embedding = nn.Linear(input_dim - 1 - 1 + merchant_emb_dim, d_model) # -1 because last covariate feature is group embedding index, -1 because last target feature is repeat spend
        else:
            self.input_embedding = nn.Linear(input_dim, d_model) # Mapping from input size to d_model size

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(d_model, dropout, input_chunk_length)
        
        # initialize a transformer model using the nn.Transformer class
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead,  
            num_encoder_layers=num_encoder_layers,  
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=d_feedforward,  
            dropout=dropout,  
            activation=activation,
            batch_first=True,  # if True, (batch, seq, feature). if False, (seq, batch, feature).
        )

        # 3-layers FFNN: Mapping from d_model size to output_chunk_length=1
        self.task_1_fc = nn.Sequential(
            nn.Linear(d_model, d_feedforward_task), nn.ReLU(), 
            nn.Linear(d_feedforward_task, d_feedforward_task//8), nn.ReLU(), 
            nn.Linear(d_feedforward_task//8, output_chunk_length))

        self.task_2_fc = nn.Sequential(
            nn.Linear(d_model, d_feedforward_task), nn.ReLU(), 
            nn.Linear(d_feedforward_task, d_feedforward_task//8), nn.ReLU(), 
            nn.Linear(d_feedforward_task//8, output_chunk_length))
        
        self.task_3_fc = nn.Sequential(
            nn.Linear(d_model, d_feedforward_task), nn.ReLU(), 
            nn.Linear(d_feedforward_task, d_feedforward_task//8), nn.ReLU(), 
            nn.Linear(d_feedforward_task//8, output_chunk_length))



    def forward(self, src_input):
        ## src: a sequence of features of shape (batch_size, input_chunk_length, input_dim)
        ## tgt: a sequence of features of shape (batch_size, output_chunk_length=1, input_dim)

        if self.use_merchant_embedding:
            ###### merchant embedding #####       
            # Convert src_input to a LongTensor and send it to the same device as the model
            src_input_last_dim = src_input[:,:,-1].long().to(self.merchant_emb.weight.device)
            merchant_embedding = self.merchant_emb(src_input_last_dim)
            src_input = torch.cat((src_input[:,:,:-1], merchant_embedding), dim=-1)
        else:
            pass
        
        # Get the last value of src and use it as target input
        tgt = src_input[:, -1, :] # shape: (batch_size, input_dim)
        tgt = tgt.unsqueeze(1) # unsqueezes it to match the shape of the target tensor. shape: (batch_size, 1, input_dim)
        
        # Apply the encoder layer and positional encoding to the source input
        src = self.input_embedding(src_input) * math.sqrt(self.d_model) # common scaling factor
        src = self.positional_encoding(src) # (batch_size, input_chunk_length, d_model)

        # Apply the encoder layer and positional encoding to the target input
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt) # (batch_size, 1, d_model)
   
        # Pass the source and target input through the transformer
        x = self.transformer(src=src, tgt=tgt) # (batch_size, 1, d_model)

        if self.use_single_task_specific_layer:
            # Pass the output through the decoder
            pred_1 = self.task_1_fc(x)[:, 0, :] # (batch_size, 1) # N_week_cohort_alongweek
            return pred_1
        else:
            # Pass the output through the decoder
            pred_1 = self.task_1_fc(x)[:, 0, :] # (batch_size, 1) # N_week_cohort_alongweek
            pred_2 = self.task_2_fc(x)[:, 0, :] # (batch_size, 1) # rpt_orders_per_cust
            pred_3 = self.task_3_fc(x)[:, 0, :] # (batch_size, 1) # aov
            return pred_1, pred_2, pred_3








class MTL_Transformer_new(nn.Module):
    def __init__(
        self,
        input_dim: int, # input (target + covariates) dimension
        feature_dict: dict,
        d_model: int = hyperparam.D_MODEL, # input embedding dimension. number of features in the Transformer encoder/decoder inputs
        nhead: int = hyperparam.N_HEAD, # the number of heads in the multihead attention mechanism
        num_encoder_layers: int = hyperparam.N_ENCODER_LAYERS, # the number of layers in Transformer encoder
        num_decoder_layers: int = hyperparam.N_DECODER_LAYERS, # the number of layers in Transformer decoder
        d_feedforward: int = hyperparam.D_FEEDFORWARD, # the dimension of the feedforward network in Transformer encoder/decoder
        d_feedforward_task: int = hyperparam.D_FEEDFORWARD_TASK, # the dimension of the feedforward network in the task-specific decoder
        dropout: float = hyperparam.DROPOUT, # the dropout probability
        activation: str = hyperparam.ACTIVATION,  # the activation function
        input_chunk_length: int = hyperparam.INPUT_CHUNK_LENGTH, # sequence length
        output_chunk_length: int = hyperparam.OUTPUT_CHUNK_LENGTH, # =1, single horizon prediction
        num_merchant: int = None, # number of coompanies
    ):
        super().__init__()

       # set parameters
        self.input_dim = input_dim # input (target + covariates) dimension
        self.tgt_dim = input_dim - len(feature_dict.keys())
        self.feature_dict = feature_dict
        self.input_chunk_length = input_chunk_length # input sequence length
        self.output_chunk_length = output_chunk_length # output sequence length
        self.d_model = d_model
                
        total_input_dim = input_dim
        # Mapping from input size to d_model size
        if 'merchant_emb_int' in feature_dict.keys():
            assert num_merchant is not None
            self.merchant_emb = nn.Embedding(num_embeddings=num_merchant, embedding_dim=hyperparam.MERCHANT_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.MERCHANT_EMB_DIM - 1

        if 'week_int' in feature_dict.keys():
            self.week_emb = nn.Embedding(num_embeddings=53, embedding_dim=hyperparam.WEEK_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.WEEK_EMB_DIM - 1

        if 'cohort_month_int' in feature_dict.keys():
            self.cohort_month_emb = nn.Embedding(num_embeddings=13, embedding_dim=hyperparam.COHORT_MONTH_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.COHORT_MONTH_EMB_DIM - 1

        if 'group_censored_int' in feature_dict.keys():
            self.group_censored_emb = nn.Embedding(num_embeddings=hyperparam.COHORT_EMB_NUM + 1, embedding_dim=hyperparam.GROUP_CENSORED_EMB_DIM) # entity embedding
            total_input_dim += hyperparam.GROUP_CENSORED_EMB_DIM - 1
                
        self.input_embedding = nn.Linear(total_input_dim, d_model) 

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(d_model, dropout, input_chunk_length)

        # initialize a transformer model using the nn.Transformer class
        self.transformer = nn.Transformer(
            d_model=d_model, # number of features in the encoder/decoder inputs
            nhead=nhead,  # the number of heads in the multihead attention mechanism
            num_encoder_layers=num_encoder_layers,  # the number of layers in the encoder
            num_decoder_layers=num_decoder_layers,  # the number of layers in the decoder
            dim_feedforward=d_feedforward,  # the dimension of the feedforward network
            dropout=dropout,  # the dropout probability
            activation=activation,  # the activation function
            batch_first=True,  # if True, (batch, seq, feature). if False, (seq, batch, feature).
        )

        def create_task_fc(d_model, d_feedforward_task, output_chunk_length):
            return nn.Sequential(
                nn.Linear(d_model, d_feedforward_task), nn.ReLU(), 
                nn.Linear(d_feedforward_task, d_feedforward_task//2), nn.ReLU(), 
                nn.Linear(d_feedforward_task//2, output_chunk_length))

        # Then you can create each task like this:
        self.task_1_fc = create_task_fc(d_model+3, d_feedforward_task, output_chunk_length)
        self.task_2_fc = create_task_fc(d_model+3, d_feedforward_task, output_chunk_length)
        self.task_3_fc = create_task_fc(d_model, d_feedforward_task, output_chunk_length)



    def forward(self, src_input):
        ## src: a sequence of features of shape (batch_size, input_chunk_length, input_dim)
        ## tgt: a sequence of features of shape (batch_size, output_chunk_length=1, input_dim)

        tgt_input = src_input[:,:,:self.tgt_dim]
        cov_input = src_input[:,:,self.tgt_dim:]

        # non-embedding features
        non_embed_covariate_keys = set(self.feature_dict.keys()) - set(['merchant_emb_int','week_int','cohort_month_int','group_censored_int'])
        remaining_feature_idx = [self.feature_dict[key] for key in non_embed_covariate_keys]
        remaining_feature_idx.sort()
        extended_src_input = torch.cat((tgt_input, cov_input[:,:,remaining_feature_idx]), dim=-1)
        
        global_feature_idx = [self.feature_dict[key] for key in ['holidays_1w_ahead', 'linear_trend', 'quadratic_trend']]
        
        # embedding        
        if 'merchant_emb_int' in self.feature_dict.keys():
            # Convert src_input to a LongTensor and send it to the same device as the model
            src_input_merchant = cov_input[:,:,self.feature_dict['merchant_emb_int']].long().to(self.merchant_emb.weight.device)
            merchant_embedding = self.merchant_emb(src_input_merchant)
            extended_src_input = torch.cat((extended_src_input, merchant_embedding), dim=-1)
        if 'week_int' in self.feature_dict.keys():
            src_input_week = cov_input[:,:,self.feature_dict['week_int']].long().to(self.week_emb.weight.device)
            week_embedding = self.week_emb(src_input_week)
            extended_src_input = torch.cat((extended_src_input, week_embedding), dim=-1)
        if 'cohort_month_int' in self.feature_dict.keys():
            src_input_cohort_month = cov_input[:,:,self.feature_dict['cohort_month_int']].long().to(self.cohort_month_emb.weight.device)
            cohort_month_embedding = self.cohort_month_emb(src_input_cohort_month)
            extended_src_input = torch.cat((extended_src_input, cohort_month_embedding), dim=-1)
        if 'group_censored_int' in self.feature_dict.keys():
            src_input_group_censored = cov_input[:,:,self.feature_dict['group_censored_int']].long().to(self.group_censored_emb.weight.device)
            group_censored_embedding = self.group_censored_emb(src_input_group_censored)
            extended_src_input = torch.cat((extended_src_input, group_censored_embedding), dim=-1)

        # Get the last value of src and use it as target input
        tgt = extended_src_input[:, -1, :] # shape: (batch_size, input_dim)
        tgt = tgt.unsqueeze(1) # unsqueezes it to match the shape of the target tensor. shape: (batch_size, 1, input_dim)
 
        # Apply the encoder layer and positional encoding to the source input
        src = self.input_embedding(extended_src_input) * math.sqrt(self.d_model) # common scaling factor
        src = self.positional_encoding(src) # (batch_size, input_chunk_length, d_model)

        # Apply the encoder layer and positional encoding to the target input
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt) # (batch_size, 1, d_model)
   
        # Pass the source and target input through the transformer
        x = self.transformer(src=src, tgt=tgt) # (batch_size, 1, d_model)

        # Pass the output through the decoder
        pred_1 = self.task_1_fc(torch.cat( (x,cov_input[:,:,global_feature_idx]), dim=-1))[:, 0, :] # (batch_size, 1) # N_week_cohort_alongweek
        pred_2 = self.task_2_fc(torch.cat( (x,cov_input[:,:,global_feature_idx]), dim=-1))[:, 0, :] # (batch_size, 1) # rpt_orders_per_cust
        pred_3 = self.task_3_fc(x)[:, 0, :] # (batch_size, 1) # aov
        
        return pred_1, pred_2, pred_3

