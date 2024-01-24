from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)

import os
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from cprint import *

def calculate_metrics(df):
    result_metrics = {'mae' : mean_absolute_error(df.value, df.prediction),
                      'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2' : r2_score(df.value, df.prediction)}
    return result_metrics

def watts_to_dbm(watts):
    epsilon = 1e-10  # Small constant to avoid log(0)
    dbm = 10 * np.log10(watts + epsilon) + 30
    return dbm

def dbm_to_watts(dbm):
    epsilon = 1e-10  # Small constant to avoid log(0)
    watts = 10 ** ((dbm - 30) / 10) #- epsilon
    return watts

# Value scaling function for feeding into nn
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
        "power_yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "power_box-cox": PowerTransformer(method="box-cox"),
        "quantiletransformer-uniform": QuantileTransformer(output_distribution="uniform", random_state=42),
        "quantiletransformer-gaussian": QuantileTransformer(output_distribution="normal", random_state=42),
    }
    return scalers.get(scaler.lower())

class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        data: torch.tensor,
        indices: list, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence. 
                     The sub-sequence is split into src, trg and trg_y later.  

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        
        super().__init__()

        self.indices = indices

        self.data = data

        print("From get_src_trg: data size = {}".format(data.size()))

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):
        
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        #print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y
    
    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 

        Args:

            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)

        Return: 

            src: tensor, 1D, used as input to the transformer model

            trg: tensor, 1D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input
        src = sequence[:enc_seq_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor, 
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=False
    ) -> torch.Tensor:

    """
    NB! This function is currently only tested on models that work with 
    batch_first = False
    
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  
    
    This function returns a prediction of length forecast_window for each batch in src
    
    NB! If you want the inference to be done without gradient calculation, 
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()
        
    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be 
    backpropagated during training and gradient calculation hence is required.
    
    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that 
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.
    
    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before 
               passed to this function.
               
        src: The input to the model
        
        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.
                           
        batch_size: batch size
        
        batch_first: If true, the shape of the model input should be 
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]
    
    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, 0] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            # device=device
            )

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
            # device=device
            )

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask) 

        # If statement simply makes sure that the predicted value is 
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step 
            # represented in tgt
            last_predicted_value = prediction[-1, :, :] 

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
    
    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        # device=device
        )

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        # device=device
        )

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction