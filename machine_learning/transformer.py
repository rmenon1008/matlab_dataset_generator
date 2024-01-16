## Based on influenza transformer from this repo https://github.com/KasperGroesLudvigsen/influenza_transformer

import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import datetime
from models import TimeSeriesTransformer
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from utils import calculate_metrics

from dataset_consumer import DatasetConsumer

from utils import watts_to_dbm, get_scaler, dbm_to_watts

from cprint import *

DEBUG = True
SCALER = 'minmax'
SAVE_PATH = './machine_learning/models/model.pth'
NUM_PATHS = 100

# Hyperparams
test_size = 0.1
batch_size = 128
shuffle = True

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
# exogenous_vars = [] # should contain strings. Each string must correspond to a column name
# input_variables = [target_col_name] + exogenous_vars
# target_idx = 0 # index position of target in batched trg_y

# input_size = len(input_variables)
input_variables = 1 # Univariate

DATASET = 'dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
d.print_info()

# Scale mag data
d.csi_mags = watts_to_dbm(d.csi_mags) # Convert to dBm
scaler = get_scaler('minmax')
scaler.fit(d.csi_mags.T)
d.csi_mags = d.scale(scaler.transform, d.csi_mags.T).T

# Find paths
d.csi_phases = d.unwrap(d.csi_phases)
paths = d.generate_straight_paths(NUM_PATHS, 10)
dataset_mag = d.paths_to_dataset_mag_only(paths)
dataset_phase = d.paths_to_dataset_phase_only(paths)
dataset_positions = d.paths_to_dataset_positions(paths)

# # Convert 'split_sequences' to a PyTorch tensor
dataset_mag = torch.from_numpy(dataset_mag)
# Split dataset into train, val and test
X_train, X_test, y_train, y_test = train_test_split(dataset_mag[:,:9,0], dataset_mag[:,9:10,0].squeeze(), train_size = 0.85, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, shuffle=False)

# Dataset
train = TensorDataset(X_train, y_train)
validate = TensorDataset(X_val, y_val)
test = TensorDataset(X_test, y_test)

# train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
# validate_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
# test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

# Making dataloader
training_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

i, batch = next(enumerate(training_data))

src, trg, trg_y = batch
cprint.ok("src shape: {}".format(src.shape))
cprint.ok("trg shape: {}".format(trg.shape))
cprint.ok("trg_y shape: {}".format(trg_y.shape))
# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
if batch_first == False:

    shape_before = src.shape
    src = src.permute(1, 0, 2)
    print("src shape changed from {} to {}".format(shape_before, src.shape))

    shape_before = trg.shape
    trg = trg.permute(1, 0, 2)
    print("src shape changed from {} to {}".format(shape_before, src.shape))

model = TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    )

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    )

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=output_sequence_length,
    dim2=output_sequence_length
    )

output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )

