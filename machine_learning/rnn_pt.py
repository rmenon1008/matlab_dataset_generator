import datetime
import h5py
import numpy as np
import torch
import torch.nn as nn
from cprint import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import platform

np.set_printoptions(threshold=sys.maxsize)

# Specify the path to your HDF5 file
if platform.system() == 'Windows':
    hdf5_file_path = '..\dataset_0_5m_spacing.h5'
else:
    hdf5_file_path = '../dataset_0_5m_spacing.h5'

SCALE_DATASET = True

# Value scaling function for feeding into nn
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

# Open the HDF5 file for reading
with h5py.File(hdf5_file_path, 'r') as file:
    # Access the dataset you want to read
    csis_mag = file['csis_mag']  # Replace 'your_dataset_name' with the actual dataset name
    positions = file['positions']
    # Read the data into a NumPy array
    csis_mag = csis_mag[:]
    positions = positions[:]

if SCALE_DATASET:
    # cprint.info(csis_mag)
    # Scale dataset
    scaler = get_scaler('minmax')
    csis_mag_scaled = scaler.fit_transform(csis_mag)
    # cprint(csis_mag_scaled[:,17:23])

    # Convert the NumPy array to a PyTorch tensor
    csis_mag_scaled = torch.from_numpy(csis_mag_scaled)
    positions = torch.from_numpy(positions)

    # for i in range(10):
    #     rand = np.random.randint(csis_mag_scaled.shape[1])
    #     # cprint.info(f'position {positions[:,790]}')
    #     plt.figure(1)
    #     # plt.plot(positions[:1,:10], positions[1:2,:10],'.', color='b')
    #     plt.subplot(211)
    #     plt.plot(np.arange(csis_mag_scaled.shape[0]), csis_mag_scaled[:,rand:rand + 1])
    #     plt.subplot(212)
    #     plt.plot(np.arange(csis_mag.shape[0]), csis_mag[:,rand:rand + 1])
    #     plt.show()

# Define the sequence length
sequence_length = 10
grid_length = int(np.sqrt(csis_mag_scaled.size(1)))

# Calculate the number of sequences that can be obtained
num_sequences = csis_mag_scaled.size(1) // sequence_length

# Create a list to store the split sequences
split_sequences = []

# Loop through the input tensor and split it into sequences
for i in range(num_sequences):
    if i % grid_length < grid_length - sequence_length:
        start = i * sequence_length
        end = (i + 1) * sequence_length
        sequence = csis_mag_scaled[:, start:end]
        split_sequences.append(sequence)

# Convert 'split_sequences' to a PyTorch tensor if needed
split_sequences_tensor = torch.stack(split_sequences, dim=2).T
cprint.info(f'split_sequences_tensor {split_sequences_tensor.shape}')
# Split dataset into train, val and test
X_train, X_val, y_train, y_val = train_test_split(split_sequences_tensor[:,:9,:], split_sequences_tensor[:,9:10,:].squeeze(), train_size = 0.8, shuffle=False)

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is a list of 10-element sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        return sequence

# # Create a sample dataset (you should replace this with your own data)
# # Assuming 'split_sequences_tensor' from the previous example
# data = split_sequences_tensor.T

# # Initialize the custom dataset
# dataset = MyDataset(data)

train = TensorDataset(X_train, y_train)
cprint.warn(f'y_train size {y_train.shape}')
cprint.warn(f'X_train size {X_train.shape}')
# Define hyperparameters
batch_size = 100
shuffle = True

# Create a data loader
dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

# Loop through the data using the data loader
for i, batch in enumerate(dataloader, 0):
    # 'batch' will contain a batch of 10-element sequences
    # cprint.warn(type(batch))
    # cprint.info(batch[0].shape)
    # cprint.info(batch[1].shape)
    cprint(f'Batch {i} size {batch[0].shape[0]}')  # Check the batch size

# # Define a simple RNN class
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward pass through the RNN layer
#         out, _ = self.rnn(x, h0)
        
#         # Reshape the output for the fully connected layer
#         out = out[:, -1, :]
#         out = self.fc(out)
#         return out
    
# # Define a simple LSTM class
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward pass through the LSTM layer
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Reshape the output for the fully connected layer
#         out = self.fc(out[:, -1, :])
#         return out
    
# # Define a simple GRU class
# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
#         super(GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward pass through the GRU layer
#         out, _ = self.gru(x, h0)
        
#         # Reshape the output for the fully connected layer
#         out = self.fc(out[:, -1, :])
#         return out

# def get_model(model, model_params):
#     models = {
#         "rnn": RNN,
#         "lstm": LSTM,
#         "gru": GRU,
#     }
#     return models.get(model.lower())(**model_params)

# # Hyperparameters
# input_size = 128
# hidden_size = 64
# num_layers = 5
# output_size = 128
# sequence_length = 9
# learning_rate = 0.0001
# dropout = .2
# num_epochs = 1000
# batch_size = batch_size

# model_params = {'input_size': input_size,
#                 'hidden_size' : hidden_size,
#                 'num_layers' : num_layers,
#                 'output_size' : output_size,
#                 'dropout_prob' : dropout}
# current = datetime.datetime.now()
# writer = SummaryWriter(f"runs/test{num_epochs}_{num_layers}_{learning_rate}_{dropout}_{current.month}-{current.day}-{current.hour}-{current.minute}")
# # writer.add_text(
# #     "hyperparameters",
# #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),

# # Create a simple RNN model
# model = get_model("RNN", model_params)

# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# model = model.float()
# i = 0
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for batch in dataloader:
#         sequences, targets = batch
#         # cprint.warn(sequences.shape)
#         # cprint.warn(targets.shape)
#         outputs = model(sequences.float())
#         # cprint.warn(f'outputs {outputs.shape}')
#         # cprint.warn(f'targets {targets.shape}')
#         loss = criterion(outputs, targets.float())
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print statistics
#         running_loss += loss.item()
#         writer.add_scalar("losses/running_loss", loss.item(), i)
#         # if (epoch) % 100 == 99:
#         #     print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
#         if (i) % 100 == 99:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {num_epochs}] loss: {running_loss:.3f}')
#             running_loss = 0.0
#         i += 1

# for i in range(10):
#     # To use the trained model for prediction, you can pass new sequences to the model:
#     # new_input = torch.randn(1, sequence_length, input_size)
#     new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]

#     # Prediction
#     prediction = model(new_input.to(torch.float32))

#     # Graphs
#     fig, axs = plt.subplots(4)
#     new_input = new_input.squeeze()
#     cprint.warn(new_input[9,:].shape)
#     axs[0].plot(new_input[7,:])
#     axs[1].plot(new_input[8,:])
#     axs[2].plot(new_input[9,:])

#     prediction = prediction.detach().numpy()
#     cprint.info(prediction)
#     cprint.info(prediction.size)
#     axs[3].plot(prediction.squeeze())
#     axs[3].set_title("Prediction")
#     plt.show()

# writer.close()

# # NOTE Run "tensorboard --logdir runs" to see results