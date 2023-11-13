import datetime
import h5py
import numpy as np
from models import RNN, LSTM, GRU
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

from dataset_consumer import DatasetConsumer

np.set_printoptions(threshold=sys.maxsize)

# Specify the path to your HDF5 file
if platform.system() == 'Windows':
    hdf5_file_path = '..\dataset_0_5m_spacing.h5'
else:
    hdf5_file_path = '../dataset_0_5m_spacing.h5'

DEBUG = False
SCALE_DATASET = True
SAVE_PATH = './machine_learning/models/model.pth'

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
# with h5py.File(hdf5_file_path, 'r') as file:
#     # Access the dataset you want to read
#     csis_mag = file['csis_mag']  # Replace 'your_dataset_name' with the actual dataset name
#     positions = file['positions']
#     # Read the data into a NumPy array
#     csis_mag = csis_mag[:]
#     positions = positions[:]
# cprint.info(csis_mag.shape)


DATASET = 'dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
d.print_info()
scaler = get_scaler('minmax')
# d.csi_mags = d.scale(scaler.fit_transform, d.csi_mags.T).T
d.csi_mags = d.scale(scaler.fit_transform, d.csi_mags)
cprint.info(f'd.csi_mags.shape {d.csi_mags.shape}')
d.csi_phases = d.unwrap(d.csi_phases)
paths = d.generate_straight_paths(200, 10)
dataset_mag = d.paths_to_dataset_mag_only(paths)
dataset_phase = d.paths_to_dataset_phase_only(paths)
dataset_positions = d.paths_to_dataset_positions(paths)
cprint(f"dataset_positions.shape {dataset_positions.shape}")

for i in range(10):
    rand = np.random.randint(dataset_positions.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset_positions[rand,0,:], dataset_positions[rand,1,:], dataset_positions[rand,2,:])
    plt.show()

# # if DEBUG:
# #     for i in range(1):
# #         rand1 = np.random.randint(dataset_mag.shape[0])
# #         rand2 = np.random.randint(dataset_mag.shape[1])
# #         # cprint.info(f'position {positions[:,790]}')
# #         plt.figure(1)
# #         # plt.plot(positions[:1,:10], positions[1:2,:10],'.', color='b')
# #         plt.subplot(211)
# #         plt.plot(np.arange(dataset_mag.shape[2]), dataset_mag[rand1][rand2])
# #         plt.subplot(212)
# #         plt.plot(np.arange(dataset_phase.shape[2]), dataset_phase[rand1][rand2])
# #         plt.show()

# # if SCALE_DATASET:
# #     # cprint.info(csis_mag)
# #     # Scale dataset
# #     scaler = get_scaler('minmax')
# #     csis_mag_scaled = scaler.fit_transform(csis_mag)
# #     # cprint(csis_mag_scaled[:,17:23])

# #     # Convert the NumPy array to a PyTorch tensor
# #     csis_mag_scaled = torch.from_numpy(csis_mag_scaled)
# #     positions = torch.from_numpy(positions)

#     # for i in range(10):
#     #     rand = np.random.randint(csis_mag_scaled.shape[1])
#     #     # cprint.info(f'position {positions[:,790]}')
#     #     plt.figure(1)
#     #     # plt.plot(positions[:1,:10], positions[1:2,:10],'.', color='b')
#     #     plt.subplot(211)
#     #     plt.plot(np.arange(csis_mag_scaled.shape[0]), csis_mag_scaled[:,rand:rand + 1])
#     #     plt.subplot(212)
#     #     plt.plot(np.arange(csis_mag.shape[0]), csis_mag[:,rand:rand + 1])
#     #     plt.show()

# # # Define the sequence length
# # sequence_length = 10
# # grid_length = int(np.sqrt(csis_mag_scaled.size(1)))

# # # Calculate the number of sequences that can be obtained
# # num_sequences = csis_mag_scaled.size(1) // sequence_length

# # # Create a list to store the split sequences
# # split_sequences = []

# # # Loop through the input tensor and split it into sequences
# # for i in range(num_sequences):
# #     if i % grid_length < grid_length - sequence_length:
# #         start = i * sequence_length
# #         end = (i + 1) * sequence_length
# #         sequence = csis_mag_scaled[:, start:end]
# #         split_sequences.append(sequence)

# # # Convert 'split_sequences' to a PyTorch tensor
# dataset_mag = torch.from_numpy(dataset_mag)
# # Split dataset into train, val and test
# X_train, X_test, y_train, y_test = train_test_split(dataset_mag[:,:9,:], dataset_mag[:,9:10,:].squeeze(), train_size = 0.9, shuffle=False)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, shuffle=False)

# train = TensorDataset(X_train, y_train)
# validate = TensorDataset(X_val, y_val)
# test = TensorDataset(X_test, y_test)
# # Define hyperparameters
# batch_size = 100
# shuffle = True

# # Create a data loader
# train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
# validate_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
# test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

# # Loop through the data using the data loader
# for i, batch in enumerate(train_dataloader, 0):
#     # 'batch' will contain a batch of 10-element sequences
#     # cprint.warn(type(batch))
#     # cprint.info(batch[0].shape)
#     # cprint.info(batch[1].shape)
#     cprint(f'Batch {i} has size {batch[0].shape[0]}')  # Check the batch size


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
# num_epochs = 2000
# batch_size = batch_size
# model_type = "rnn"
# model_params = {'input_size': input_size,
#                 'hidden_size' : hidden_size,
#                 'num_layers' : num_layers,
#                 'output_size' : output_size,
#                 'dropout_prob' : dropout}

# current = datetime.datetime.now()
# writer = SummaryWriter(f"runs/test{model_type}_{num_epochs}_{num_layers}_{learning_rate}_{dropout}_{current.month}-{current.day}-{current.hour}-{current.minute}")

# # Create a simple RNN model
# model = get_model(model_type, model_params)

# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# model = model.float()
# i = 0
# for epoch in range(num_epochs):
#     running_train_loss = 0.0
#     running_val_loss = 0

#     for batch in train_dataloader:
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

#         running_train_loss += loss.item()
#         writer.add_scalar("losses/running_train_loss", loss.item(), i)
#         i += 1

#     model.eval()
#     with torch.no_grad():
#         for batch in validate_dataloader:
#             sequences, targets = batch
#             outputs = model(sequences.float())
#             val_loss = criterion(outputs, targets.float()).item()
#             running_val_loss += val_loss
#             writer.add_scalar("losses/running_val_loss", val_loss, i)


#     if epoch % 100 == 99:
#         print(f'[{epoch + 1}, {num_epochs}] loss: {running_train_loss:.3f}')
#         running_train_loss = 0.0

#         #TODO What is this?
#         # running_train_loss += loss.item()
#         # writer.add_scalar("losses/running_train_loss", loss.item(), i)
#         # i += 1

# # Save test tensors
# torch.save(X_test, f'./machine_learning/models/X_test_{model_type}.pt')
# torch.save(y_test, f'./machine_learning/models/y_test_{model_type}.pt')
# # # Run Test dataset
# # model.eval()

# # test_loss = 0.0  # Initialize the test loss
# # total_samples = 0  # Total samples in the test dataset

# # for i, batch in enumerate(test_dataloader, 0):  # Assuming you have a DataLoader for the test dataset
# #     sequences, targets = batch  # Get input sequences and their targets

# #     with torch.no_grad():
# #         outputs = model(sequences.float())  # Make predictions

# #     loss = criterion(outputs, targets)  # Calculate the loss
# #     writer.add_scalar("losses/running_test_loss", loss.item(), i)
# #     test_loss += loss.item()
# #     total_samples += 1

# # # Calculate the average test loss
# # average_test_loss = test_loss / total_samples

# # # Optionally, calculate and print other evaluation metrics if needed
# # print(f"Average Test Loss: {average_test_loss:.4f}")


# if DEBUG:
#     # Sanity check
#     for i in range(5):
#         # To use the trained model for prediction, you can pass new sequences to the model:
#         # new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]
#         rand = torch.randint(0, X_test.shape[0], (1,))
#         new_input = X_test[rand,:]
#         ground_truth = y_test[rand,:]
#         cprint.info(f'ground truth {ground_truth.shape}')
#         # Prediction
#         prediction = model(new_input.to(torch.float32))

#         # Graphs
#         fig, axs = plt.subplots(5)
#         new_input = new_input.squeeze()
#         # cprint.warn(f'validation shape {new_input[8,:].shape}')
#         axs[0].plot(new_input[6,:])
#         axs[0].set_title("CSI Reading 7")
#         axs[1].plot(new_input[7,:])
#         axs[1].set_title("CSI Reading 8")
#         axs[2].plot(new_input[8,:])
#         axs[2].set_title("CSI Reading 9")

#         prediction = prediction.detach().numpy()
#         # writer.add_figure(f'Comparison {i}', fig, global_step=0)
#         # plt.close(fig)
#         # plt.show()
        
#         # prediction = np.expand_dims(prediction, axis=1)
#         # prediction = scaler.inverse_transform(prediction)
#         # cprint.info(f'Prediction shape {prediction.shape}')
#         # cprint.info(f'Prediction shape {prediction}')
#         # cprint.info(f'Prediction {scaler.inverse_transform(prediction).shape}')
#         axs[3].plot(ground_truth.squeeze())
#         axs[3].set_title("Ground Truth")
#         axs[4].plot(prediction.squeeze())
#         axs[4].set_title("Prediction")

#         # writer.add_figure(f'Comparison {i}', fig, global_step=0)
#         # plt.close(fig)
#         plt.show()

# # Close tensorboard writer
# writer.close()
# # Save model
# torch.save(model.state_dict(), SAVE_PATH)


# # # NOTE Run "tensorboard --logdir runs" to see results