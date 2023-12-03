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

from dataset_consumer import DatasetConsumer

np.set_printoptions(threshold=sys.maxsize)

DEBUG = True
SCALE_DATASET = True
SAVE_PATH = './machine_learning/models/model.pth'
NUM_PATHS = 500

# Value scaling function for feeding into nn
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


DATASET = 'dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
d.print_info()
# d.csi_mags = d.scale(scaler.fit_transform, d.csi_mags.T).T
not_scaled = d.csi_mags
scaler = get_scaler('minmax')
d.csi_mags = d.scale(scaler.fit_transform, d.csi_mags)
d.csi_phases = d.unwrap(d.csi_phases)
paths = d.generate_straight_paths(NUM_PATHS, 10)
dataset_mag = d.paths_to_dataset_mag_only(paths)
dataset_phase = d.paths_to_dataset_phase_only(paths)
dataset_positions = d.paths_to_dataset_positions(paths)

# # # Check 10 random positions in dataset
# # if DEBUG:
# #     for i in range(10):
# #         rand = np.random.randint(dataset_positions.shape[0])
# #         fig = plt.figure()
# #         ax = fig.add_subplot(111, projection='3d')
# #         ax.scatter(dataset_positions[rand,0,:], dataset_positions[rand,1,:], dataset_positions[rand,2,:])
# #         ax.set_xlim([0, 200])
# #         ax.set_ylim([0, 200])
# #         ax.set_zlim([-50, 50])
# #         plt.show()

# # Check 10 random CSI readings in dataset
# if DEBUG:
#     for i in range(5):
#         rand1 = np.random.randint(dataset_mag.shape[0])
#         rand2 = np.random.randint(dataset_mag.shape[1])
#         # cprint.info(f'position {positions[:,790]}')
#         plt.figure(1)
#         # plt.plot(positions[:1,:10], positions[1:2,:10],'.', color='b')
#         plt.subplot(211)
#         plt.plot(np.arange(128), not_scaled[:,rand1])
#         plt.subplot(212)
#         plt.plot(np.arange(128), descaled[:,rand1])
#         plt.show()

# # Convert 'split_sequences' to a PyTorch tensor
dataset_mag = torch.from_numpy(dataset_mag)
# Split dataset into train, val and test
X_train, X_test, y_train, y_test = train_test_split(dataset_mag[:,:9,:], dataset_mag[:,9:10,:].squeeze(), train_size = 0.95, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, shuffle=False)

# # Scale
# scaler = get_scaler('minmax')
# cprint.info(f'X_train shape {type(X_train)}')
# X_train = torch.from_numpy(scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape))
# cprint.info(f'X_train shape {type(X_train)}')
# cprint.info(f'y_train shape {type(y_train)}')
# X_val = torch.from_numpy(scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape))
# X_test = torch.from_numpy(scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape))

# Dataset
train = TensorDataset(X_train, y_train)
validate = TensorDataset(X_val, y_val)
test = TensorDataset(X_test, y_test)
# Define hyperparameters
batch_size = 100
shuffle = True

# Create a data loader
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
validate_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

# Loop through the data using the data loader
for i, batch in enumerate(train_dataloader, 0):
    cprint(f'Batch {i} has size {batch[0].shape[0]}')  # Check the batch size


def get_model(model, model_params):
    models = {
        "rnn": RNN,
        "lstm": LSTM,
        "gru": GRU,
    }
    return models.get(model.lower())(**model_params)

# Hyperparameters
input_size = 128
hidden_size = 32
num_layers = 5
output_size = 128
sequence_length = 9
learning_rate = 0.005
dropout = .2
num_epochs = 100
batch_size = batch_size
model_type = "gru"
model_params = {'input_size': input_size,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'output_size' : output_size,
                'dropout_prob' : dropout}

current = datetime.datetime.now()
writer = SummaryWriter(f"runs/test{model_type}_{num_epochs}_{num_layers}_{hidden_size}_{learning_rate}_{dropout}_{NUM_PATHS}_{current.month}-{current.day}-{current.hour}:{current.minute}")

# Create a simple RNN model
model = get_model(model_type, model_params)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate adjustment
# decay_lr_lambda = lambda epoch: 0.2 if optimizer.param_groups[0]['lr'] < 0.0001 else 0.915 ** (epoch // 5)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr_lambda)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# Training loop
model = model.float()
i = 0
for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_val_loss = 0

    model.train()
    for batch in train_dataloader:
        sequences, targets = batch
        outputs = model(sequences.float())
        loss = criterion(outputs, targets.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        writer.add_scalar("losses/running_train_loss", loss.item(), i)
        i += 1

    model.eval()
    with torch.no_grad():
        for batch in validate_dataloader:
            sequences, targets = batch
            outputs = model(sequences.float())
            val_loss = criterion(outputs, targets.float())
            running_val_loss += val_loss.item()
            writer.add_scalar("losses/running_val_loss", val_loss.item(), i)


    if epoch % 100 == 99:
        print(f'[{epoch + 1}, {num_epochs}] loss: {running_train_loss:.3f}')
        running_train_loss = 0.0

    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i)
    scheduler.step(val_loss.item())

# Save test dataset tensors
torch.save(X_test, f'./machine_learning/models/X_test_{model_type}.pt')
torch.save(y_test, f'./machine_learning/models/y_test_{model_type}.pt')

# # Run Test dataset
# model.eval()

# test_loss = 0.0  # Initialize the test loss
# total_samples = 0  # Total samples in the test dataset

# for i, batch in enumerate(test_dataloader, 0):  # Assuming you have a DataLoader for the test dataset
#     sequences, targets = batch  # Get input sequences and their targets

#     with torch.no_grad():
#         outputs = model(sequences.float())  # Make predictions

#     loss = criterion(outputs, targets)  # Calculate the loss
#     writer.add_scalar("losses/running_test_loss", loss.item(), i)
#     test_loss += loss.item()
#     total_samples += 1

# # Calculate the average test loss
# average_test_loss = test_loss / total_samples

# # Optionally, calculate and print other evaluation metrics if needed
# print(f"Average Test Loss: {average_test_loss:.4f}")


if DEBUG:
    # Sanity check
    for i in range(3):
        # To use the trained model for prediction, you can pass new sequences to the model:
        # new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]
        rand = torch.randint(0, X_test.shape[0], (1,))
        new_input = X_test[rand,:]
        ground_truth = y_test[rand,:]
        cprint.info(f'ground truth {ground_truth.shape}')
        # Prediction
        prediction = model(new_input.to(torch.float32))

        # Descale
        input_descaled = new_input.squeeze().detach().numpy()
        # input_descaled = scaler.inverse_transform(new_input.squeeze().detach().numpy())
        # Graphs
        fig, axs = plt.subplots(5)
        new_input = new_input.squeeze()
        # cprint.warn(f'validation shape {new_input[8,:].shape}')
        axs[0].plot(input_descaled[6,:])
        axs[0].set_title("CSI Reading 7")
        axs[1].plot(input_descaled[7,:])
        axs[1].set_title("CSI Reading 8")
        axs[2].plot(input_descaled[8,:])
        axs[2].set_title("CSI Reading 9")

        prediction = prediction.detach().numpy()
        # writer.add_figure(f'Comparison {i}', fig, global_step=0)
        # plt.close(fig)
        # plt.show()
        
        # prediction = np.expand_dims(prediction, axis=1)
        # prediction = scaler.inverse_transform(prediction)
        # cprint.info(f'Prediction shape {prediction.shape}')
        # cprint.info(f'Prediction shape {prediction}')
        # cprint.info(f'Prediction {scaler.inverse_transform(prediction).shape}')
        axs[3].plot(ground_truth.squeeze())
        axs[3].set_title("Ground Truth")
        axs[4].plot(prediction.squeeze())
        axs[4].set_title("Prediction")

        writer.add_figure(f'Comparison {i}', fig, global_step=0)
        plt.close(fig)
        # plt.show()

# Close tensorboard writer
writer.close()
# Save model
torch.save(model.state_dict(), SAVE_PATH)

# Final learning rate
final_learning_rate = optimizer.param_groups[0]['lr']
cprint.ok(f'Final learning rate {final_learning_rate}')

# # NOTE Run "tensorboard --logdir runs" to see results