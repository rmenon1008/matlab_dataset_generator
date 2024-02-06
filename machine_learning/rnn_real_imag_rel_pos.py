import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset.consumer import DatasetBuilder
from models import GRU, LSTM, RNN
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import (
#     MaxAbsScaler,
#     MinMaxScaler,
#     RobustScaler,
#     StandardScaler,
# )
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from utils import calculate_metrics, dbm_to_watts, get_scaler, watts_to_dbm

# import os


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# np.set_printoptions(threshold=sys.maxsize)


# for hidden_size in [8, 16, 32, 64, 128]:
DEBUG = True
N_PATHS = 50_000
SCALER = "quantile"
DATASET = "dataset_0_5m_spacing.h5"
DS_OPTIONS = {
    "n_paths": N_PATHS,
    "mag_scaler": SCALER,
    "paths": {
        "path_type": "curved",
        "path_length": 20,
        "terminal_length": 0,
        "terminal_direction": "center",
    },
    "dataset_sections": ["mag", "phase", "relative_position"],
}


# Hyperparameters
batch_size = 5_000
shuffle = True
input_size = 258
hidden_size = 130
num_layers = 5
output_size = 258
sequence_length = 19
learning_rate = 0.005
dropout = 0.15
num_epochs = 100
batch_size = batch_size
model_type = "gru"
model_params = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "output_size": output_size,
    "dropout_prob": dropout,
}

DATASET = "dataset_0_5m_spacing.h5"
d = DatasetBuilder(DATASET)
d.print_info()

# Get dataset
dataset, path_point_to_csi = d.generate_dataset(DS_OPTIONS)
dataset = torch.from_numpy(dataset)
print(f"Dataset shape: {dataset.shape}")

# Split dataset into train, val and test
X_train, X_test, y_train, y_test = train_test_split(
    dataset[:, :sequence_length, :],
    dataset[:, sequence_length:, :].squeeze(),
    train_size=0.85,
    shuffle=False,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, shuffle=False
)

# Dataset
train = TensorDataset(X_train, y_train)
validate = TensorDataset(X_val, y_val)
test = TensorDataset(X_test, y_test)

# Create a data loader
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
validate_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

# Loop through the data using the data loader
for i, batch in enumerate(train_dataloader, 0):
    print(f"Batch {i} has size {batch[0].shape[0]}")  # Check the batch size


def get_model(model, model_params):
    models = {
        "rnn": RNN,
        "lstm": LSTM,
        "gru": GRU,
    }
    return models.get(model.lower())(**model_params)


layout = {
    "ABCDE": {
        "loss": [
            "Multiline",
            [
                "losses/running_train_loss",
                "losses/running_val_loss",
                "losses/test_loss",
            ],
        ],
        "accuracy": [
            "Multiline",
            ["accuracy_val/mae", "accuracy_val/rmse", "accuracy_val/r2"],
        ],
    },
}
current = datetime.datetime.now()
writer = SummaryWriter(
    f"runs/{current.month}-{current.day}-{current.hour}:{current.minute}_{model_type}_{num_epochs}_{num_layers}_{hidden_size}_{learning_rate}_{dropout}_{N_PATHS}_{batch_size}_{SCALER}"
)
writer.add_custom_scalars(layout)

# Create a simple RNN model
model = get_model(model_type, model_params)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate adjustment
# decay_lr_lambda = lambda epoch: 0.2 if optimizer.param_groups[0]['lr'] < 0.0001 else 0.915 ** (epoch // 5)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr_lambda)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5, verbose=True
)

# Training loop
model = model.float()
i = 0
j = 0
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
            # val_loss = criterion(path_point_to_csi(outputs), targets.float())
            running_val_loss += val_loss.item()
            writer.add_scalar("losses/running_val_loss", val_loss.item(), j)

            # Create dataframe
            df_result = pd.DataFrame(
                {
                    "value": targets.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
                    "prediction": outputs.flatten(),
                }
            )

            # Calcuate metrics
            result_metrics = calculate_metrics(df_result)
            writer.add_scalar("accuracy_val/mae", result_metrics["mae"], j)
            writer.add_scalar("accuracy_val/rmse", result_metrics["rmse"], j)
            writer.add_scalar("accuracy_val/r2", result_metrics["r2"], j)
            j += 1

    if epoch % 100 == 99:
        print(f"[{epoch + 1}, {num_epochs}] loss: {running_train_loss:.3f}")
        running_train_loss = 0.0

    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], i)
    scheduler.step(val_loss.item())

# Save test dataset tensors
torch.save(X_test, f"./machine_learning/models/X_test_{model_type}.pt")
torch.save(y_test, f"./machine_learning/models/y_test_{model_type}.pt")

## Run Test dataset
model.eval()

test_loss = 0.0  # Initialize the test loss
total_samples = 0  # Total samples in the test dataset

predictions = []
with torch.no_grad():
    for batch in test_dataloader:  # Assuming you have a DataLoader for the test dataset
        sequences, targets = batch  # Get input sequences and their targets
        print(f"sequences {sequences.shape}")
        outputs = model(sequences.float())  # Make predictions
        predictions.append(outputs)
        loss = criterion(outputs, targets)  # Calculate the loss
        writer.add_scalar("losses/test_loss", loss.item(), total_samples)
        test_loss += loss.item()
        total_samples += 1
        print(f"total_samples {total_samples}")

# Calculate the average test loss
average_test_loss = test_loss / total_samples

# Optionally, calculate and print other evaluation metrics if needed
print(f"Average Test Loss: {average_test_loss:.4f}")

# Create dataframe
# df_result = pd.DataFrame(
#     {
#         "value": y_test.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
#         "prediction": predictions[0].flatten(),
#     }
# )

# # Calcuate metrics
# result_metrics = calculate_metrics(df_result)
# print(f"Results: {result_metrics}")

if DEBUG:
    # Sanity check
    for i in range(10):
        # To use the trained model for prediction, you can pass new sequences to the model:
        rand = torch.randint(0, X_test.shape[0], (1,))
        new_input = X_test[rand, :]
        print(f"new_input {new_input.shape}")
        prediction = model(new_input.float())
        ground_truth = y_test[rand, :]

        prediction_descaled = path_point_to_csi(prediction.detach().numpy().squeeze())
        ground_truth_descaled = path_point_to_csi(
            ground_truth.detach().numpy().squeeze()
        )

        # Graphs
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(prediction_descaled[0], label="prediction")
        ax[0].plot(ground_truth_descaled[0], label="ground truth")
        ax[0].set_title("Magnitude")
        ax[0].legend()

        ax[1].plot(prediction_descaled[1], label="prediction")
        ax[1].plot(ground_truth_descaled[1], label="ground truth")
        ax[1].set_title("Phase")
        ax[1].legend()

        writer.add_figure(f"Comparison {i}", fig, global_step=0)
        plt.close(fig)

    # Close tensorboard writer
    writer.close()
    # Save model
    torch.save(model.state_dict(), f"./machine_learning/models/{model_type}.pt")

    # Final learning rate
    final_learning_rate = optimizer.param_groups[0]["lr"]
    print(f"Final learning rate {final_learning_rate}")

# # NOTE Run "tensorboard --logdir runs" to see results
