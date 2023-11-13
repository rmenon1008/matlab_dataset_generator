import torch.nn as nn
import torch
from models import RNN, LSTM, GRU
from matplotlib import pyplot as plt
from cprint import *

SAVE_PATH = './models'
SAVE_PATH = './machine_learning/models/model.pth'

def get_model(model, model_params):
    models = {
        "rnn": RNN,
        "lstm": LSTM,
        "gru": GRU,
    }
    return models.get(model.lower())(**model_params)

# Hyperparameters
input_size = 128
hidden_size = 64
num_layers = 5
output_size = 128
sequence_length = 9
learning_rate = 0.0001
dropout = .2
num_epochs = 1000
batch_size = 100
model_type = "rnn"
model_params = {'input_size': input_size,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'output_size' : output_size,
                'dropout_prob' : dropout}

# Create a simple RNN model
model = get_model(model_type, model_params)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
X_test = torch.load('./machine_learning/models/X_test_rnn.pt')
y_test = torch.load('./machine_learning/models/y_test_rnn.pt')

for i in range(10):
        # To use the trained model for prediction, you can pass new sequences to the model:
        # new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]
        rand = torch.randint(0, X_test.shape[0], (1,))
        new_input = X_test[rand,:]
        ground_truth = y_test[rand,:]
        cprint.info(f'ground truth {ground_truth.shape}')
        # Prediction
        prediction = model(new_input.to(torch.float32))

        # Graphs
        fig, axs = plt.subplots(5)
        new_input = new_input.squeeze()
        # cprint.warn(f'validation shape {new_input[8,:].shape}')
        axs[0].plot(new_input[6,:])
        axs[0].set_title("CSI Reading 7")
        axs[1].plot(new_input[7,:])
        axs[1].set_title("CSI Reading 8")
        axs[2].plot(new_input[8,:])
        axs[2].set_title("CSI Reading 9")

        prediction = prediction.detach().numpy()
        # writer.add_figure(f'Comparison {i}', fig, global_step=0)
        # plt.close(fig)
        # plt.show()
        
        # prediction = scaler.inverse_transform(prediction)
        axs[3].plot(ground_truth.squeeze())
        axs[3].set_title("Ground Truth")
        axs[4].plot(prediction.squeeze())
        axs[4].set_title("Prediction")

        # writer.add_figure(f'Comparison {i}', fig, global_step=0)
        # plt.close(fig)
        plt.show()