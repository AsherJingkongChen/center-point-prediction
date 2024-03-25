from torch import nn
from torch import optim

HIDDEN_NODES = (5, 8, 11)
ACTIVATION_FUNCTIONS = (nn.Tanh, nn.ReLU)
WEIGHT_INITIALIZERS = (nn.init.normal_, nn.init.xavier_normal_, nn.init.kaiming_normal_)
LOSS_FUNCTIONS = (nn.functional.mse_loss)
REGULARIZATION_FACTORS = (0.001, 0.0001)
OPTIMIZERS = (optim.SGD, optim.Adam)
LEARNING_EPOCHS = (100, 200, 300)

