# Center Point Prediction

## Introduction

Predict the center of 30 1D-points using a two-layer neural network.

## Experiment

Please open your terminal and change the working directory to this project folder, and run these commands:

```shell
python3 -m pip install -r requirements.txt
python3 -m modules.prediction.experiment
```

## Output

```plaintext
100%|███████████████████████████████████████████████████████| 1296/1296 [01:30<00:00, 14.60it/s]
'Top 5 hyper-parameters:'
(0.010496296919882298,
 TrainingHyperParameters(hidden_node_count=5,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x????????>,
                         loss_function=<function mse_loss at 0x????????>,
                         regularization_factor=0.001,
                         optimizer=<class 'modules.prediction.optimizers.Momentum'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None))
(0.015598484314978123,
 TrainingHyperParameters(hidden_node_count=8,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x????????>,
                         loss_function=<function mse_loss at 0x????????>,
                         regularization_factor=0.001,
                         optimizer=<class 'modules.prediction.optimizers.Momentum'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None))
(0.03217921778559685,
 TrainingHyperParameters(hidden_node_count=5,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function kaiming_normal_ at 0x????????>,
                         loss_function=<function mse_loss at 0x????????>,
                         regularization_factor=0.0001,
                         optimizer=<class 'modules.prediction.optimizers.Momentum'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None))
(0.034538690000772476,
 TrainingHyperParameters(hidden_node_count=8,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function kaiming_normal_ at 0x????????>,
                         loss_function=<function mse_loss at 0x????????>,
                         regularization_factor=0.0001,
                         optimizer=<class 'modules.prediction.optimizers.Momentum'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None))
(0.04320593550801277,
 TrainingHyperParameters(hidden_node_count=5,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x????????>,
                         loss_function=<function mse_loss at 0x????????>,
                         regularization_factor=0.0001,
                         optimizer=<class 'modules.prediction.optimizers.Momentum'>,
                         learning_epochs=200,
                         learning_rate_scheduler=None,
                         normalizer=None))
```
