from torch import nn, tensor
from .hyper_parameters import TrainingHyperParameters
from ..data_construction.random_construct import random_construct
from tqdm.auto import tqdm

HP = list(TrainingHyperParameters.get_all_combinations())[-1]
DATA = random_construct(100, 30)
X = tensor(DATA.get_column('x'))
Y = tensor(DATA.get_column('y')).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(X.size(1), HP.hidden_node_count),
    *((HP.normalizer(HP.hidden_node_count),) if HP.normalizer else ()),
    HP.activation_function(),
    nn.Linear(HP.hidden_node_count, Y.size(1)),
)

def init_weights(module: nn.Module) -> None:
    try:
        HP.weight_initializer(module.weight)
        nn.init.zeros_(module.bias)
    except AttributeError:
        return

model.apply(init_weights)

# # In PyTorch, `weight_decay` is also called L2 penalty.
optimizer = HP.optimizer(model.parameters(), weight_decay=HP.regularization_factor)
scheduler = HP.learning_rate_scheduler(optimizer) if HP.learning_rate_scheduler else None

epochs = HP.learning_epochs
for _ in tqdm(range(epochs), total=epochs):
    optimizer.zero_grad()

    loss = HP.loss_function(model(X), Y)
    loss.backward()

    optimizer.step()
    if scheduler:
        scheduler.step()

# TrainingHyperParameters(hidden_node_count=11,
#                 activation_function=<class 'torch.nn.modules.activation.ReLU'>,
#                 weight_initializer=<function kaiming_normal_ at 0x1293c0040>,
#                 loss_function=<function mse_loss at 0x129321750>,
#                 regularization_factor=0.0001,
#                 optimizer=<class 'torch.optim.adam.Adam'>,
#                 learning_epochs=300,
#                 learning_rate_scheduler=<class 'torch.optim.lr_scheduler.CosineAnnealingLR'>,
#                 ensemble_count=5,
#                 normalizer=<class 'torch.nn.modules.batchnorm.BatchNorm1d'>)
