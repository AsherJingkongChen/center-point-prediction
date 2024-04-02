from torch import no_grad, nn, tensor
from .hyper_parameters import TrainingHyperParameters
from ..data_construction.random_construct import random_construct
from tqdm.auto import tqdm

# Define the constants for the experiment
DEVICE = "cpu"
DATA = random_construct(100, 30)
DATA_EVALUATION = random_construct(25, 30)
X = tensor(DATA.get_column("x"), device=DEVICE)
Y = tensor(DATA.get_column("y"), device=DEVICE).unsqueeze(1)
X_EVALUATION = tensor(DATA_EVALUATION.get_column("x"), device=DEVICE)
Y_EVALUATION = tensor(DATA_EVALUATION.get_column("y"), device=DEVICE).unsqueeze(1)

# Define the hyper-parameters
HP = list(TrainingHyperParameters.get_all_combinations())[-1]
print(HP)

# Define the model
model = nn.Sequential(
    nn.Linear(X.size(1), HP.hidden_node_count, device=DEVICE),
    *((HP.normalizer(HP.hidden_node_count, device=DEVICE),) if HP.normalizer else ()),
    HP.activation_function(),
    nn.Linear(HP.hidden_node_count, Y.size(1), device=DEVICE),
)


def init_weights(module: nn.Module) -> None:
    """
    A helper function to initialize the weights of a `torch.nn.Module`
    """
    try:
        HP.weight_initializer(module.weight)
        nn.init.zeros_(module.bias)
    except Exception:
        return


# Initialize the weights of the model
model.apply(init_weights)

# Define the optimizer and the learning rate scheduler
optimizer = HP.optimizer(
    model.parameters(),
    weight_decay=HP.regularization_factor,  # In PyTorch, `weight_decay` is also called L2 penalty.
)
scheduler = (
    HP.learning_rate_scheduler(optimizer) if HP.learning_rate_scheduler else None
)

# Train the model
model.train()
for _ in tqdm(range(HP.learning_epochs), total=HP.learning_epochs):
    optimizer.zero_grad()

    # Forward pass
    loss = HP.loss_function(model(X), Y)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Update the learning rate
    if scheduler:
        scheduler.step()

# Evaluate the model
model.eval()
with no_grad():
    loss_evaluation = HP.loss_function(model(X_EVALUATION), Y_EVALUATION)

# Print the loss
print(f'{{"loss": {{"train": {loss:.5f}, "eval": {loss_evaluation:.5f}}}}}')
