from pprint import pprint as print
from torch import no_grad, nn, tensor
from tqdm.auto import tqdm
from ..data_construction.random_construct import random_construct
from .hyper_parameters import TrainingHyperParameters
from .top_k import TopK

# Define the settings of the experiment
DEVICE = "cpu"
ENSEMBLE_COUNT: int = 5
DATA = random_construct(1000, 30)
DATA_EVALUATION = random_construct(250, 30)
X = tensor(DATA.get_column("x"), device=DEVICE)
Y = tensor(DATA.get_column("y"), device=DEVICE).unsqueeze(1)
X_EVALUATION = tensor(DATA_EVALUATION.get_column("x"), device=DEVICE)
Y_EVALUATION = tensor(DATA_EVALUATION.get_column("y"), device=DEVICE).unsqueeze(1)

progress_bar = tqdm(
    total=TrainingHyperParameters.get_all_combination_count(),
)
top_combinations = TopK(k=ENSEMBLE_COUNT)

# 1. Enumerate all hyper-parameters
# 2. update the top K combinations of training hyper-parameters
for HP in TrainingHyperParameters.get_all_combinations():
    # Define the model
    model = nn.Sequential(
        nn.Linear(X.size(1), HP.hidden_node_count, device=DEVICE),
        *(
            (HP.normalizer(HP.hidden_node_count, device=DEVICE),)
            if HP.normalizer
            else ()
        ),
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
        weight_decay=HP.regularization_factor, # In PyTorch, `weight_decay` is also called L2 penalty.
    )
    scheduler = (
        HP.learning_rate_scheduler(optimizer) if HP.learning_rate_scheduler else None
    )

    # Train the model
    model.train()
    for epoch in range(HP.learning_epochs):
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

    # Update the top K combinations of hyper-parameters
    top_combinations.update((loss_evaluation.item(), HP), key=lambda t: t[0])

    progress_bar.update()

progress_bar.close()

print(f"Top {ENSEMBLE_COUNT} hyper-parameters: ")
for entry in top_combinations:
    print(entry)
