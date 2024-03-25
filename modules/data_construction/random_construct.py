from polars.dataframe import DataFrame
from .schema import SCHEMA


def random_construct(
    sample_count: int,
    vector_count: int,
    dimension: int,
) -> DataFrame:
    """
    Constructs a random dataset of the given count and dimension

    ## Parameters
    - `sample_count`: The number of samples to generate
    - `vector_count`: The number of vectors to generate
    - `dimension`: The dimension of each vector

    ## Returns
    - (`DataFrame`):
        - `schema`: `SCHEMA`
    """
    from numpy import random, typing

    shape = (sample_count, vector_count, dimension)

    # Element-wise salting:
    # `2 * R() * R() + R()`
    x = 2 * random.randn(*shape) * random.randn(*shape) + random.randn(*shape)

    y: typing.NDArray = x.mean(axis=1)

    x: list = x.tolist()
    y: list = y.tolist()

    return DataFrame((x, y), schema=SCHEMA)
