from polars.dataframe import DataFrame
from .schema import SCHEMA


def random_construct(
    sample_count: int,
    point_count: int,
) -> DataFrame:
    """
    Constructs a random dataset from the given parameters

    ## Parameters
    - `sample_count`: The number of samples to generate
    - `point_count`: The number of points to generate

    ## Returns
    - (`DataFrame`):
        - `schema`: `SCHEMA`
    """
    from numpy import random

    shape = (sample_count, point_count)

    # Element-wise salting:
    # `10 * R() * R() + R()`
    x = 10 * random.randn(*shape) * random.randn(*shape) + random.randn(*shape)
    y = x.mean(axis=1)

    x = x.tolist()
    y = y.tolist()

    return DataFrame((x, y), schema=SCHEMA)
