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
    from math import e as std

    shape = (sample_count, point_count)
    mean = 0

    # Element-wise salted log-normal distribution:
    # `NORMDIST() * NORMDIST() + NORMDIST()`
    x = random.normal(
        mean,
        std,
        shape,
    ) * random.normal(
        mean,
        std,
        shape,
    ) + random.normal(
        mean,
        std,
        shape,
    )
    y = x.mean(axis=1)

    x = x.tolist()

    return DataFrame((x, y), schema=SCHEMA)
