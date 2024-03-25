from typing import TextIO
from polars.dataframe import DataFrame

def write_jsonl(
    dataset: DataFrame,
    target: TextIO | None = None,
) -> None | str:
    """
    Serialize the dataset in JSON format.

    ## Parameters
    - `dataset` (`DataFrame`):
        - `schema`: `SCHEMA`
    - `target`:
        The file to write the result to.
        If `None`, the function returns the result instead.

    ## Returns
    - If `target` is `None`, then the result is returned as `str`.
    """
    return dataset.write_ndjson(target)

