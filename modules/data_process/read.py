from typing import TextIO
from polars.dataframe import DataFrame
from ..data_construction.schema import SCHEMA


def read_jsonl(source: TextIO | str) -> DataFrame:
    """
    Deserialize the dataset in JSON format.

    ## Parameters
    - `source`:
        The file or string representation to read the result from.

    ## Returns
    - (`DataFrame`):
        - `schema`: `SCHEMA`
    """
    from io import StringIO
    from polars.io import read_ndjson

    if type(source) is str:
        return read_ndjson(StringIO(source), schema=SCHEMA)
    else:
        return read_ndjson(source, schema=SCHEMA)
