from polars.datatypes import Float32, List

SCHEMA = {
    "x": List(List(Float32)),
    "y": List(Float32),
}
"""
For each tuple of `DataFrame`:
- `x` (`List[Float32]`): Randomly generated vectors with count `vector_count`
- `y` (`Float32`): The arithmetic mean of the vectors
"""
