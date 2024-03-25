from polars.datatypes import Float32, List

SCHEMA = {
    "x": List(Float32),
    "y": Float32,
}
"""
For each tuple of `DataFrame`:
- `x` (`List[Float32]`): Randomly generated points
- `y` (`Float32`): The arithmetic mean of the points
"""
