import pandas as pd


class LookaheadBiasError(Exception):
    """Raised when a strategy attempts to access data from the future."""


class LookaheadBarrier:
    """
    Wraps the raw OHLCV DataFrame and enforces that strategies can only access
    data strictly before a given signal date. This is structural prevention of
    lookahead bias — enforced at the API level, not by convention.
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data

    def get_history(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Returns all rows with index strictly before `as_of`."""
        if as_of not in self._data.index:
            raise LookaheadBiasError(
                f"{as_of} is not in the data index. "
                "Signal dates must be trading days within the loaded data range."
            )
        return self._data.loc[self._data.index < as_of]

    def get_shifted_data(self) -> pd.DataFrame:
        """
        Returns the full data shifted forward by 1 trading day.
        Value at index T reflects data available at market close of T-1.
        Use this for fully vectorized signal generation.
        """
        return self._data.shift(1)
