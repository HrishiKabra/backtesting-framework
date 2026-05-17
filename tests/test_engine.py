import pandas as pd
import numpy as np
import pytest
from core.engine import LookaheadBarrier, LookaheadBiasError


def test_get_history_excludes_signal_date(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    signal_date = synthetic_data.index[10]
    history = barrier.get_history(signal_date)
    assert history.index.max() < signal_date


def test_get_history_raises_on_date_not_in_index(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    bad_date = pd.Timestamp("2019-01-01")  # before data starts
    with pytest.raises(LookaheadBiasError):
        barrier.get_history(bad_date)


def test_get_shifted_data_lags_by_one(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    shifted = barrier.get_shifted_data()
    # Row at index[1] should equal the original row at index[0]
    original_row = synthetic_data.iloc[0]
    shifted_row = shifted.iloc[1]
    pd.testing.assert_series_equal(original_row, shifted_row, check_names=False)


def test_get_shifted_data_first_row_is_nan(synthetic_data):
    barrier = LookaheadBarrier(synthetic_data)
    shifted = barrier.get_shifted_data()
    assert shifted.iloc[0].isna().all()
