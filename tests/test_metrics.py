import numpy as np
import pytest

from kneemil.metrics import bootstrap_macro_auc, macro_auc


def test_macro_auc_perfect_and_constant():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    m, per = macro_auc(y[:, :1], y[:, :1].astype(float))
    assert m == pytest.approx(1.0)
    m, per = macro_auc(np.c_[y[:, 0], np.ones(4)], np.c_[y[:, 0], np.ones(4)])
    assert np.isnan(per[1]) and m == pytest.approx(1.0)


def test_bootstrap_interval_contains_mean():
    rs = np.random.RandomState(0)
    y = rs.randint(0, 2, (58, 12))
    p = y * 0.3 + rs.rand(58, 12)  # informative but imperfect
    mean, (lo, hi) = bootstrap_macro_auc(y, p, n=200)
    assert lo < mean < hi and 0.5 < mean < 1
