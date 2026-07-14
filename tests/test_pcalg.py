import numpy as np
import pandas as pd

from pcalg.pc import skeleton
from pcalg.utils import Matrix, independence_test, partial_correlation


def test_matrix_helpers():
    m = Matrix(np.array([[0, 1], [1, 0]], dtype=float))
    assert m.any()
    m.diag(0)
    assert m.M[0, 0] == 0


def test_partial_correlation_on_identity():
    corr = np.eye(4)
    assert partial_correlation(0, 1, 2, corr) == 0.0


def test_skeleton_runs_on_random_data():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 4))
    corr = pd.DataFrame(data).corr().to_numpy()
    graph = skeleton(
        suff_stat=[corr, len(data)],
        indep_test=independence_test,
        alpha=0.05,
        labels=list(range(4)),
    )
    assert graph.shape == (4, 4)
    assert graph.M.diagonal().tolist() == [0, 0, 0, 0]
