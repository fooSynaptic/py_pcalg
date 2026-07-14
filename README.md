# py-pcalg

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

**Lightweight PC algorithm for causal skeleton discovery** — a pip-installable Python tool for learning causal graph structure from observational data.

> 纯 Python 实现的 **PC 算法**（Peter–Clark），用于从观测数据中发现因果骨架。比 R `pcalg` 更轻量，比 `causal-learn` 更易读。

---

## Why this tool

| | py-pcalg | causal-learn | R pcalg |
|---|----------|--------------|---------|
| Install | `pip install` + CLI | heavier deps | R runtime |
| Code size | ~1k LOC, readable | full toolkit | reference impl |
| Use case | quick skeleton + viz | research pipeline | production stats |

---

## Install

```bash
pip install -e .          # from source
# or after publishing:
# pip install py-pcalg
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `networkx`.

---

## CLI

```bash
# bundled demo (test_data.csv)
pcalg demo

# your CSV (numeric columns)
pcalg run your_data.csv --alpha 0.05 -o skeleton.png

# headless CI
pcalg run your_data.csv --no-plot
```

---

## Python API

```python
import pandas as pd
from pcalg import skeleton, independence_test, visualize_graph

data = pd.read_csv("your_data.csv")
corr = data.corr()
labels = list(range(data.shape[1]))
label_dict = {i: str(c) for i, c in enumerate(data.columns)}

graph = skeleton(
    suff_stat=[corr.to_numpy(), len(data)],
    indep_test=independence_test,
    alpha=0.05,
    labels=labels,
)

visualize_graph(graph, label_dict, save_path="skeleton.png")
print(graph.M)
```

Legacy imports still work: `from pc import skeleton`, `from utils import Matrix`.

---

## Algorithm

1. Start from a complete undirected graph.
2. **Skeleton discovery**: for each edge (X, Y), search conditioning sets S; remove X–Y if X ⊥ Y | S (Fisher-Z test, α = 0.05 by default).
3. **Orientation** (partial): V-structures and Meek rules — full CPDAG orientation is a roadmap item.

Independence test: Fisher's Z on partial correlations (see `pcalg.utils.independence_test`).

---

## Project layout

```
py_pcalg/
├── pcalg/           # installable package
│   ├── pc.py        # skeleton(), pc()
│   ├── utils.py     # Matrix, Fisher-Z tests
│   ├── graph.py     # NetworkX + matplotlib viz
│   └── cli.py       # pcalg command
├── tests/           # pytest
├── test_data.csv    # demo dataset
└── pyproject.toml
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Related work

- [causal-learn](https://github.com/cmu-phil/causal-learn) — CMU toolkit (PC, FCI, GES, …)
- [gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) — Huawei causal structure learning
- [pcalg (R)](https://CRAN.R-project.org/package=pcalg) — original reference implementation

**References:** Spirtes, Glymour & Scheines (2000); Kalisch & Bühlmann (2007).

---

## Assumptions & limits

- Causal Markov + faithfulness; no latent confounders (use FCI for that).
- Needs sufficient sample size for stable partial-correlation tests.
- Dense graphs can be slow — `m_max` caps conditioning-set size.

---

## License

MIT — see repository root.

