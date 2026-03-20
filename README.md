## dagnabbit
Is it possible to use machine learning and neural networks to automatically search through the space of all possible fixed-indegree DAGs?

Idk but I want to find out.

### Prerequisites

DAG rendering requires the [Graphviz](https://graphviz.org/) system package (the `dot` layout engine). Install it before using `render_dag`:

**macOS (Homebrew):**
```bash
brew install graphviz
```

**Ubuntu / Debian:**
```bash
sudo apt install graphviz
```

The Python `graphviz` wrapper is included in the project dependencies and will be installed automatically via `uv sync` or `pip install`.