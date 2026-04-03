# dagnabbit
Is it possible to use machine learning and neural networks to automatically search through the space of all possible fixed-indegree DAGs?

Idk but I want to find out.

## Prerequisites

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

## Ideas

### Encoder
* Encoding itself is pretty straightforward and kind of the backbone of the whole approach
* Get down to one vector
* Then, primary loss is going to be from guided autoregressive decoding — use the known structure of the graph for matching targets and predictions.
    * Objective 1 — correctly classify the _node type_ (one type per root node and output node, one type per kind of trunk node, one type for aggregator/scaffolding nodes)
    * Objective 2 — parent node embeddings predicted from different children for the same parent node should be similar
    * Objective 3 — postprocessing MLP which strips the impact of the scaffolding node from the embedding — trained with CLIP-style contrastive loss. Embeddings from multiple random scaffolds should be similar, embeddings from different underlying graphs should be different. Ideally, this encourages a monotonic/smooth embedding trajectory from the base graph to the final graph.
    * Objective 4 (?) — Explicitly enforce monotonicity of similarity between empty graph and a given random graph? Not sure how to do this.

### Decoder
* Blind decoding —
    * one idea is to grow from the leaves backwards (or well the single aggregate leaf in this scenario)
        * Would require a *shit ton* of compute though. Bc you'd have to predict the *whole* ancestor set *un-deduplicated* *before* you can dedupe it, because you can only dedupe confidently if you can guarantee the parent nodes of a given node are identical, and you don't know *that* until you've traced ancestry all the way back to root nodes along *all* branches.
        * Another issue with this is that it isn't guaranteed to produce a valid graph, and ways of coercing it into always terminating feel unwieldy and inelegant.
    * Alternatively, we cant try to build node-by-node from the roots up.
        * Start with the root nodes as your base graph. Embed that graph.
        * Give a "builder" network access to the *current* graph embedding and the *target* graph embedding
        * Have it iteratively predict (1) the type of the next trunk node, and then (2) the set of the parent node embeddings for that child node.
        * Match the parent node embeddings against all topologically previous nodes
        * Re-embed the new graph, rinse and repeat.
        * After you've hit your trunk node budget, you then predict the single-input for each output node.
        * You'd use the sanitized embedding from the contrastive head for this, since it should "blur out" information about the random aggregation.
        * This would probably need to be trained with RL, which kind of sucks. I don't see an easy way to make this differentiable because the child node embedding can't be computed without knowing its discrete type, so we have to make discrete type samples at every step, and those break differentiability *I THINK*. Because there's no way to differentiate through a max().
        * At least this has a more elegant termination criteria though.
        * And it can be trained separately from the encoder.