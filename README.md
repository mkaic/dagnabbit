## dagnabbit
This project is a continuation of [dagnabbit-nim](https://github.com/mkaic/dagnabbit-nim), a program I used to learn the [Nim programming language.](https://nim-lang.org). While it was very educational, I didn't feel like figuring out multithreading, so I'm just making a NumPy/PyTorch port/continuation of it.
## Environment
You'll need to install `uv` if you don't already have it:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
then create and activate the venv:
```
uv sync
source .venv/bin/activate
```

## Profiling
I learned about profiling for this project. To generate a logfile, I run:
```python -m cProfile -o profile.dat -m dagnabbit.scripts.main```
And to visualize that logfile:
```python -m snakeviz profile.dat```
This requires the PyPI package `snakeviz`. It will create and open a web interface that visualizes how much CPU time my functions are taking.

## Things to try:
* Make compute graph evaluation much more efficient by only evaluating nodes which are upstream of the outputs
    * Could also probably reduce memory footprint by dynamically deleting node cached states after they have zero references, but compute seems to be much more of a bottleneck than memory right now.
* Add auxiliary loss(es) to discourage collapse to the (grey, monotone) mean.
* Make bitaddresses more informative by giving X, Y, and channel coordinates as separate bitstrings concatenated together. Would make addresses a bit longer, but in theory much easier for the model to understand.