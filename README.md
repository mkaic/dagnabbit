## dagnabbit
This project is a continuation of [dagnabbit-nim](https://github.com/mkaic/dagnabbit-nim), a program I used to learn the [Nim programming language.](https://nim-lang.org). While it was very educational, I didn't feel like figuring out multithreading, so I'm just making a NumPy/PyTorch port/continuation of it.
## Environment
```
conda env create -f environment.yaml
conda activate dagnabbit
```

## Profiling
I learned about profiling for this project. To generate a logfile, I run:
```python -m cProfile -o profile.dat -m dagnabbit.scripts.main```
And to visualize that logfile:
```python -m snakeviz profile.dat```
This requires the PyPI package `snakeviz`. It will create and open a web interface that visualizes how much CPU time my functions are taking.