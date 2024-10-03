## dagnabbit-torch
This project is a continuation of [dagnabbit](https://github.com/mkaic/dagnabbit), a project I used to learn the [Nim programming language.](https://nim-lang.org)

## Environment
I develop inside the [January 2024 Nvidia PyTorch Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags). To run GPU-enabled containers like this one on Linux, you'll need the [Nvidia Container Toolkit.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```
docker run \
-it \
-d \
--gpus all \
-v /workspace:/workspace \
nvcr.io/nvidia/pytorch:24.01-py3
```