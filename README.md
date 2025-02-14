# AIDevContainers

Devcontainers for different AI Applications

Build the base containers:

```bash
# at root

# for CPU
docker build -f dev_cpu.dockerfile -t <user>/python:cpu-3.10

# for base GPU (Moslty all use this)
docker build -f dev_gpu.dockerfile -t <user>/python:gpu-3.10

# for dev GPU (Hunyuan3d needs nvcc)
docker build -f dev_gpu.dockerfile --build-arg TAG_LABEL=12.3.1-devel-ubuntu20.04 -t <user>/python:gpu-devel-3.10
```

then go to any directory and run `docker compose up`
