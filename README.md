# raytracerinc
A ray-tracer that renders in 16-color VGA palette at 640x480 resolution.
It runs on most OSes, uses CUDA for GPU parallelism

Compilation instructions (requires libgd-dev, nvidia-cuda-dev / nvidia-cuda-toolkit):

nvcc -x cu trace_cuda.cc -O3 $(pkg-config gdlib --cflags --libs) \
    -Xcompiler '-Wall -Wextra -Ofast'

Note: Depends on helper_cuda.h.
