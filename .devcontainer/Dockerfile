FROM python:3.11-slim-bullseye AS pip-installs
RUN apt-get update && apt-get -y install libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/kipackenv
ENV PATH="/opt/kipackenv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda12_pip] \
    ml-collections \
    numpy \
    pyfftw \
    cupy-cuda12x \
    nvidia-curand-cu12 \
    nvidia-cuda-nvrtc-cu12

FROM python:3.11-slim-bullseye
COPY --from=pip-installs /opt/kipackenv /opt/kipackenv
COPY --from=pip-installs /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1

ENV PATH="/opt/kipackenv/bin:$PATH"

ARG NVIDIA_LIB_DIR=/opt/kipackenv/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_LIB_DIR}/cuda_runtime/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cuda_nvrtc/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cublas/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cufft/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/curand/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cusolver/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cusparse/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cudnn/lib
