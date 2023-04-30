# JAX and Python
ARG JAX_VERSION=0.4.8
ARG PYTHON_VERSION=3.11
ARG VENV_PATH=/opt/kipackenv
ARG CUDA_MAJOR_VERSION=12

FROM python:${PYTHON_VERSION}-slim-bullseye AS pip-installs
ARG JAX_VERSION
ARG VENV_PATH
ARG CUDA_MAJOR_VERSION

RUN apt-get update && apt-get -y install libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda${CUDA_MAJOR_VERSION}_pip]==${JAX_VERSION} \
    ml-collections \
    numpy \
    pyfftw \
    cupy-cuda12x \
    nvidia-curand-cu12 \
    nvidia-cuda-nvrtc-cu12

FROM python:${PYTHON_VERSION}-slim-bullseye
ARG VENV_PATH
ARG NVIDIA_LIB_DIR=${VENV_PATH}/lib/python3.11/site-packages/nvidia

# Python packages
COPY --from=pip-installs ${VENV_PATH} ${VENV_PATH}
COPY --from=pip-installs /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1

ENV PATH=${VENV_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH=${NVIDIA_LIB_DIR}/cuda_runtime/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cuda_nvrtc/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cublas/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cufft/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/curand/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cusolver/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_LIB_DIR}/cusparse/lib
