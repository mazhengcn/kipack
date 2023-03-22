# JAX and Python
ARG JAX_VERSION=0.4.6
ARG PYTHON_VERSION=3.11
ARG VENV_PATH=/opt/kipackenv
ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=8
ARG CUDA_PATCH_VERSION=0
ARG CUDA_VERSION=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
ARG CUDNN_MAJOR_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}.${CUDA_PATCH_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu22.04 AS lib

FROM python:${PYTHON_VERSION}-slim AS py
ARG JAX_VERSION
ARG VENV_PATH

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    git \
    gfortran \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv ${VENV_PATH}

ENV PATH=${VENV_PATH}/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda11_cudnn86]==${JAX_VERSION} \
    ml-collections \
    numpy \
    pyfftw \
    cupy-cuda11x \
    https://files.pythonhosted.org/packages/7b/3a/6b727ea26bf2945946e474ce93988722bdadffe6b9c1c1ec3935ed0806ec/clawpack-5.9.0.tar.gz

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION
ARG VENV_PATH
ARG CUDA_MAJOR_VERSION
ARG CUDA_VERSION
ARG CUDNN_MAJOR_VERSION
# Python packages
COPY --from=py ${VENV_PATH} ${VENV_PATH}
# CUDA libraries
COPY --from=lib /usr/local/cuda/compat /usr/local/cuda/compat
COPY --from=lib /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
COPY --from=lib \
    /usr/local/cuda/lib64/libcublas.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcublasLt.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcudart.so.${CUDA_MAJOR_VERSION}.0 \
    /usr/local/cuda/lib64/libcufft.so.10 \
    /usr/local/cuda/lib64/libcufftw.so.10 \
    /usr/local/cuda/lib64/libcuinj64.so.$CUDA_VERSION \
    /usr/local/cuda/lib64/libcupti.so.$CUDA_VERSION \
    /usr/local/cuda/lib64/libcurand.so.10 \
    /usr/local/cuda/lib64/libcusolver.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcusolverMg.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcusparse.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libnvrtc-builtins.so.${CUDA_VERSION} \
    /usr/local/cuda/lib64/libnvrtc.so.${CUDA_MAJOR_VERSION}.2 \
    /usr/local/cuda/lib64/
COPY --from=lib \
    /usr/lib/x86_64-linux-gnu/libcudnn.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/
COPY --from=lib /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas

ENV PATH=${VENV_PATH}/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gpu:$LD_LIBRARY_PATH
ENV LC_ALL=C.UTF-8
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
