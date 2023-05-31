FROM ghcr.io/mazhengcn/devcontainers/jax:latest AS base

FROM base AS claw-install
RUN apt-get update && apt-get -y install gfortran libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir setuptools wheel \
    && python -m pip install pyfftw \
    https://files.pythonhosted.org/packages/7b/3a/6b727ea26bf2945946e474ce93988722bdadffe6b9c1c1ec3935ed0806ec/clawpack-5.9.0.tar.gz

FROM base
COPY --from=claw-install /opt/venv /opt/venv