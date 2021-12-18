# [Choice] CUDA and Python version: 11.4-3.9
ARG VARIANT=11.4-3.8
FROM zhengma/cuda-python:${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements-dev.txt /tmp/pip-tmp/
# Temporary hack to build dm-tree from source in Python 3.10, so we need to install cmake first.
# RUN pip3 --disable-pip-version-check --no-cache-dir install cmake
RUN pip3 install -U pip wheel \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-dev.txt \
    && rm -rf /tmp/pip-tmp
