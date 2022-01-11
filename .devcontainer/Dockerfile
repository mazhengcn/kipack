# [Choice] CUDA and Python version: 11.4-3.9
ARG VARIANT=3.10-11.5
FROM zhengma/python-3-cuda:${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends gfortran \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements-dev.txt /tmp/pip-tmp/
RUN pip3 install -U pip wheel \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-dev.txt \
    # Install clawpack
    && pip3 --disable-pip-version-check --no-cache-dir install https://files.pythonhosted.org/packages/e0/8a/51dfbd1b597f79ed59e9b651b31cfa9c7eb75d3e6558652da7eff06bd989/clawpack-5.8.2.tar.gz \
    && rm -rf /tmp/pip-tmp
