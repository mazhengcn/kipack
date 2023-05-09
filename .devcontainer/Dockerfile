FROM python:3.11-slim-bullseye AS base

FROM base AS pip-installs
RUN apt-get update && apt-get -y install gfortran libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/kipackenv
ENV PATH="/opt/kipackenv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir setuptools wheel \
    && pip install https://files.pythonhosted.org/packages/7b/3a/6b727ea26bf2945946e474ce93988722bdadffe6b9c1c1ec3935ed0806ec/clawpack-5.9.0.tar.gz

FROM base
COPY --from=pip-installs /opt/kipackenv /opt/kipackenv
ENV PATH="/opt/kipackenv/bin:$PATH"