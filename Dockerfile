# The forecast API. Two stages: uv installs the locked *core* dependencies (no training,
# tracking or monitoring extras) into a virtualenv in the builder; the runtime image gets only
# that virtualenv and runs as an unprivileged user.
#
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t nem-forecast-api .
#   docker run -p 8000:8000 -e DATABASE_URL=... -e MODEL_REPO_ID=... nem-forecast-api

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install --yes --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
# Polars' runtime library alone ships 187 MB with full debug info; stripping it (and the rest)
# is the single biggest saving. Test suites and type stubs are dead weight at runtime.
RUN uv sync --frozen --no-dev --no-editable \
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyi' -delete \
    && find /opt/venv -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null; \
    true


FROM python:${PYTHON_VERSION}-slim AS runtime
# Empty unless passed: on Render the running service reports RENDER_GIT_COMMIT instead.
ARG GIT_SHA=
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_SHA=${GIT_SHA} \
    MODEL_CACHE_DIR=/home/app/.cache/models \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
# One thread per pool: on Render's 0.1-CPU free tier, thread pools sized to the host's cores
# only contend for the quota (measured: ready in 57 s instead of 105 s at 0.1 CPU).
# LightGBM needs the OpenMP runtime, which the slim base does not ship.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
# Fail the build, not the deploy, if the app cannot import (a missing system library, say).
RUN python -c "import nem_forecast.serve.app"
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
CMD ["python", "-m", "nem_forecast.serve"]
