# syntax=docker/dockerfile:1

###############################################################################
# EulSim — Eulerian Frame Simulator
# Multi-stage build: deps are compiled into a venv in `builder`, then the venv
# and the pure-Python package are copied into a slim, non-root runtime image.
###############################################################################

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Isolated venv so the runtime stage gets deps without pip's own footprint.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="EulSim" \
      org.opencontainers.image.description="Eulerian Frame Simulator — interactive web demo" \
      org.opencontainers.image.source="https://github.com/AlessandroRuzza/Eulerian-frame" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8001 \
    EULSIM_SHARE=local

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Unprivileged runtime user; /app stays read-only to it.
RUN groupadd --system --gid 1001 eulsim \
 && useradd --system --uid 1001 --gid eulsim --home-dir /app --shell /usr/sbin/nologin eulsim

# Application code. `eulsim/web/index.html` is required at runtime by page.py.
COPY --chown=root:root eulsim/ ./eulsim/
COPY --chown=root:root run_eulsim.py ./
COPY --chown=root:root docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY --chown=root:root docker-healthcheck.py /usr/local/bin/docker-healthcheck.py
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

USER eulsim

EXPOSE 8001

# Stdlib-only probe: no curl/wget in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "/usr/local/bin/docker-healthcheck.py"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD []
