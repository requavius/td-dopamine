# The playable task (web.py). Data lives in /data: mount a persistent volume there.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY *.py ./
COPY web ./web

ENV TEMPORAL_DATA=/data PORT=8000
VOLUME /data
EXPOSE 8000
# One worker: live sessions are held in memory.
CMD ["uv", "run", "--no-sync", "python", "web.py", "--host", "0.0.0.0"]
