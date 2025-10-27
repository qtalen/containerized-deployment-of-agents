FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app
ADD .env /app/.env

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

EXPOSE 8000

CMD ["uv", "run", "chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]