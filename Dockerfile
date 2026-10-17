# Lint-and-test image for hive-rl.
#
# Houdini is not in this image: it holds the pure-Python package (model, PPO maths,
# curriculum) and the tests, which fake the `hou` module. Training still runs on
# Windows under hython.
#
#   docker build -t hive-rl .
#   docker run --rm hive-rl

# Python 3.10 matches the interpreter inside Houdini 20.0.
FROM python:3.10-slim

# Pin uv to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer, so editing code doesn't re-download torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["sh", "-c", "uv run ruff check . && uv run ruff format --check . && uv run pytest -v"]