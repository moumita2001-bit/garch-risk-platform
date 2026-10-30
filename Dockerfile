# VolRisk daily job: download -> clean -> fit -> VaR -> results JSON
FROM python:3.12-slim

# No .pyc files, unbuffered logs (so they appear immediately in CloudWatch later)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first, so code changes do not trigger a full reinstall.
#    A stub package lets pip resolve the project before the real code is copied.
COPY pyproject.toml ./
RUN mkdir -p src/volrisk && touch src/volrisk/__init__.py \
    && pip install -e . \
    && rm -rf src

# 2) The application code
COPY src/ ./src/
RUN pip install -e . --no-deps

# 3) Run as an unprivileged user with writable data and results folders
RUN useradd --create-home --uid 1000 volrisk \
    && mkdir -p data/raw data/processed results \
    && chown -R volrisk:volrisk /app
USER volrisk

# API keys are supplied at run time (--env-file locally, Secrets Manager on AWS)
ENTRYPOINT ["volrisk"]
CMD ["run"]
