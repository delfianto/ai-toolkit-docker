# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ostris/aitoolkit@sha256:d0b5f7298631bf1a6db9d9f43a3a1e8072296f9c072477adabd9456ee4015ae1
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="Ostris AI Toolkit" \
    org.opencontainers.image.source="https://github.com/delfianto/ai-toolkit-docker" \
    org.opencontainers.image.description="Ostris AI Toolkit with explicit local model catalogs and authenticated MCP"

# Keep MCP dependencies separate from the upstream training environment.
COPY docker/mcp-requirements.txt /tmp/ai-toolkit-mcp-requirements.txt
RUN python -m venv /opt/toolkit-mcp \
    && /opt/toolkit-mcp/bin/pip install --no-cache-dir -r /tmp/ai-toolkit-mcp-requirements.txt

COPY src /opt/ai-toolkit
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/ai-toolkit-models.pth /tmp/ai-toolkit-models.pth
# Locate site-packages rather than depending on an Ubuntu/Python version.
RUN python -c 'import pathlib, site, shutil; shutil.copy("/tmp/ai-toolkit-models.pth", pathlib.Path(site.getsitepackages()[0]) / "ai-toolkit-models.pth")' \
    && /opt/toolkit-mcp/bin/python -c 'import pathlib, site, shutil; shutil.copy("/tmp/ai-toolkit-models.pth", pathlib.Path(site.getsitepackages()[0]) / "ai-toolkit-models.pth")' \
    && mkdir -p /app/ai-toolkit/state \
    && rm -f /app/ai-toolkit/aitk_db.db \
    && ln -s state/aitk_db.db /app/ai-toolkit/aitk_db.db
ENV AI_TOOLKIT_OFFLINE=1
WORKDIR /app/ai-toolkit
ENTRYPOINT ["sh", "/entrypoint.sh"]
