# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ostris/aitoolkit@sha256:d0b5f7298631bf1a6db9d9f43a3a1e8072296f9c072477adabd9456ee4015ae1
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="Ostris AI Toolkit" \
    org.opencontainers.image.source="https://github.com/delfianto/ai-toolkit-docker" \
    org.opencontainers.image.description="Ostris AI Toolkit with explicit local model catalogs and authenticated MCP"

# CI resolves HEAD to an immutable commit. Local builds can retain the pinned
# base's source by omitting UPSTREAM_COMMIT.
ARG UPSTREAM_COMMIT=""
WORKDIR /app/ai-toolkit
RUN if [ -n "$UPSTREAM_COMMIT" ]; then \
      test "${#UPSTREAM_COMMIT}" -eq 40 && \
      case "$UPSTREAM_COMMIT" in *[!0-9a-f]*) exit 1 ;; esac && \
      git remote set-url origin https://github.com/ostris/ai-toolkit.git && \
      git fetch --depth=1 origin "$UPSTREAM_COMMIT" && \
      git checkout --detach --force "$UPSTREAM_COMMIT" && \
      test "$(git rev-parse HEAD)" = "$UPSTREAM_COMMIT" && \
      python -c 'import importlib.metadata as m; from pathlib import Path; packages = {d.metadata["Name"].lower(): d.version for d in m.distributions()}; names = ("torch", "torchvision", "torchaudio", "flash-attn", "natten", "torchcodec"); Path("/tmp/accelerator-constraints.txt").write_text("".join(f"{n}=={packages[n]}\n" for n in names if n in packages))' && \
      pip install --no-cache-dir --break-system-packages -c /tmp/accelerator-constraints.txt -r requirements.txt && \
      rm -rf ui/.next ui/dist && \
      cd ui && npm ci && ./node_modules/.bin/prisma generate && npm run build && \
      install -m 755 /app/ai-toolkit/docker/start.sh /start.sh; \
    fi
LABEL io.github.delfianto.ai-toolkit.upstream-revision="${UPSTREAM_COMMIT}"
ENV AI_TOOLKIT_UPSTREAM_COMMIT=${UPSTREAM_COMMIT}

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

# Catch changes to the common upstream integration APIs before publishing.
RUN python -m aitoolkit_models --help > /dev/null \
    && python -c 'from toolkit.config_modules import ModelConfig; from toolkit.models.v2._mixin import OstrisModelMixin; from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder; from toolkit.models.v2.vae.qwen_image import QwenImageVAE; assert callable(OstrisModelMixin.resolve_comfy_weights); assert callable(Qwen3VLTextEncoder.load_model); assert callable(QwenImageVAE.load_model)'
