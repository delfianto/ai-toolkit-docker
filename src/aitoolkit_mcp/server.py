"""Authenticated Streamable HTTP entrypoint for the Toolkit adapter."""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2 as httpx
import uvicorn
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .adapter import ToolkitAPI, create_server


class BearerAuth:
    """Authenticate every MCP request, including initialize, SSE and session deletion."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = ("Bearer " + token).encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] != "/health":
            headers = dict(scope["headers"])
            if not hmac.compare_digest(
                headers.get(b"authorization", b""), self.expected
            ):
                response = JSONResponse(
                    {"error": "Bearer token required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(api: ToolkitAPI, token: str, allowed_hosts: list[str]) -> ASGIApp:
    server = create_server(api)
    app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        host="0.0.0.0",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=allowed_hosts
        ),
    )
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Starlette) -> AsyncIterator[None]:
        async with api.http, original_lifespan(application):
            yield

    async def health(_request: Request) -> JSONResponse:
        try:
            await api.request("jobs")
        except (httpx.HTTPError, ToolError):
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    app.router.lifespan_context = lifespan
    app.routes.append(Route("/health", health))
    return BearerAuth(app, token)


def main() -> None:
    token = (
        os.environ.get("AI_TOOLKIT_AUTH")
        or Path(os.environ.get("TOOLKIT_AUTH_FILE", "/run/secrets/AI_TOOLKIT_AUTH"))
        .read_text()
        .strip()
    )
    if not token:
        raise ValueError("A nonempty AI_TOOLKIT_AUTH token is required")
    url = os.environ.get("TOOLKIT_API_URL", "http://127.0.0.1:8675/api/")
    # No inherited proxies: this connection remains within the container.
    http = httpx.AsyncClient(
        base_url=url,
        headers={"Authorization": "Bearer " + token},
        timeout=30,
        trust_env=False,
    )
    api = ToolkitAPI(http)
    hosts = os.environ.get(
        "TOOLKIT_MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*,ai-toolkit:*"
    ).split(",")
    app = create_app(api, token, hosts)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("TOOLKIT_MCP_PORT", "8676")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
