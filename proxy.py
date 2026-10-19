"""Forward Google API bodies unchanged to Vertex using Application Default Credentials."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import re
from dataclasses import dataclass

import aiohttp
from aiohttp import web
import google.auth
from google.auth.transport.requests import Request
import requests
from yarl import URL

LOG = logging.getLogger("vertex-adc-router")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "proxy-connection",
}
REQUEST_HEADERS = {
    "accept", "content-type", "content-encoding", "x-goog-api-client",
    "x-goog-api-version",
}
MODEL_ROUTE = re.compile(
    r"/(v1|v1beta|v1beta1)/models/([A-Za-z0-9._-]+):"
    r"(generateContent|streamGenerateContent|countTokens)"
)


def explicit_content_roles(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return body
    if not isinstance(payload, dict):
        return body
    contents = payload.get("contents", [])
    contents = [contents] if isinstance(contents, dict) else contents
    if not isinstance(contents, list):
        return body
    missing = [content for content in contents if isinstance(content, dict) and "role" not in content]
    if not missing:
        return body
    for content in missing:
        content["role"] = "user"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Target:
    project: str
    location: str = "global"

    def __post_init__(self):
        for value in (self.project, self.location):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError("Project and location must be plain identifiers")

    def url(self, path: str) -> str:
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        elif self.location in {"us", "eu"}:
            host = f"aiplatform.{self.location}.rep.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"
        parent = f"projects/{self.project}/locations/{self.location}"
        if match := MODEL_ROUTE.fullmatch(path):
            version, model, action = match.groups()
            version = "v1" if version == "v1" else "v1beta1"
            suffix = f"{version}/{parent}/publishers/google/models/{model}:{action}"
        else:
            raise web.HTTPNotFound(text="Unsupported Google API route")
        return f"https://{host}/{suffix}"


class ADC:
    def __init__(self, credentials):
        self.credentials = credentials
        self.lock = asyncio.Lock()
        self.session = requests.Session()
        self.request = functools.partial(Request(session=self.session), timeout=30)

    async def headers(self, method: str, url: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        async with self.lock:
            refresh = asyncio.create_task(asyncio.to_thread(
                self.credentials.before_request, self.request, method, url, headers
            ))
            try:
                await asyncio.shield(refresh)
            except asyncio.CancelledError:
                await refresh
                raise
        return headers

    def close(self):
        self.session.close()


def create_app(target: Target, adc: ADC) -> web.Application:
    async def upstream_session(app):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
        async with aiohttp.ClientSession(
            timeout=timeout, auto_decompress=False, trust_env=True,
            skip_auto_headers={"Accept-Encoding", "User-Agent", "Content-Type"},
        ) as client:
            app[client_key] = client
            try:
                yield
            finally:
                adc.close()

    async def health(request):
        return web.json_response({"status": "ok", "project": target.project, "location": target.location})

    async def forward(request):
        if request.headers.get("Origin"):
            raise web.HTTPForbidden(text="Browser origins are not supported")
        url = target.url(request.path)
        if request.method != "POST":
            raise web.HTTPMethodNotAllowed(request.method, {"POST"})
        query = [(k, v) for k, v in request.query.items()
                 if k.lower() not in {"key", "api_key", "auth_token", "access_token"}]
        url = str(URL(url).with_query(query))
        connection_headers = {x.strip().lower() for x in request.headers.get("Connection", "").split(",")}
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() in REQUEST_HEADERS and k.lower() not in connection_headers}
        try:
            headers.update(await adc.headers(request.method, url))
        except Exception as exc:
            LOG.error("ADC authentication failed (%s)", type(exc).__name__)
            raise web.HTTPBadGateway(text="ADC authentication failed. Run gcloud auth application-default login.") from None
        body = await request.read()
        if not request.headers.get("Content-Encoding"):
            body = explicit_content_roles(body)
        response = None
        try:
            async with request.app[client_key].request(
                request.method, url, headers=headers, data=body, allow_redirects=False,
            ) as upstream:
                excluded = HOP_HEADERS | {
                    x.strip().lower() for x in upstream.headers.get("Connection", "").split(",")
                }
                response_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded]
                response = web.StreamResponse(status=upstream.status, headers=response_headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, TimeoutError) as exc:
            LOG.warning("Vertex connection failed (%s)", type(exc).__name__)
            if response is not None and response.prepared:
                if request.transport:
                    request.transport.close()
                return response
            raise web.HTTPBadGateway(text="Vertex connection failed") from None

    client_key = web.AppKey("upstream", aiohttp.ClientSession)
    app = web.Application(client_max_size=32 * 1024 * 1024, handler_args={"auto_decompress": False})
    app.cleanup_ctx.append(upstream_session)
    app.router.add_get("/healthz", health)
    app.router.add_route("*", "/{path:.*}", forward)
    return app


async def configured_app(project: str | None, location: str) -> web.Application:
    credentials, detected_project = await asyncio.to_thread(google.auth.default, scopes=SCOPES)
    project = project or detected_project
    if not project:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT or pass --project")
    target = Target(project, location)
    adc = ADC(credentials)
    try:
        await adc.headers("POST", target.url("/v1beta/models/gemini-4-argon:generateContent"))
    except BaseException:
        adc.close()
        raise
    LOG.info("Vertex project=%s location=%s using ADC", project, location)
    return create_app(target, adc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    web.run_app(
        configured_app(args.project, args.location), host="127.0.0.1", port=args.port,
        access_log=None, handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
