import asyncio
import gzip
import json
import threading
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
from google.auth.credentials import Credentials
from yarl import URL

from proxy import ADC, Target, create_app


class TestADC:
    __test__ = False

    async def headers(self, method, url):
        return {"Authorization": "Bearer adc-token", "x-goog-user-project": "quota-project"}

    def close(self):
        pass


@asynccontextmanager
async def relay(handler, adc=None):
    upstream = web.Application(handler_args={"auto_decompress": False})
    upstream.router.add_route("*", "/{path:.*}", handler)
    async with TestServer(upstream) as server:
        class LocalTarget(Target):
            def url(self, path):
                vertex_path = URL(super().url(path)).path
                return str(server.make_url(vertex_path))

        app = create_app(LocalTarget("test-project"), adc or TestADC())
        async with TestClient(TestServer(app, handler_cancellation=True)) as client:
            yield client


@pytest.mark.parametrize("path,location,expected", [
    ("/v1beta/models/gemini-4-argon:generateContent", "global",
     "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/publishers/google/models/gemini-4-argon:generateContent"),
    ("/v1/models/gemini-4-argon:countTokens", "us-central1",
     "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-4-argon:countTokens"),
    ("/v1beta1/models/gemini-4-argon:streamGenerateContent", "us",
     "https://aiplatform.us.rep.googleapis.com/v1beta1/projects/proj/locations/us/publishers/google/models/gemini-4-argon:streamGenerateContent"),

])
def test_vertex_urls(path, location, expected):
    assert Target("proj", location).url(path) == expected


@pytest.mark.parametrize("path", [
    "/v1/messages", "/v1beta/models/x:delete", "/v1beta/models/../x:generateContent",
    "/v1beta/interactions/../../metadata", "/https://evil.test/",
])
def test_rejects_unknown_routes(path):
    with pytest.raises(web.HTTPNotFound):
        Target("proj").url(path)


async def test_raw_body_credentials_query_and_compressed_response():
    payload = gzip.compress(b'{ "contents":[], "thoughtSignature":"opaque" }')
    returned = gzip.compress(b'{"error":{"message":"literal upstream failure"}}')

    async def upstream(request):
        assert request.path == "/v1beta1/projects/test-project/locations/global/publishers/google/models/gemini-4-argon:generateContent"
        assert await request.read() == payload
        assert request.headers["Authorization"] == "Bearer adc-token"
        assert request.headers["x-goog-user-project"] == "quota-project"
        assert request.headers["x-goog-api-version"] == "2026-09-01"
        assert request.headers["Content-Encoding"] == "gzip"
        for name in ("x-goog-api-key", "x-api-key", "cookie", "x-untrusted"):
            assert name not in request.headers
        assert list(request.query.items()) == [("alt", "sse"), ("tag", "a"), ("tag", "b")]
        return web.Response(status=429, body=returned, headers={
            "Content-Encoding": "gzip", "Content-Type": "application/json", "Retry-After": "7",
            "Connection": "x-private", "x-private": "drop",
        })

    async with relay(upstream) as client:
        response = await client.post(
            "/v1beta/models/gemini-4-argon:generateContent?key=placeholder&alt=sse&tag=a&tag=b&access_token=secret",
            data=payload, auto_decompress=False, headers={
                "Authorization": "Bearer client-secret", "x-goog-api-key": "placeholder",
                "x-api-key": "client-secret", "cookie": "session=secret", "x-untrusted": "drop",
                "x-goog-user-project": "wrong-project", "Content-Encoding": "gzip",
                "Content-Type": "application/json", "x-goog-api-version": "2026-09-01",
            },
        )
        assert response.status == 429
        assert response.headers["Retry-After"] == "7"
        assert response.headers["Content-Encoding"] == "gzip"
        assert "x-private" not in response.headers
        assert await response.read() == returned


@pytest.mark.parametrize("contents,expected", [
    ([{"parts": [{"text": "Hi"}]}], [{"role": "user", "parts": [{"text": "Hi"}]}]),
    ({"parts": [{"text": "Hi"}]}, {"role": "user", "parts": [{"text": "Hi"}]}),
    ([{"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {}}, "thoughtSignature": "opaque"}]},
      {"parts": [{"functionResponse": {"name": "lookup", "response": {"value": 7}}}]}],
     [{"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {}}, "thoughtSignature": "opaque"}]},
      {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 7}}}]}]),
])
async def test_omitted_roles_use_the_google_user_default(contents, expected):
    async def upstream(request):
        assert await request.json() == {"contents": expected, "generationConfig": {"maxOutputTokens": 8}}
        return web.json_response({"candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}}]})

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent",
                                     json={"contents": contents, "generationConfig": {"maxOutputTokens": 8}})
        assert response.status == 200
        assert await response.json() == {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}}]}


@pytest.mark.parametrize("body", [
    b'{ "contents": [{"role":"user","parts":[{"text":"Hi"}]}] }',
    b'{ "contents": [{"role":"assistant","parts":[{"text":"Hi"}]}] }',
    b'not-json',
])
async def test_explicit_roles_and_invalid_payloads_are_forwarded_unchanged(body):
    async def upstream(request):
        assert await request.read() == body
        return web.Response(status=400, text="upstream error")

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent", data=body)
        assert response.status == 400
        assert await response.text() == "upstream error"


async def test_sse_arrives_before_upstream_completes():
    release = asyncio.Event()
    first = b'event: content.delta\ndata: {"delta":{"text":"first"}}\n\n'
    last = b'event: interaction.complete\ndata: {"status":"completed"}\n\n'

    async def upstream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(first)
        await release.wait()
        await response.write(last)
        return response

    async with relay(upstream) as client:
        async with client.post("/v1beta/models/gemini-4-argon:generateContent", data=b'{"stream":true}') as response:
            try:
                assert await asyncio.wait_for(response.content.readexactly(len(first)), 3) == first
                assert not release.is_set()
            finally:
                release.set()
            assert await response.read() == last


async def test_client_cancellation_closes_upstream():
    closed = asyncio.Event()

    async def upstream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        try:
            while request.transport is not None and not request.transport.is_closing():
                await asyncio.sleep(0.01)
        finally:
            closed.set()
        return response

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse", data=b"{}")
        assert await response.content.readexactly(13) == b"data: first\n\n"
        response.close()
        await asyncio.wait_for(closed.wait(), 3)


async def test_auth_failure_never_reaches_upstream():
    class BrokenADC(TestADC):
        async def headers(self, method, url):
            raise RuntimeError("private credential detail")

    async def upstream(request):
        pytest.fail("request with failed ADC must not reach Vertex")

    async with relay(upstream, BrokenADC()) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent", data=b"{}")
        assert response.status == 502
        assert "private credential detail" not in await response.text()


async def test_loopback_browser_origin_and_method_guard():
    async def upstream(request):
        pytest.fail("invalid request must not reach upstream")

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent", headers={"Origin": "https://evil.test"})
        assert response.status == 403
        response = await client.get("/v1beta/models/gemini-4-argon:generateContent")
        assert response.status == 405
        response = await client.post("/v1/messages")
        assert response.status == 404
        response = await client.get("/healthz")
        assert await response.json() == {"status": "ok", "project": "test-project", "location": "global"}


async def test_concurrent_requests_share_one_adc_refresh():
    class ExpiredCredentials(Credentials):
        def __init__(self):
            super().__init__()
            self.refreshes = 0

        def refresh(self, request):
            self.refreshes += 1
            self.token = "refreshed-token"
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

    credentials = ExpiredCredentials()
    adc = ADC(credentials)
    try:
        headers = await asyncio.gather(*[adc.headers("POST", "https://aiplatform.googleapis.com/") for _ in range(8)])
        assert headers == [{"authorization": "Bearer refreshed-token"}] * 8
        assert credentials.refreshes == 1
    finally:
        adc.close()


async def test_upstream_redirect_is_returned_without_following():
    async def upstream(request):
        return web.Response(status=307, headers={"Location": "https://example.invalid/steal-token"})

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent", data=b"{}", allow_redirects=False)
        assert response.status == 307
        assert response.headers["Location"] == "https://example.invalid/steal-token"


async def test_disconnect_during_adc_refresh_keeps_auth_serialized():
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class SlowCredentials:
        calls = 0

        def before_request(self, auth_request, method, url, headers):
            self.calls += 1
            if self.calls == 1:
                loop.call_soon_threadsafe(started.set)
                if not release.wait(3):
                    raise TimeoutError("test did not release refresh")
            headers["authorization"] = "Bearer refreshed-token"

    credentials = SlowCredentials()
    adc = ADC(credentials)
    first = asyncio.create_task(adc.headers("POST", "https://aiplatform.googleapis.com/"))
    try:
        await asyncio.wait_for(started.wait(), 3)
        first.cancel()
        second = asyncio.create_task(adc.headers("POST", "https://aiplatform.googleapis.com/"))
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert credentials.calls == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == {"authorization": "Bearer refreshed-token"}
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        adc.close()


async def test_broken_upstream_stream_does_not_become_success():
    async def upstream(request):
        response = web.StreamResponse(headers={"Content-Length": "100", "Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: partial\n\n")
        request.transport.close()
        return response

    async with relay(upstream) as client:
        response = await client.post("/v1beta/models/gemini-4-argon:generateContent", data=b"{}")
        with pytest.raises(aiohttp.ClientPayloadError):
            await response.read()
