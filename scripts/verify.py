"""Make billable model calls to verify streaming and two-turn tool use."""

import argparse
import asyncio
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


async def post(client, url, body, stream):
    async with client.post(url, json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1200]}")
        if not stream:
            return await response.json()
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Streaming request did not return SSE")
        frames = []
        data = []
        async for raw in response.content:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                payload = "\n".join(data)
                data.clear()
                if payload != "[DONE]":
                    frame = json.loads(payload)
                    if "error" in frame or frame.get("type") == "error":
                        raise RuntimeError(f"SSE error: {frame}")
                    frames.append(frame)
        if data:
            raise RuntimeError("Incomplete final SSE event")
        if not frames:
            raise RuntimeError("No SSE data events")
        return frames


async def gemini(client, base, model, stream):
    action = "streamGenerateContent?alt=sse" if stream else "generateContent"
    url = f"{base}/v1beta/models/{model}:{action}"
    declaration = {"name": "lookup_nonce", "description": "Get the verification nonce.",
                   "parameters": {"type": "OBJECT", "properties": {"tag": {"type": "STRING"}}, "required": ["tag"]}}
    contents = [{"role": "user", "parts": [{"text": 'Call lookup_nonce with tag "probe". Then reply with the nonce from its result.'}]}]
    body = {"contents": contents, "tools": [{"functionDeclarations": [declaration]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup_nonce"]}},
            "generationConfig": {"maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "LOW"}}}
    first = await post(client, url, body, stream)
    frames = first if stream else [first]
    parts = [part for frame in frames for candidate in frame.get("candidates", [])
             for part in candidate.get("content", {}).get("parts", [])]
    calls = [part["functionCall"] for part in parts if "functionCall" in part]
    if len(calls) != 1 or calls[0]["name"] != "lookup_nonce" or calls[0].get("args") != {"tag": "probe"}:
        raise RuntimeError(f"Wrong tool call: {calls}")
    nonce = secrets.token_hex(12)
    contents.extend([{"role": "model", "parts": parts},
                     {"role": "user", "parts": [{"functionResponse": {"name": "lookup_nonce", "response": {"nonce": nonce}}}]}])
    body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    second = await post(client, url, body, stream)
    final_frames = second if stream else [second]
    text = "".join(part.get("text", "") for frame in final_frames for candidate in frame.get("candidates", [])
                   for part in candidate.get("content", {}).get("parts", []) if not part.get("thought"))
    finishes = [candidate.get("finishReason") for frame in final_frames for candidate in frame.get("candidates", [])]
    if nonce not in text or "STOP" not in finishes:
        raise RuntimeError(f"Tool result did not complete correctly: {text!r}, finishes={finishes}")
    return {"tool": "lookup_nonce", "arguments_verified": True, "result_verified": True,
            "first_sse_events": len(frames) if stream else 0,
            "final_sse_events": len(final_frames) if stream else 0}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="gemini-4-argon")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "tmp" / "verification-results.json"))
    parser.add_argument("--timeout", type=float, default=120, help="Deadline in seconds for each small verification request")
    args = parser.parse_args()
    results = {"time_utc": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
               "model": args.model, "protocol": "gemini", "checks": {}}
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_connect=10, sock_read=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
        for stream in (False, True):
            name = "streaming_tool_roundtrip" if stream else "nonstream_tool_roundtrip"
            try:
                details = await gemini(client, args.base_url.rstrip("/"), args.model, stream)
                results["checks"][name] = {"status": "passed", **details}
                print(f"PASS {name}: {json.dumps(details)}", flush=True)
            except Exception as exc:
                results["checks"][name] = {"status": "failed", "error": str(exc)}
                print(f"FAIL {name}: {exc}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return all(result["status"] == "passed" for result in results["checks"].values())


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
