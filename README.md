# Vertex ADC router

A lightweight local proxy that routes Gemini API requests to Google Vertex AI using Application Default Credentials. No Google API keys or service account private keys required.

```text
Gemini API client -> localhost:8765 -> Vertex AI
```

The proxy handles ADC authentication and URL routing. Streaming responses pass through unchanged.

## Setup

Requires Python 3.11+, uv, gcloud, and a Google Cloud project with Vertex AI enabled, billing, and permission to invoke models.

```sh
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Skip authentication if you already have working ADC.

## Run

Double-click `tui.bat` on Windows, or run `sh tui.sh` on macOS or Linux. The launchers install dependencies through uv and open the terminal menu.

Use **S** to start, **T** to stop, **R** to restart, **L** for logs, and **Q** to stop and quit.

To run directly:

```sh
uv run python proxy.py --project YOUR_PROJECT_ID --location global --port 8765
```

The launchers accept the same options. Defaults are `127.0.0.1:8765`, the ADC project, and location `global`. Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` to override the project and location. Menu logs go to `tmp/logs/proxy.log`.

## Usage

Send Gemini API requests to the local endpoint:

```sh
curl http://127.0.0.1:8765/v1beta/models/MODEL_ID:generateContent \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'
```

Replace `MODEL_ID` with a Gemini model available in your Vertex AI project. Client API keys are unnecessary and are stripped before forwarding.

Supports `generateContent`, `streamGenerateContent`, and `countTokens` under `/v1`, `/v1beta`, and `/v1beta1`. Omitted content roles default to `user`.

## Tests

```sh
uv sync --dev
uv run pytest -q
```

With the proxy running, `uv run python scripts/verify.py --model MODEL_ID` checks streaming and tool calls against Vertex AI. Live checks incur model charges. Reports go in gitignored `tmp/`.

## License

[Apache 2.0](LICENSE).
