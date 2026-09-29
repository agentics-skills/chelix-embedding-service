# chelix-embedding-service

HTTP embedding provider. It loads a Hugging Face model, quantizes it to Q8, and serves embedding vectors.

The process exits if any required environment variable is missing, empty, or invalid. Logs are written to stdout.

## Required environment

| Variable | Purpose |
| --- | --- |
| `API_KEY` | Bearer token clients must send |
| `HUGGINGFACE_API_KEY` | Token used to download the model |
| `MODEL` | Hugging Face model id or snapshot directory |
| `CACHE_DIR` | Directory for the Hugging Face cache and Q8 artifacts |
| `LISTEN_IP` | Address to bind |
| `LISTEN_PORT` | TCP port to bind; `0` is rejected |
| `RUST_LOG` | Tracing filter |

## HTTP

`POST /v1/embed` and `GET /health` require `Authorization: Bearer <API_KEY>`.

```json
{ "text": "hello", "priority": 0 }
```

`priority` `0` is an index job. `priority` `1` is a search job and is served first.

See [docs/embedding.md](docs/embedding.md).
