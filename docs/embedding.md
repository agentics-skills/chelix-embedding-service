# Embedding provider

The service downloads `MODEL` on first start, quantizes the weights to Q8_0, and writes UQFF artifacts under `CACHE_DIR/uqff/<sha256 of the model id>`. Later starts load those artifacts.

`HUGGINGFACE_API_KEY` is sent to Hugging Face for that download. A GGUF path is refused. The listen address is `LISTEN_IP:LISTEN_PORT` from the environment. There is no default address and no default model.

`POST /v1/embed` accepts a JSON body of at most 1 MiB:

```json
{ "text": "hello", "priority": 0 }
```

The response is `{ "embedding": [/* f32 */] }` or `{ "error": "..." }`. `GET /health` returns the loaded model metadata. Both routes require `Authorization: Bearer <API_KEY>`.

The process stops on SIGINT or SIGTERM. All logs go to stdout.
