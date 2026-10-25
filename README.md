# EssentialRoute — GeoAI Assistant API

Flood-aware route-safety assistant for Mumbai commuters. Backend for the **Ask AI** tab in the [EssentialRoute app](https://essentialroute.vercel.app) ([frontend source](https://github.com/Arush-prog/essentialroute)).

- **Live endpoint:** `POST https://essentialroute-api.onrender.com/ask`
- **Model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — open weights, 1.5B parameters, Apache 2.0
- **Data:** live hourly rainfall from [Open-Meteo](https://open-meteo.com/) (no API key) at 11 Mumbai flood hotspots

## How it works

1. Fetches current and last-3-hour rainfall for 11 chronic waterlogging spots (Hindmata, Sion, Milan Subway, Andheri Subway, Kurla, Chembur and more).
2. Classifies each spot as clear / moderate / severe. Subways use lower thresholds because they flood first.
3. Sends the user's question to Qwen2.5-1.5B-Instruct with the live flood status as context, so answers name specific flooded junctions and suggest alternatives.
4. If the model is unreachable, falls back to a rule-based answer from the same flood data, so the endpoint always responds.

## API

### `POST /ask`

```bash
curl -X POST https://essentialroute-api.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is Andheri Subway safe right now?"}'
```

Response:

```json
{
  "answer": "Andheri Subway is waterlogged right now — avoid it...",
  "flood_zones": [{"name": "Andheri Subway", "now_mm_h": 3.1, "last3h_mm": 7.4, "severity": "severe"}],
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "timestamp": "2026-10-25T10:00:00+00:00"
}
```

### `GET /health`

Returns `{"status": "ok", "model": "Qwen/Qwen2.5-1.5B-Instruct"}`.

> Hosted on a free tier that sleeps when idle — the first request after a pause can take up to a minute.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 10000
```

Optional: set `HF_TOKEN` for higher Hugging Face Inference API rate limits.

## Stack

FastAPI · Uvicorn · httpx · Docker · Render

Built for the IIT Bombay GeoAI Hackathon 2026.
