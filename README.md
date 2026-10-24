---
title: EssentialRoute GeoAI Assistant
emoji: 🌧️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# EssentialRoute — Flood-Aware Mumbai Navigation Assistant

A GeoAI assistant that answers flood and route safety questions for Mumbai commuters using:

- **Live rain data** from [Open-Meteo](https://open-meteo.com/) (no API key needed)
- **[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** open-weights model (1.5B parameters)
- **11 flood hotspots** across Mumbai with real-time severity classification

## API

### POST `/ask`
```json
{
  "question": "Is it safe to drive from Dadar to Andheri right now?"
}
```

**Response:**
```json
{
  "answer": "Based on current conditions, Andheri Subway is waterlogged...",
  "flood_zones": [...],
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "timestamp": "2026-10-25T10:00:00+00:00"
}
```

### GET `/health`
Returns `{"status": "ok", "model": "Qwen/Qwen2.5-1.5B-Instruct"}`

## Built for
IIT Bombay GeoAI Hackathon 2026 — companion API for the [EssentialRoute PWA](https://essentialroute.netlify.app).
