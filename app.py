"""
EssentialRoute GeoAI Assistant
Flood-aware Mumbai navigation assistant — uses HF Inference API (no local model).

POST /ask  {"question": "Is it safe to drive from Dadar to Andheri now?"}
GET  /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from datetime import datetime, timezone

app = FastAPI(title="EssentialRoute GeoAI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}/v1/chat/completions"

# ---------------------------------------------------------------------------
# Flood hotspots
# ---------------------------------------------------------------------------
HOTSPOTS = [
    {"name": "Hindmata Junction",      "lat": 19.0136, "lon": 72.8317},
    {"name": "Sion Underpass",         "lat": 19.0422, "lon": 72.8620},
    {"name": "Milan Subway Santacruz", "lat": 19.0763, "lon": 72.8388},
    {"name": "Andheri Subway",         "lat": 19.1136, "lon": 72.8452},
    {"name": "Malad Link Road",        "lat": 19.1864, "lon": 72.8476},
    {"name": "Kurla LBS Road",         "lat": 19.0728, "lon": 72.8826},
    {"name": "Chembur Naka",           "lat": 19.0618, "lon": 72.9003},
    {"name": "Dharavi Cross Road",     "lat": 19.0422, "lon": 72.8554},
    {"name": "Wadala Truck Terminal",  "lat": 19.0186, "lon": 72.8639},
    {"name": "Bhandup Pumping Stn",   "lat": 19.1441, "lon": 72.9358},
    {"name": "Vikhroli Pipe Road",     "lat": 19.1086, "lon": 72.9296},
]
SUBWAY_NAMES = {"Milan Subway Santacruz", "Andheri Subway", "Sion Underpass"}

# ---------------------------------------------------------------------------
# Live rain from Open-Meteo
# ---------------------------------------------------------------------------
async def fetch_rain() -> list[dict]:
    lats = ",".join(str(h["lat"]) for h in HOTSPOTS)
    lons = ",".join(str(h["lon"]) for h in HOTSPOTS)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&hourly=precipitation&past_days=0&forecast_days=1"
        "&timeformat=unixtime&timezone=Asia%2FKolkata"
    )
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    results = []
    now_ts = datetime.now(timezone.utc).timestamp()
    items = data if isinstance(data, list) else [data]
    for i, item in enumerate(items):
        h = item.get("hourly", {})
        times = h.get("time", [])
        precip = h.get("precipitation", [])
        idx = next((j for j, t in enumerate(times) if t >= now_ts), len(times) - 1)
        now_mm = precip[idx] if idx < len(precip) else 0
        last3 = sum(precip[max(0, idx - 3):idx]) if precip else 0
        spot = HOTSPOTS[i]
        is_subway = spot["name"] in SUBWAY_NAMES
        if is_subway:
            sev = "severe" if (last3 > 5 or now_mm > 2) else "moderate" if (last3 > 1 or now_mm > 0.5) else "clear"
        else:
            sev = "severe" if (last3 > 20 or now_mm > 7) else "moderate" if (last3 > 8 or now_mm > 3) else "clear"
        results.append({"name": spot["name"], "now_mm_h": round(now_mm, 1), "last3h_mm": round(last3, 1), "severity": sev})
    return results

def flood_summary(zones: list[dict]) -> str:
    severe   = [z["name"] for z in zones if z["severity"] == "severe"]
    moderate = [z["name"] for z in zones if z["severity"] == "moderate"]
    lines = []
    if severe:   lines.append(f"FLOODED (avoid): {', '.join(severe)}")
    if moderate: lines.append(f"WATERLOGGED (caution): {', '.join(moderate)}")
    if not lines: lines.append("All 11 monitored zones are currently clear.")
    return "\n".join(lines)

# -------------------------------------------------------------------