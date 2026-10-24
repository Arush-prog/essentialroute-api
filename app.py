"""
EssentialRoute GeoAI Assistant — Hugging Face Spaces endpoint
Flood-aware Mumbai navigation assistant powered by Qwen2.5-1.5B-Instruct.

POST /ask  {"question": "Is it safe to drive from Dadar to Andheri now?"}
GET  /health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import json
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI(title="EssentialRoute GeoAI Assistant")

# ---------------------------------------------------------------------------
# Flood hotspots (matches EssentialRoute frontend)
# ---------------------------------------------------------------------------
HOTSPOTS = [
    {"name": "Hindmata Junction",     "lat": 19.0136, "lon": 72.8317},
    {"name": "Sion Underpass",        "lat": 19.0422, "lon": 72.8620},
    {"name": "Milan Subway Santacruz","lat": 19.0763, "lon": 72.8388},
    {"name": "Andheri Subway",        "lat": 19.1136, "lon": 72.8452},
    {"name": "Malad Link Road",       "lat": 19.1864, "lon": 72.8476},
    {"name": "Kurla LBS Road",        "lat": 19.0728, "lon": 72.8826},
    {"name": "Chembur Naka",          "lat": 19.0618, "lon": 72.9003},
    {"name": "Dharavi Cross Road",    "lat": 19.0422, "lon": 72.8554},
    {"name": "Wadala Truck Terminal", "lat": 19.0186, "lon": 72.8639},
    {"name": "Bhandup Pumping Stn",  "lat": 19.1441, "lon": 72.9358},
    {"name": "Vikhroli Pipe Road",    "lat": 19.1086, "lon": 72.9296},
]

SUBWAY_NAMES = {"Milan Subway Santacruz", "Andheri Subway", "Sion Underpass"}

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {MODEL_ID}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3,
    do_sample=True,
)
print("Model ready.")

# ---------------------------------------------------------------------------
# Live rain from Open-Meteo (same source as the PWA)
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
    except Exception as e:
        return []

    results = []
    now_ts = datetime.now(timezone.utc).timestamp()
    items = data if isinstance(data, list) else [data]

    for i, item in enumerate(items):
        h = item.get("hourly", {})
        times = h.get("time", [])
        precip = h.get("precipitation", [])
        # find current hour
        idx = next((j for j, t in enumerate(times) if t >= now_ts), len(times) - 1)
        now_mm = precip[idx] if idx < len(precip) else 0
        last3 = sum(precip[max(0, idx-3):idx]) if precip else 0
        spot = HOTSPOTS[i]
        # severity
        is_subway = spot["name"] in SUBWAY_NAMES
        if is_subway:
            if last3 > 5 or now_mm > 2: sev = "severe"
            elif last3 > 1 or now_mm > 0.5: sev = "moderate"
            else: sev = "clear"
        else:
            if last3 > 20 or now_mm > 7: sev = "severe"
            elif last3 > 8 or now_mm > 3: sev = "moderate"
            else: sev = "clear"
        results.append({
            "name": spot["name"],
            "now_mm_h": round(now_mm, 1),
            "last3h_mm": round(last3, 1),
            "severity": sev,
        })
    return results

def flood_summary(zones: list[dict]) -> str:
    severe  = [z["name"] for z in zones if z["severity"] == "severe"]
    moderate = [z["name"] for z in zones if z["severity"] == "moderate"]
    clear   = [z["name"] for z in zones if z["severity"] == "clear"]
    lines = []
    if severe:   lines.append(f"FLOODED (avoid): {', '.join(severe)}")
    if moderate: lines.append(f"WATERLOGGED (use caution): {', '.join(moderate)}")
    if clear:    lines.append(f"Clear: {', '.join(clear)}")
    return "\n".join(lines) if lines else "All zones clear."

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None   # optional extra context from caller

class AskResponse(BaseModel):
    answer: str
    flood_zones: list[dict]
    model: str
    timestamp: str

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    zones = await fetch_rain()
    summary = flood_summary(zones) if zones else "Live rain data unavailable — advise caution."

    system_prompt = f"""You are EssentialRoute, a flood-aware navigation assistant for Mumbai.
You help commuters decide the safest route based on real-time flood and waterlogging data.

Current flood status (live Open-Meteo data, {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}):
{summary}

Rules:
- Give a direct, practical answer in 3-5 sentences.
- Name specific flooded locations if relevant to the user's route.
- If a key junction is flooded, suggest avoiding it and taking an alternate road.
- Do not invent flood data beyond what is listed above.
- Keep the tone calm and helpful.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": req.question},
    ]

    try:
        out = pipe(messages)
        answer = out[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    return AskResponse(
        answer=answer,
        flood_zones=zones,
        model=MODEL_ID,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
