"""
EssentialRoute GeoAI Assistant
Flood-aware Mumbai navigation assistant.

POST /ask    {"question": "...", "app_state": {...optional flood state from the app...}}
GET  /health
GET  /debug  -> shows whether the LLM and rain data are reachable (last errors)
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

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MODEL_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct").strip()
# Hugging Face's current OpenAI-compatible router (the old api-inference.huggingface.co host is retired).
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
# Tried in order: HF picks a provider automatically, then Featherless explicitly.
MODEL_CANDIDATES = [MODEL_ID] + ([f"{MODEL_ID}:featherless-ai"] if ":" not in MODEL_ID else [])

LAST = {"llm_error": None, "llm_model": None, "rain_error": None, "rain_source": None}

# ---------------------------------------------------------------------------
# Flood hotspots (same 11 spots as the EssentialRoute app)
# ---------------------------------------------------------------------------
HOTSPOTS = [
    {"name": "Hindmata, Dadar East",          "lat": 19.0067, "lon": 72.8437, "subway": False},
    {"name": "King's Circle / Gandhi Market", "lat": 19.0285, "lon": 72.8566, "subway": False},
    {"name": "Sion (Road No. 24 area)",       "lat": 19.0405, "lon": 72.8618, "subway": False},
    {"name": "Chunabhatti",                   "lat": 19.0517, "lon": 72.8695, "subway": False},
    {"name": "Kalanagar, Bandra East",        "lat": 19.0603, "lon": 72.8494, "subway": False},
    {"name": "Khar Subway",                   "lat": 19.0712, "lon": 72.8402, "subway": True},
    {"name": "Kurla (LBS Marg area)",         "lat": 19.0742, "lon": 72.8846, "subway": False},
    {"name": "Milan Subway, Santacruz",       "lat": 19.0826, "lon": 72.8418, "subway": True},
    {"name": "Andheri Subway",                "lat": 19.1189, "lon": 72.8467, "subway": True},
    {"name": "Malad Subway",                  "lat": 19.1868, "lon": 72.8484, "subway": True},
    {"name": "Dahisar Subway",                "lat": 19.2502, "lon": 72.8594, "subway": True},
]
ORDER = [None, "low", "moderate", "high", "extreme"]


def risk_for(now_mm: float, last3: float, subway: bool):
    """Same thresholds as the app (Mumbai drains ~25 mm/h design capacity); subways one level worse."""
    if now_mm >= 50 or last3 >= 120: s = "extreme"
    elif now_mm >= 25 or last3 >= 60: s = "high"
    elif now_mm >= 10 or last3 >= 25: s = "moderate"
    elif now_mm >= 4 or last3 >= 10: s = "low"
    else: s = None
    if s and subway:
        s = ORDER[min(len(ORDER) - 1, ORDER.index(s) + 1)]
    return s


async def fetch_rain():
    """Live rain from Open-Meteo. Returns (zones, ok). zones lists every hotspot with severity or 'clear'."""
    lats = ",".join(str(h["lat"]) for h in HOTSPOTS)
    lons = ",".join(str(h["lon"]) for h in HOTSPOTS)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&hourly=precipitation&past_days=1&forecast_days=1"
        "&timeformat=unixtime&timezone=Asia%2FKolkata"
    )
    try:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "EssentialRoute/1.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        LAST["rain_error"] = f"{type(e).__name__}: {e}"
        return [], False

    items = data if isinstance(data, list) else [data]
    if len(items) != len(HOTSPOTS):
        LAST["rain_error"] = "unexpected Open-Meteo response"
        return [], False

    now_ts = datetime.now(timezone.utc).timestamp()
    zones = []
    for spot, item in zip(HOTSPOTS, items):
        h = item.get("hourly", {})
        times = h.get("time", [])
        precip = [p or 0 for p in h.get("precipitation", [])]
        if not times or not precip:
            continue
        # current hour = last timestamp not in the future
        idx = max((j for j, t in enumerate(times) if t <= now_ts), default=0)
        now_mm = precip[idx]
        last3 = sum(precip[max(0, idx - 2): idx + 1])
        sev = risk_for(now_mm, last3, spot["subway"]) or "clear"
        zones.append({"name": spot["name"], "now_mm_h": round(now_mm, 1), "last3h_mm": round(last3, 1), "severity": sev})
    LAST["rain_error"] = None if zones else "no rainfall values"
    return zones, bool(zones)


def zones_from_app(state: dict):
    """Use the flood state the app is showing (live or simulated rain) so answers match the map."""
    mode = state.get("mode")
    zones = []
    for z in state.get("zones") or []:
        sev = z.get("severity") or "moderate"
        zones.append({
            "name": str(z.get("label", "Unknown spot"))[:80],
            "now_mm_h": z.get("nowMmH"),
            "last3h_mm": z.get("last3hMm"),
            "severity": sev,
            "kind": z.get("kind", "rain"),
        })
    return zones, mode


def describe(zones, source_note: str) -> str:
    bad = [z for z in zones if z["severity"] in ("high", "extreme")]
    mid = [z for z in zones if z["severity"] in ("moderate", "low")]
    lines = [source_note]
    if bad: lines.append("IMPASSABLE / AVOID: " + "; ".join(f'{z["name"]} ({z["severity"]})' for z in bad))
    if mid: lines.append("WATERLOGGING RISK, SLOW / CAUTION: " + "; ".join(f'{z["name"]} ({z["severity"]})' for z in mid))
    if not bad and not mid: lines.append("No flood risk at any of the 11 monitored hotspots.")
    return "\n".join(lines)


def rule_answer(question: str, zones, have_data: bool) -> str:
    if not have_data:
        return ("Live rainfall data is unavailable right now, so I can't confirm road conditions. "
                "During rain, treat low-lying subways (Andheri, Milan, Khar, Malad) and Hindmata as risky "
                "and check the map before you travel.")
    q = question.lower()
    bad = [z for z in zones if z["severity"] in ("high", "extreme")]
    mid = [z for z in zones if z["severity"] in ("moderate", "low")]
    # if the question names a hotspot, answer about that one first
    generic = {"subway", "road", "area", "east", "west", "market", "circle", "marg", "junction"}
    def keywords(name):
        return [w for w in name.lower().replace(",", " ").replace("/", " ").replace("(", " ").replace(")", " ").split()
                if len(w) > 3 and w not in generic and not w.isdigit()]
    named = [z for z in zones if any(w in q for w in keywords(z["name"]))]
    parts = []
    for z in named[:2]:
        if z["severity"] in ("high", "extreme"):
            parts.append(f'{z["name"]} is at {z["severity"]} flood risk — avoid it.')
        elif z["severity"] in ("moderate", "low"):
            parts.append(f'{z["name"]} has {z["severity"]} waterlogging risk — expect delays.')
        else:
            parts.append(f'{z["name"]} looks clear right now.')
    other_bad = [z for z in bad if z not in named[:2]]
    other_mid = [z for z in mid if z not in named[:2]]
    if other_bad:
        parts.append(("Also avoid: " if named else "Avoid: ") + ", ".join(z["name"] for z in other_bad) + ".")
    if other_mid:
        parts.append("Go carefully near: " + ", ".join(z["name"] for z in other_mid) + ".")
    if not bad and not mid:
        parts.append("All 11 monitored flood hotspots are clear right now.")
    parts.append("EssentialRoute's map reroutes around flooded spots automatically.")
    return " ".join(parts)


async def ask_llm(system_prompt: str, question: str):
    if not HF_TOKEN:
        LAST["llm_error"] = "HF_TOKEN is not set on the server"
        return None, None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {HF_TOKEN}"}
    errors = []
    async with httpx.AsyncClient(timeout=40) as client:
        for model in MODEL_CANDIDATES:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 220,
                "temperature": 0.3,
            }
            try:
                r = await client.post(HF_ROUTER_URL, json=payload, headers=headers)
                if r.status_code != 200:
                    errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")
                    continue
                answer = r.json()["choices"][0]["message"]["content"].strip()
                if answer:
                    LAST["llm_error"] = None
                    LAST["llm_model"] = model
                    return answer, model
            except Exception as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")
    LAST["llm_error"] = " | ".join(errors)[:800]
    return None, None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None
    app_state: Optional[dict] = None


class AskResponse(BaseModel):
    answer: str
    flood_zones: list[dict]
    model: str
    answered_by: str      # "llm" or "rules"
    data_source: str      # where the flood data came from
    timestamp: str


@app.get("/")
async def root():
    return {"service": "EssentialRoute GeoAI Assistant", "endpoints": ["POST /ask", "GET /health", "GET /debug"]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/debug")
async def debug():
    zones, ok = await fetch_rain()
    return {
        "hf_token_set": bool(HF_TOKEN),
        "model_candidates": MODEL_CANDIDATES,
        "last_llm_model": LAST["llm_model"],
        "last_llm_error": LAST["llm_error"],
        "rain_ok": ok,
        "rain_error": LAST["rain_error"],
        "rain_sample": zones[:3],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question is required")
    question = question[:500]

    # 1) Prefer the flood state the app is showing; 2) otherwise fetch live rain ourselves.
    zones, have_data, source = [], False, ""
    state = req.app_state or {}
    mode = state.get("mode")
    if state and mode and mode != "off" and state.get("ok", True):
        zones, mode = zones_from_app(state)
        have_data = True
        source = ("Live Open-Meteo rainfall, as shown in the EssentialRoute app" if mode == "live"
                  else f"SIMULATED '{mode}' rain scenario selected in the app (not real-time)")
        flagged = {z["name"] for z in zones}
        zones += [{"name": h["name"], "severity": "clear"} for h in HOTSPOTS if h["name"] not in flagged]
    if not have_data:
        zones, have_data = await fetch_rain()
        source = "Live Open-Meteo rainfall (server)" if have_data else "unavailable"

    now = datetime.now(timezone.utc).astimezone().strftime("%d %b %Y %H:%M")
    if have_data:
        status = describe(zones, f"Data source: {source}. Time: {now}.")
    else:
        status = "Live rainfall data is UNAVAILABLE right now. Do not claim roads are clear."

    system_prompt = (
        "You are EssentialRoute, a flood-aware navigation assistant for Mumbai commuters.\n"
        "Answer using ONLY the flood status below. Never invent rainfall or flooding that is not listed.\n"
        "If the user's route passes an 'avoid' spot, say so and suggest a practical alternative road or flyover.\n"
        "If data is simulated, mention it is a simulated scenario. Keep it to 2-4 short sentences.\n\n"
        f"FLOOD STATUS (11 monitored hotspots):\n{status}"
    )

    answer, used_model = await ask_llm(system_prompt, question)
    answered_by = "llm" if answer else "rules"
    if not answer:
        answer = rule_answer(question, zones, have_data)

    return AskResponse(
        answer=answer,
        flood_zones=[z for z in zones if z.get("severity") not in (None, "clear")],
        model=used_model or MODEL_ID,
        answered_by=answered_by,
        data_source=source or "unavailable",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
