from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Ranking
# ---------------------------
def compute_rankings(matches):
    scores = {}

    for m in matches:
        w1 = str(m["wrestlerA"]).strip()
        w2 = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()

        scores.setdefault(w1, 0)
        scores.setdefault(w2, 0)

        if winner == w1:
            scores[w1] += 1
        else:
            scores[w2] += 1

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------
# Build History
# ---------------------------
def build_history(matches):
    history = {}

    for m in matches:
        w1 = m["wrestlerA"]
        w2 = m["wrestlerB"]
        winner = m["winner"]

        history.setdefault(w1, {"wins": [], "losses": []})
        history.setdefault(w2, {"wins": [], "losses": []})

        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w2]["losses"].append(w1)
        else:
            history[w2]["wins"].append(w1)
            history[w1]["losses"].append(w2)

    return history


# ---------------------------
# Confidence Score
# ---------------------------
def build_confidence(seeds, history):
    confidence = {}

    for i, (w, score) in enumerate(seeds):
        wins = len(history[w]["wins"])
        losses = len(history[w]["losses"])

        base = wins / (wins + losses) if (wins + losses) > 0 else 0

        gap = 0
        if i < len(seeds) - 1:
            gap = score - seeds[i+1][1]

        confidence[w] = round((base * 70 + gap * 30), 1)

    return confidence


# ---------------------------
# Reasons
# ---------------------------
def build_reasons(seeds, history):
    reasons = {}

    for i in range(len(seeds)):
        w1 = seeds[i][0]
        reasons[w1] = []

        wins = len(history[w1]["wins"])
        losses = len(history[w1]["losses"])

        if losses == 0:
            reasons[w1].append(f"Undefeated ({wins}-0)")

        for j in range(i+1, len(seeds)):
            w2 = seeds[j][0]

            if w2 in history[w1]["wins"]:
                reasons[w1].append(f"Beat {w2} head-to-head")

        if not reasons[w1]:
            reasons[w1].append("Higher overall performance score")

    return reasons


# ---------------------------
# Controversy Detection
# ---------------------------
def detect_controversy(seeds):
    alerts = []

    for i in range(len(seeds)-1):
        if abs(seeds[i][1] - seeds[i+1][1]) <= 1:
            alerts.append(f"Close call between #{i+1} and #{i+2}")

    return alerts


# ---------------------------
# Compare Wrestlers
# ---------------------------
def compare(w1, w2, history):
    result = {}

    w1_wins = len(history[w1]["wins"])
    w2_wins = len(history[w2]["wins"])

    result["w1_record"] = w1_wins
    result["w2_record"] = w2_wins

    result["head_to_head"] = w2 in history[w1]["wins"]

    common = set(history[w1]["wins"] + history[w1]["losses"]) & \
             set(history[w2]["wins"] + history[w2]["losses"])

    result["common"] = list(common)

    result["recommended"] = w1 if w1_wins >= w2_wins else w2

    return result


# ---------------------------
# MAIN
# ---------------------------
@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):

    if file:
    contents = await file.read()
    from io import StringIO
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    matches = df.to_dict(orient="records")
    else:
        data = await request.json()
        matches = data.get("matches", [])

    for m in matches:
        m["weight"] = m.get("weight", "unknown")

    results = {}

    df = pd.DataFrame(matches)

    for weight, group in df.groupby("weight"):

        group_matches = group.to_dict(orient="records")

        cleaned = []
        for m in group_matches:
            cleaned.append({
                "wrestlerA": str(m["wrestlerA"]).strip(),
                "wrestlerB": str(m["wrestlerB"]).strip(),
                "winner": str(m["winner"]).strip()
            })

        seeds = compute_rankings(cleaned)
        history = build_history(cleaned)
        reasons = build_reasons(seeds, history)
        confidence = build_confidence(seeds, history)
        alerts = detect_controversy(seeds)

        results[str(weight)] = {
            "seeds": [(i+1, w[0], w[1]) for i,w in enumerate(seeds)],
            "history": history,
            "reasons": reasons,
            "confidence": confidence,
            "alerts": alerts
        }

    return JSONResponse(results)
