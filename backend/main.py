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

@app.get("/")
def root():
    return {"status": "ok"}

# ---------------------------
# History
# ---------------------------
def build_history(matches):
    history = {}
    for m in matches:
        w1, w2, winner = m["wrestlerA"], m["wrestlerB"], m["winner"]

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
# SOS
# ---------------------------
def build_sos(history):
    sos = {}
    for w in history:
        opps = history[w]["wins"] + history[w]["losses"]
        if not opps:
            sos[w] = 0
            continue

        vals = []
        for o in opps:
            wins = len(history[o]["wins"])
            losses = len(history[o]["losses"])
            total = wins + losses
            if total:
                vals.append(wins / total)

        sos[w] = round(sum(vals)/len(vals),3) if vals else 0

    return sos

# ---------------------------
# Ranking (IMPROVED)
# ---------------------------
def compute_rankings(history, sos):
    scores = {}

    for w in history:
        wins = len(history[w]["wins"])
        losses = len(history[w]["losses"])
        total = wins + losses

        pct = wins/total if total else 0

        scores[w] = (
            wins * 1.0 +
            pct * 10 +
            sos[w] * 10
        )

    # head-to-head boost
    for w in history:
        for opp in history[w]["wins"]:
            scores[w] += 0.5

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------
# Common Opponents
# ---------------------------
def build_common(history):
    common = {}
    for w1 in history:
        common[w1] = {}
        for w2 in history:
            if w1 == w2:
                continue

            c = set(history[w1]["wins"] + history[w1]["losses"]) & \
                set(history[w2]["wins"] + history[w2]["losses"])

            rows = []
            for opp in c:
                rows.append({
                    "opponent": opp,
                    "w1": "W" if opp in history[w1]["wins"] else "L",
                    "w2": "W" if opp in history[w2]["wins"] else "L"
                })

            common[w1][w2] = rows

    return common

# ---------------------------
# Compare Score
# ---------------------------
def compare_score(w1, w2, history, sos):
    score = 0

    # HEAD TO HEAD (strongest)
    if w2 in history[w1]["wins"]:
        score += 5
    elif w1 in history[w2]["wins"]:
        score -= 5

    # COMMON
    common = set(history[w1]["wins"] + history[w1]["losses"]) & \
             set(history[w2]["wins"] + history[w2]["losses"])

    for c in common:
        if c in history[w1]["wins"] and c in history[w2]["losses"]:
            score += 1
        elif c in history[w2]["wins"] and c in history[w1]["losses"]:
            score -= 1

    # SOS
    score += (sos[w1] - sos[w2]) * 5

    # RECORD
    r1 = len(history[w1]["wins"]) - len(history[w1]["losses"])
    r2 = len(history[w2]["wins"]) - len(history[w2]["losses"])
    score += (r1 - r2) * 0.1

    return score

# ---------------------------
# Confidence
# ---------------------------
def build_confidence(seeds, history, sos):
    conf = {}

    for i in range(len(seeds)):
        w1 = seeds[i][0]

        if i == len(seeds)-1:
            conf[w1] = 60
            continue

        w2 = seeds[i+1][0]

        s = compare_score(w1, w2, history, sos)

        conf[w1] = max(5, min(95, round(50 + s*5)))

    return conf

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

    df = pd.DataFrame(matches)
    results = {}

    for weight, group in df.groupby("weight"):
        gm = group.to_dict(orient="records")

        history = build_history(gm)
        sos = build_sos(history)
        seeds = compute_rankings(history, sos)
        common = build_common(history)
        confidence = build_confidence(seeds, history, sos)

        results[str(weight)] = {
            "seeds": [(i+1, w[0]) for i,w in enumerate(seeds)],
            "history": history,
            "sos": sos,
            "common": common,
            "confidence": confidence
        }

    return JSONResponse(results)
