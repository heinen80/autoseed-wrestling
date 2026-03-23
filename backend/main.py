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

sessions = {}

@app.get("/")
def root():
    return {"status": "ok"}

# ---------------- HISTORY
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

# ---------------- SOS
def build_sos(history):
    sos = {}
    for w in history:
        opps = history[w]["wins"] + history[w]["losses"]
        vals = []

        for o in opps:
            total = len(history[o]["wins"]) + len(history[o]["losses"])
            if total:
                vals.append(len(history[o]["wins"]) / total)

        sos[w] = round(sum(vals)/len(vals),3) if vals else 0

    return sos

# ---------------- QUALITY
def build_quality(history):
    top_wins = {}
    bad_losses = {}

    for w in history:
        top_wins[w] = []
        bad_losses[w] = []

        for opp in history[w]["wins"]:
            total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if total == 0: continue
            pct = len(history[opp]["wins"]) / total

            if pct >= 0.7:
                top_wins[w].append(opp)

        for opp in history[w]["losses"]:
            total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if total == 0: continue
            pct = len(history[opp]["wins"]) / total

            if pct <= 0.3:
                bad_losses[w].append(opp)

    return top_wins, bad_losses

# ---------------- COMMON
def build_common(history):
    common = {}
    for w1 in history:
        common[w1] = {}
        for w2 in history:
            if w1 == w2: continue

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

# ---------------- RANKING
def compute_rankings(history, sos):
    scores = {}

    for w in history:
        wins = len(history[w]["wins"])
        losses = len(history[w]["losses"])
        total = wins + losses

        win_pct = wins / total if total else 0
        score = win_pct * 50

        for opp in history[w]["wins"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                score += (len(history[opp]["wins"]) / opp_total) * 10

        for opp in history[w]["losses"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                score -= (1 - (len(history[opp]["wins"]) / opp_total)) * 8

        score += sos[w] * 20
        scores[w] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------- CONFIDENCE
def build_confidence(seeds, history, sos):
    conf = {}
    for i in range(len(seeds)):
        w1 = seeds[i][0]

        if i == len(seeds)-1:
            conf[w1] = 60
            continue

        w2 = seeds[i+1][0]

        diff = len(history[w1]["wins"]) - len(history[w2]["wins"])
        diff += (sos[w1] - sos[w2]) * 10

        conf[w1] = max(10, min(95, int(50 + diff*5)))

    return conf

# ---------------- UPLOAD
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
        top_wins, bad_losses = build_quality(history)
        seeds = compute_rankings(history, sos)
        common = build_common(history)
        confidence = build_confidence(seeds, history, sos)

        results[str(weight)] = {
            "seeds": [(i+1, w[0]) for i,w in enumerate(seeds)],
            "history": history,
            "sos": sos,
            "common": common,
            "confidence": confidence,
            "top_wins": top_wins,
            "bad_losses": bad_losses
        }

    return JSONResponse(results)

# ---------------- VOTING
@app.post("/session/{weight}")
def create_session(weight: str):
    sessions[weight] = {"votes": {}}
    return {"status": "created"}

@app.post("/vote/{weight}")
async def vote(weight: str, request: Request):
    data = await request.json()
    name = data.get("name")

    sessions.setdefault(weight, {"votes": {}})
    sessions[weight]["votes"].setdefault(name, 0)
    sessions[weight]["votes"][name] += 1

    return sessions[weight]

@app.get("/votes/{weight}")
def get_votes(weight: str):
    return sessions.setdefault(weight, {"votes": {}})
