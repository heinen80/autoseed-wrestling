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
# GLOBAL SESSION STORE
# ---------------------------
sessions = {}

# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# ---------------------------
# HISTORY
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
        vals = []

        for o in opps:
            total = len(history[o]["wins"]) + len(history[o]["losses"])
            if total:
                vals.append(len(history[o]["wins"]) / total)

        sos[w] = round(sum(vals)/len(vals),3) if vals else 0

    return sos

# ---------------------------
# COMMON OPPONENTS
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
# COMPARE SCORE (CORE LOGIC)
# ---------------------------
def compare_score(a, b, history, sos):
    score = 0

    # Head-to-head
    if b in history[a]["wins"]:
        score += 5
    elif a in history[b]["wins"]:
        score -= 5

    # Record
    r1 = len(history[a]["wins"]) - len(history[a]["losses"])
    r2 = len(history[b]["wins"]) - len(history[b]["losses"])
    score += (r1 - r2) * 0.2

    # Common opponents
    common = set(history[a]["wins"] + history[a]["losses"]) & \
             set(history[b]["wins"] + history[b]["losses"])

    for c in common:
        if c in history[a]["wins"] and c in history[b]["losses"]:
            score += 1
        elif c in history[b]["wins"] and c in history[a]["losses"]:
            score -= 1

    # SOS
    score += (sos[a] - sos[b]) * 5

    return score

# ---------------------------
# RANKING (USES COMPARE)
# ---------------------------
def compute_rankings(history, sos):
    wrestlers = list(history.keys())

    scores = {w: 0 for w in wrestlers}

    for w1 in wrestlers:
        for w2 in wrestlers:
            if w1 == w2:
                continue

            s = compare_score(w1, w2, history, sos)

            if s > 0:
                scores[w1] += 1

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------
# CONFIDENCE
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
# UPLOAD
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

# ---------------------------
# SESSION / VOTING (FIXED)
# ---------------------------
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
