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


def compute_rankings(matches):
    scores = {}
    for m in matches:
        w1, w2, winner = m["wrestlerA"], m["wrestlerB"], m["winner"]
        scores.setdefault(w1, 0)
        scores.setdefault(w2, 0)
        if winner == w1:
            scores[w1] += 1
        else:
            scores[w2] += 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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


# 🔥 NEW — Strength of Schedule
def build_sos(history):
    sos = {}

    for w in history:
        opponents = history[w]["wins"] + history[w]["losses"]

        if not opponents:
            sos[w] = 0
            continue

        opp_pcts = []
        for o in opponents:
            wins = len(history[o]["wins"])
            losses = len(history[o]["losses"])
            total = wins + losses
            if total > 0:
                opp_pcts.append(wins / total)

        sos[w] = round(sum(opp_pcts) / len(opp_pcts), 3) if opp_pcts else 0

    return sos


def build_confidence(seeds, history):
    confidence = {}

    for i in range(len(seeds)):
        w1 = seeds[i][0]

        if i == len(seeds)-1:
            confidence[w1] = 60
            continue

        w2 = seeds[i+1][0]

        score = 0

        if w2 in history[w1]["wins"]:
            score += 2
        elif w1 in history[w2]["wins"]:
            score -= 2

        diff = len(history[w1]["wins"]) - len(history[w2]["wins"])
        score += diff * 0.05

        conf = 50 + score * 10
        confidence[w1] = max(5, min(95, round(conf)))

    return confidence


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


def build_bracket(seeds):
    wrestlers = [w[0] for w in seeds]
    bracket = []
    while len(wrestlers) >= 2:
        bracket.append([wrestlers.pop(0), wrestlers.pop(-1)])
    return bracket


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
        group_matches = group.to_dict(orient="records")

        seeds = compute_rankings(group_matches)
        history = build_history(group_matches)
        sos = build_sos(history)

        results[str(weight)] = {
            "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
            "history": history,
            "confidence": build_confidence(seeds, history),
            "common": build_common(history),
            "bracket": build_bracket(seeds),
            "sos": sos
        }

    return JSONResponse(results)
