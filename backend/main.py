from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SIMPLE RANKING LOGIC ---
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

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def build_bracket(seeds):
    wrestlers = [w[0] for w in seeds]
    bracket = []

    while len(wrestlers) >= 2:
        bracket.append([wrestlers.pop(0), wrestlers.pop(-1)])

    return bracket


def schedule_matches(bracket):
    return [{"match": i+1, "teams": b} for i, b in enumerate(bracket)]


# --- MAIN ENDPOINT ---
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    matches = df.to_dict(orient="records")

    seeds = compute_rankings(matches)
    bracket = build_bracket(seeds)
    schedule = schedule_matches(bracket)

    # Build wrestler history
    history = {}

    for m in matches:
        w1 = str(m["wrestlerA"]).strip()
        w2 = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()

        if w1 not in history:
            history[w1] = {"wins": [], "losses": []}
        if w2 not in history:
            history[w2] = {"wins": [], "losses": []}

        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w2]["losses"].append(w1)
        else:
            history[w2]["wins"].append(w1)
            history[w1]["losses"].append(w2)

    return JSONResponse({
        "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
        "bracket": bracket,
        "schedule": schedule,
        "history": history
    })
