from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# ✅ FIX CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Ranking Engine
# ---------------------------
def build_graph(matches):
    wrestlers = sorted(set([w for m in matches for w in [m['wrestlerA'], m['wrestlerB']]]))
    index = {w:i for i,w in enumerate(wrestlers)}
    n = len(wrestlers)

    matrix = np.zeros((n,n))

    for m in matches:
        winner = m['winner']
        loser = m['wrestlerB'] if winner == m['wrestlerA'] else m['wrestlerA']
        i, j = index[winner], index[loser]
        matrix[j][i] += 1

    for i in range(n):
        s = matrix[:,i].sum()
        if s > 0:
            matrix[:,i] /= s

    return matrix, wrestlers


def rank_wrestlers(matrix):
    n = len(matrix)
    rank = np.ones(n)/n

    for _ in range(100):
        rank = matrix.dot(rank)

    return rank


def compute_rankings(matches):
    matrix, wrestlers = build_graph(matches)
    scores = rank_wrestlers(matrix)

    ranking = list(zip(wrestlers, scores))
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking

# ---------------------------
# Bracket Builder
# ---------------------------
def build_bracket(seeds):
    n = len(seeds)
    return [(seeds[i][0], seeds[n-1-i][0]) for i in range(n//2)]

# ---------------------------
# Scheduler
# ---------------------------
def schedule_matches(bracket, mats=3):
    schedule = []
    mat_times = [0]*mats

    for match in bracket:
        mat = mat_times.index(min(mat_times))
        start = mat_times[mat]
        end = start + 10

        schedule.append({
            "match": match,
            "mat": mat+1,
            "start": start,
            "end": end
        })

        mat_times[mat] = end

    return schedule

# ---------------------------
# API
# ---------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    matches = df.to_dict(orient="records")

    seeds = compute_rankings(matches)
    bracket = build_bracket(seeds)
    schedule = schedule_matches(bracket)

    return JSONResponse({
        "seeds": [(i+1, w[0], round(w[1],3)) for i,w in enumerate(seeds)],
        "bracket": bracket,
        "schedule": schedule
    })
