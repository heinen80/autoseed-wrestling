from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Ranking Logic
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
# Bracket Builder
# ---------------------------
def build_bracket(seeds):
    wrestlers = [w[0] for w in seeds]
    bracket = []

    while len(wrestlers) >= 2:
        bracket.append([wrestlers.pop(0), wrestlers.pop(-1)])

    return bracket


# ---------------------------
# Reasoning Engine
# ---------------------------
def build_reasons(seeds, history):
    reasons = {}

    def record(w):
        return len(history[w]["wins"]), len(history[w]["losses"])

    for i in range(len(seeds)):
        w1 = seeds[i][0]
        reasons[w1] = []

        w1_wins, w1_losses = record(w1)

        if w1_losses == 0:
            reasons[w1].append(f"Undefeated ({w1_wins}-0)")

        head_wins = []
        better_records = 0

        for j in range(i+1, len(seeds)):
            w2 = seeds[j][0]
            w2_wins, _ = record(w2)

            if w2 in history[w1]["wins"]:
                head_wins.append(w2)

            if w1_wins > w2_wins:
                better_records += 1

        if head_wins:
            reasons[w1].append(f"Head-to-head wins over: {', '.join(head_wins[:3])}")

        if better_records:
            reasons[w1].append(f"Better record than {better_records} opponents")

        if not reasons[w1]:
            reasons[w1].append("Seeded by overall performance")

    return reasons


# ---------------------------
# MAIN ENDPOINT
# ---------------------------
@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):

    # ---------------------------
    # Handle CSV OR JSON
    # ---------------------------
    if file:
        df = pd.read_csv(file.file)
        df.columns = [c.strip() for c in df.columns]
        matches = df.to_dict(orient="records")
    else:
        data = await request.json()
        matches = data.get("matches", [])

    # ---------------------------
    # Ensure weight exists
    # ---------------------------
    for m in matches:
        if "weight" not in m or not m["weight"]:
            m["weight"] = "unknown"

    results = {}

    # ---------------------------
    # GROUP BY WEIGHT
    # ---------------------------
    df = pd.DataFrame(matches)

    for weight, group in df.groupby("weight"):

        group_matches = group.to_dict(orient="records")

        history = {}
        cleaned_matches = []

        for m in group_matches:
            w1 = str(m["wrestlerA"]).strip()
            w2 = str(m["wrestlerB"]).strip()
            winner = str(m["winner"]).strip()

            cleaned_matches.append({
                "wrestlerA": w1,
                "wrestlerB": w2,
                "winner": winner
            })

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

        seeds = compute_rankings(cleaned_matches)
        bracket = build_bracket(seeds)
        reasons = build_reasons(seeds, history)

        results[str(weight)] = {
            "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
            "bracket": bracket,
            "history": history,
            "reasons": reasons
        }

    return JSONResponse(results)
