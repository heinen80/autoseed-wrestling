from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# ---------------------------
# CORS (allow frontend)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok"}


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
# History
# ---------------------------
def build_history(matches):
    history = {}

    for m in matches:
        w1 = str(m["wrestlerA"]).strip()
        w2 = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()

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
            reasons[w1].append("Higher win total")

    return reasons


# ---------------------------
# Confidence Score
# ---------------------------
def build_confidence(seeds, history):
    confidence = {}

    def score(w1, w2):
        s = 0

        # Head-to-head
        if w2 in history[w1]["wins"]:
            s += 3
        if w1 in history[w2]["wins"]:
            s -= 3

        # Win %
        w1_wins = len(history[w1]["wins"])
        w1_losses = len(history[w1]["losses"])
        w2_wins = len(history[w2]["wins"])
        w2_losses = len(history[w2]["losses"])

        w1_pct = w1_wins / (w1_wins + w1_losses) if (w1_wins + w1_losses) else 0
        w2_pct = w2_wins / (w2_wins + w2_losses) if (w2_wins + w2_losses) else 0

        s += (w1_pct - w2_pct) * 2

        # Common opponents
        common = set(history[w1]["wins"] + history[w1]["losses"]) & \
                 set(history[w2]["wins"] + history[w2]["losses"])

        if common:
            w1_common = sum(1 for c in common if c in history[w1]["wins"])
            w2_common = sum(1 for c in common if c in history[w2]["wins"])

            total = len(common)
            s += ((w1_common - w2_common) / total) * 2

        return s

    for i in range(len(seeds)):
        w1 = seeds[i][0]

        # Last seed = no one below → lower confidence
        if i == len(seeds) - 1:
            confidence[w1] = 60.0
            continue

        w2 = seeds[i+1][0]

        s = score(w1, w2)

        # Convert to confidence %
        # Big gap = high confidence
        # Small gap = low confidence
        conf = 50 + (s * 15)

        confidence[w1] = round(max(5, min(conf, 100)), 1)

    return confidence


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
# MAIN ENDPOINT
# ---------------------------
@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):

    try:
        # ---------------------------
        # Handle CSV Upload
        # ---------------------------
        if file:
            contents = await file.read()
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
            matches = df.to_dict(orient="records")

        # ---------------------------
        # Handle JSON (paste input)
        # ---------------------------
        else:
            data = await request.json()
            matches = data.get("matches", [])

        if not matches:
            return JSONResponse({"error": "No matches found"})

        # ---------------------------
        # Ensure weight exists
        # ---------------------------
        for m in matches:
            m["weight"] = m.get("weight", "unknown")

        df = pd.DataFrame(matches)

        results = {}

        # ---------------------------
        # Group by weight
        # ---------------------------
        for weight, group in df.groupby("weight"):

            group_matches = group.to_dict(orient="records")

            seeds = compute_rankings(group_matches)
            history = build_history(group_matches)
            reasons = build_reasons(seeds, history)
            confidence = build_confidence(seeds, history)
            bracket = build_bracket(seeds)

            results[str(weight)] = {
                "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
                "history": history,
                "reasons": reasons,
                "confidence": confidence,
                "bracket": bracket
            }

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({"error": str(e)})
