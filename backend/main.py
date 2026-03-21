from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# ---------------------------
# CORS
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
                reasons[w1].append(f"Beat {w2}")

        if not reasons[w1]:
            reasons[w1].append("Higher win total")

    return reasons


# ---------------------------
# Confidence (seed vs next)
# ---------------------------
def build_confidence(seeds, history):
    confidence = {}

    def compare(w1, w2):
        score = 0

        # Head-to-head
        if w2 in history[w1]["wins"]:
            score += 3
        elif w1 in history[w2]["wins"]:
            score -= 3

        # Record comparison
        w1_w = len(history[w1]["wins"])
        w1_l = len(history[w1]["losses"])
        w2_w = len(history[w2]["wins"])
        w2_l = len(history[w2]["losses"])

        w1_pct = w1_w / (w1_w + w1_l) if (w1_w + w1_l) else 0
        w2_pct = w2_w / (w2_w + w2_l) if (w2_w + w2_l) else 0

        if abs(w1_pct - w2_pct) > 0.15:
            score += 1 if w1_pct > w2_pct else -1

        # Common opponents
        common = set(history[w1]["wins"] + history[w1]["losses"]) & \
                 set(history[w2]["wins"] + history[w2]["losses"])

        if len(common) >= 3:
            w1_common = sum(1 for c in common if c in history[w1]["wins"])
            w2_common = sum(1 for c in common if c in history[w2]["wins"])

            if abs(w1_common - w2_common) >= 2:
                score += 1 if w1_common > w2_common else -1

        return score

    for i in range(len(seeds)):
        w1 = seeds[i][0]

        if i == len(seeds) - 1:
            confidence[w1] = 60.0
            continue

        w2 = seeds[i+1][0]

        s = compare(w1, w2)

        if s >= 4:
            conf = 95
        elif s == 3:
            conf = 88
        elif s == 2:
            conf = 78
        elif s == 1:
            conf = 65
        elif s == 0:
            conf = 55
        elif s == -1:
            conf = 45
        else:
            conf = 35

        confidence[w1] = conf

    return confidence


# ---------------------------
# Bracket
# ---------------------------
def build_bracket(seeds):
    wrestlers = [w[0] for w in seeds]
    bracket = []

    while len(wrestlers) >= 2:
        bracket.append([wrestlers.pop(0), wrestlers.pop(-1)])

    return bracket


# ---------------------------
# NEW: Common Opponent Data
# ---------------------------
def build_common_opponents(history):
    common_data = {}

    wrestlers = list(history.keys())

    for w1 in wrestlers:
        common_data[w1] = {}

        for w2 in wrestlers:
            if w1 == w2:
                continue

            w1_opps = set(history[w1]["wins"] + history[w1]["losses"])
            w2_opps = set(history[w2]["wins"] + history[w2]["losses"])

            common = w1_opps & w2_opps

            results = []

            for opp in common:
                w1_result = "W" if opp in history[w1]["wins"] else "L"
                w2_result = "W" if opp in history[w2]["wins"] else "L"

                results.append({
                    "opponent": opp,
                    "w1": w1_result,
                    "w2": w2_result
                })

            common_data[w1][w2] = results

    return common_data


# ---------------------------
# MAIN
# ---------------------------
@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):

    try:
        # CSV
        if file:
            contents = await file.read()
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
            matches = df.to_dict(orient="records")

        # JSON
        else:
            data = await request.json()
            matches = data.get("matches", [])

        if not matches:
            return JSONResponse({"error": "No matches found"})

        # Ensure weight
        for m in matches:
            m["weight"] = m.get("weight", "unknown")

        df = pd.DataFrame(matches)

        results = {}

        for weight, group in df.groupby("weight"):

            group_matches = group.to_dict(orient="records")

            seeds = compute_rankings(group_matches)
            history = build_history(group_matches)
            reasons = build_reasons(seeds, history)
            confidence = build_confidence(seeds, history)
            bracket = build_bracket(seeds)
            common = build_common_opponents(history)

            results[str(weight)] = {
                "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
                "history": history,
                "reasons": reasons,
                "confidence": confidence,
                "bracket": bracket,
                "common": common
            }

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({"error": str(e)})
