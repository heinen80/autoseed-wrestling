from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Allow frontend access
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

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


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
# Scheduler
# ---------------------------
def schedule_matches(bracket):
    return [{"match": i+1, "teams": b} for i, b in enumerate(bracket)]


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

        # 1. Undefeated
        if w1_losses == 0:
            reasons[w1].append(f"Undefeated ({w1_wins}-0)")

        head_to_head_wins = []
        better_records = 0
        common_dominance = 0

        for j in range(i+1, len(seeds)):
            w2 = seeds[j][0]

            w2_wins, w2_losses = record(w2)

            # Head-to-head tracking
            if w2 in history[w1]["wins"]:
                head_to_head_wins.append(w2)

            # Record comparison
            if w1_wins > w2_wins:
                better_records += 1

            # Common opponents
            common = set(history[w1]["wins"] + history[w1]["losses"]) & \
                     set(history[w2]["wins"] + history[w2]["losses"])

            if len(common) > 0:
                w1_common_wins = sum(1 for c in common if c in history[w1]["wins"])
                w2_common_wins = sum(1 for c in common if c in history[w2]["wins"])

                if w1_common_wins > w2_common_wins:
                    common_dominance += 1

        # 2. Head-to-head summary
        if head_to_head_wins:
            wins_str = ", ".join(head_to_head_wins[:3])
            if len(head_to_head_wins) > 3:
                wins_str += f" +{len(head_to_head_wins)-3} more"
            reasons[w1].append(f"Head-to-head wins over: {wins_str}")

        # 3. Record dominance
        if better_records > 0:
            reasons[w1].append(f"Better record than {better_records} opponents")

        # 4. Common opponent dominance
        if common_dominance > 0:
            reasons[w1].append(f"Stronger vs common opponents")

        # fallback
        if not reasons[w1]:
            reasons[w1].append("Seeded by overall performance score")

    return reasons


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    matches = df.to_dict(orient="records")

    # Clean matches + build history
    history = {}

    cleaned_matches = []

    for m in matches:
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

    # Generate outputs
    seeds = compute_rankings(cleaned_matches)
    bracket = build_bracket(seeds)
    schedule = schedule_matches(bracket)
    reasons = build_reasons(seeds, history)

    return JSONResponse({
        "seeds": [(i+1, w[0], w[1]) for i, w in enumerate(seeds)],
        "bracket": bracket,
        "schedule": schedule,
        "history": history,
        "reasons": reasons
    })
