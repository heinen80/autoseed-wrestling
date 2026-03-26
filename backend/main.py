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
# in-memory live meeting store
# ---------------------------
sessions = {}

# ---------------------------
# health
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# ---------------------------
# helpers
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


def build_sos(history):
    sos = {}
    for w in history:
        opps = history[w]["wins"] + history[w]["losses"]
        vals = []

        for o in opps:
            total = len(history[o]["wins"]) + len(history[o]["losses"])
            if total:
                vals.append(len(history[o]["wins"]) / total)

        sos[w] = round(sum(vals) / len(vals), 3) if vals else 0
    return sos


def build_quality(history):
    top_wins = {}
    bad_losses = {}

    for w in history:
        top_wins[w] = []
        bad_losses[w] = []

        for opp in history[w]["wins"]:
            total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if total == 0:
                continue
            pct = len(history[opp]["wins"]) / total
            if pct >= 0.7:
                top_wins[w].append(opp)

        for opp in history[w]["losses"]:
            total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if total == 0:
                continue
            pct = len(history[opp]["wins"]) / total
            if pct <= 0.3:
                bad_losses[w].append(opp)

    return top_wins, bad_losses


def build_common(history):
    common = {}
    for w1 in history:
        common[w1] = {}
        for w2 in history:
            if w1 == w2:
                continue

            shared = set(history[w1]["wins"] + history[w1]["losses"]) & \
                     set(history[w2]["wins"] + history[w2]["losses"])

            rows = []
            for opp in sorted(shared):
                rows.append({
                    "opponent": opp,
                    "w1": "W" if opp in history[w1]["wins"] else "L",
                    "w2": "W" if opp in history[w2]["wins"] else "L",
                })

            common[w1][w2] = rows
    return common


def build_power_scores(history, sos, top_wins, bad_losses):
    scores = {}

    for w in history:
        wins = len(history[w]["wins"])
        losses = len(history[w]["losses"])
        total = wins + losses
        win_pct = wins / total if total else 0

        score = 0.0

        # baseline record
        score += win_pct * 50

        # reward quality wins
        for opp in history[w]["wins"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                score += opp_pct * 10

        # punish bad losses more than ordinary losses
        for opp in history[w]["losses"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                score -= (1 - opp_pct) * 8

        # sos
        score += sos[w] * 20

        # quality summary
        score += len(top_wins[w]) * 1.5
        score -= len(bad_losses[w]) * 1.5

        scores[w] = round(score, 3)

    return scores


def compute_rankings(power_scores):
    return sorted(power_scores.items(), key=lambda x: x[1], reverse=True)


def compare_breakdown(a, b, history, sos, top_wins, bad_losses):
    adv = 0
    reasons = []
    breakdown = {
        "head_to_head": 0,
        "record": 0,
        "common": 0,
        "sos": 0,
        "top_wins": 0,
        "bad_losses": 0
    }

    # head-to-head
    if b in history[a]["wins"]:
        adv += 5
        breakdown["head_to_head"] = 5
        reasons.append(f"{a} beat {b}")
    elif a in history[b]["wins"]:
        adv -= 5
        breakdown["head_to_head"] = -5
        reasons.append(f"{b} beat {a}")

    # record
    r1 = len(history[a]["wins"]) - len(history[a]["losses"])
    r2 = len(history[b]["wins"]) - len(history[b]["losses"])
    if r1 > r2:
        adv += 2
        breakdown["record"] = 2
        reasons.append(f"{a} better record")
    elif r2 > r1:
        adv -= 2
        breakdown["record"] = -2
        reasons.append(f"{b} better record")

    # common opponents
    common_rows = []
    shared = set(history[a]["wins"] + history[a]["losses"]) & \
             set(history[b]["wins"] + history[b]["losses"])

    for opp in shared:
        a_w = opp in history[a]["wins"]
        b_w = opp in history[b]["wins"]

        common_rows.append({
            "opponent": opp,
            "w1": "W" if a_w else "L",
            "w2": "W" if b_w else "L"
        })

        if a_w and not b_w:
            adv += 2
            breakdown["common"] += 2
            reasons.append(f"{a} better vs {opp}")
        elif b_w and not a_w:
            adv -= 2
            breakdown["common"] -= 2
            reasons.append(f"{b} better vs {opp}")

    # SOS
    diff = sos[a] - sos[b]
    if diff > 0:
        adv += 1
        breakdown["sos"] = 1
        reasons.append(f"{a} stronger schedule")
    elif diff < 0:
        adv -= 1
        breakdown["sos"] = -1
        reasons.append(f"{b} stronger schedule")

    # top wins
    tw_diff = len(top_wins[a]) - len(top_wins[b])
    if tw_diff > 0:
        adv += 1
        breakdown["top_wins"] = 1
        reasons.append(f"{a} more top wins")
    elif tw_diff < 0:
        adv -= 1
        breakdown["top_wins"] = -1
        reasons.append(f"{b} more top wins")

    # bad losses
    bl_diff = len(bad_losses[a]) - len(bad_losses[b])
    if bl_diff < 0:
        adv += 1
        breakdown["bad_losses"] = 1
        reasons.append(f"{a} fewer bad losses")
    elif bl_diff > 0:
        adv -= 1
        breakdown["bad_losses"] = -1
        reasons.append(f"{b} fewer bad losses")

    winner = a if adv > 0 else b if adv < 0 else "Too close"
    return {
        "winner": winner,
        "advantage": adv,
        "reasons": reasons,
        "breakdown": breakdown,
        "common_rows": sorted(common_rows, key=lambda x: x["opponent"])
    }


def build_confidence(seeds, history, sos, top_wins, bad_losses):
    conf = {}

    for i in range(len(seeds)):
        w1 = seeds[i][0]
        if i == len(seeds) - 1:
            conf[w1] = 60
            continue

        w2 = seeds[i + 1][0]
        comp = compare_breakdown(w1, w2, history, sos, top_wins, bad_losses)
        conf[w1] = max(10, min(95, int(50 + comp["advantage"] * 5)))

    return conf


def build_alerts(seeds, history, sos, top_wins, bad_losses, confidence):
    controversy_flags = []
    upset_alerts = []
    debate_queue = []

    names = [s[0] for s in seeds]

    for i, name in enumerate(names):
        # upset profile
        if i >= 6 and (len(top_wins[name]) >= 2 or sos[name] >= 0.55):
            upset_alerts.append(
                f"{name} is a dangerous lower seed: {len(top_wins[name])} top wins, SOS {sos[name]}"
            )

        # seed controversies with next wrestler
        if i < len(names) - 1:
            nxt = names[i + 1]
            comp = compare_breakdown(name, nxt, history, sos, top_wins, bad_losses)
            diff = abs((confidence.get(name, 50)) - (confidence.get(nxt, 50)))

            if comp["winner"] == nxt:
                controversy_flags.append(
                    f"Seed {i+1} may be wrong: {nxt} has stronger case than {name}"
                )

            if diff <= 8 or comp["winner"] == "Too close":
                debate_queue.append({
                    "higher": name,
                    "lower": nxt,
                    "reason": "Close confidence or conflicting data"
                })

    return controversy_flags, upset_alerts, debate_queue


# ---------------------------
# upload
# ---------------------------
@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):
    try:
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
            power_scores = build_power_scores(history, sos, top_wins, bad_losses)
            seeds = compute_rankings(power_scores)
            common = build_common(history)
            confidence = build_confidence(seeds, history, sos, top_wins, bad_losses)
            controversy_flags, upset_alerts, debate_queue = build_alerts(
                seeds, history, sos, top_wins, bad_losses, confidence
            )

            results[str(weight)] = {
                "seeds": [(i + 1, w[0]) for i, w in enumerate(seeds)],
                "history": history,
                "sos": sos,
                "common": common,
                "confidence": confidence,
                "top_wins": top_wins,
                "bad_losses": bad_losses,
                "power_scores": power_scores,
                "controversy_flags": controversy_flags,
                "upset_alerts": upset_alerts,
                "debate_queue": debate_queue
            }

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------
# live meeting voting
# ---------------------------
@app.post("/session/{weight}")
def create_session(weight: str):
    sessions[weight] = {
        "votes": {},
        "voters": {},
        "current_matchup": None
    }
    return {"status": "created"}

@app.post("/session/{weight}/matchup")
async def set_matchup(weight: str, request: Request):
    data = await request.json()
    a = data.get("a")
    b = data.get("b")

    sessions[weight] = {
        "votes": {a: 0, b: 0},
        "voters": {},
        "current_matchup": [a, b]
    }
    return sessions[weight]

@app.post("/vote/{weight}")
async def vote(weight: str, request: Request):
    data = await request.json()
    name = data.get("name")
    voter = data.get("voter", "Anonymous")
    role = data.get("role", "coach")

    sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})

    # one person, one active vote per matchup
    if voter in sessions[weight]["voters"]:
        prev = sessions[weight]["voters"][voter]["name"]
        prev_weight = sessions[weight]["voters"][voter]["weight"]
        if prev in sessions[weight]["votes"]:
            sessions[weight]["votes"][prev] -= prev_weight

    vote_weight = 2 if role == "head_coach" else 1

    sessions[weight]["votes"].setdefault(name, 0)
    sessions[weight]["votes"][name] += vote_weight
    sessions[weight]["voters"][voter] = {"name": name, "weight": vote_weight, "role": role}

    return sessions[weight]

@app.get("/votes/{weight}")
def get_votes(weight: str):
    return sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})
