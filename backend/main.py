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

WIN_METHOD_POINTS = {
    "fall": 6.0,
    "pin": 6.0,
    "tf": 5.0,
    "tech fall": 5.0,
    "tech_fall": 5.0,
    "md": 4.0,
    "major decision": 4.0,
    "major_decision": 4.0,
    "dec": 3.0,
    "decision": 3.0,
    "default": 3.0,
}

def get_method_points(method):
    if not method:
        return 3.0
    return WIN_METHOD_POINTS.get(str(method).strip().lower(), 3.0)

@app.get("/")
def root():
    return {"status": "ok"}

def build_history(matches):
    history = {}
    for m in matches:
        w1 = str(m["wrestlerA"]).strip()
        w2 = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()
        method = str(m.get("method", "")).strip()
        history.setdefault(w1, {"wins": [], "losses": [], "win_methods": {}, "loss_methods": {}})
        history.setdefault(w2, {"wins": [], "losses": [], "win_methods": {}, "loss_methods": {}})
        pts = get_method_points(method)
        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w1]["win_methods"][w2] = pts
            history[w2]["losses"].append(w1)
            history[w2]["loss_methods"][w1] = pts
        else:
            history[w2]["wins"].append(w1)
            history[w2]["win_methods"][w1] = pts
            history[w1]["losses"].append(w2)
            history[w1]["loss_methods"][w2] = pts
    return history

def avg_win_quality(wrestler, history):
    methods = history[wrestler]["win_methods"]
    if not methods:
        return 0.0
    return sum(methods.values()) / len(methods)

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
                w1_won = opp in history[w1]["wins"]
                w2_won = opp in history[w2]["wins"]
                rows.append({
                    "opponent": opp,
                    "w1": "W" if w1_won else "L",
                    "w2": "W" if w2_won else "L",
                    "w1_pts": history[w1]["win_methods"].get(opp, 0) if w1_won else history[w1]["loss_methods"].get(opp, 0),
                    "w2_pts": history[w2]["win_methods"].get(opp, 0) if w2_won else history[w2]["loss_methods"].get(opp, 0),
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
        score += win_pct * 30
        score += avg_win_quality(w, history) * 4
        for opp in history[w]["wins"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                method_pts = history[w]["win_methods"].get(opp, 3.0)
                score += opp_pct * 8 * (method_pts / 6.0)
        for opp in history[w]["losses"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                score -= (1 - opp_pct) * 8
        score += sos[w] * 15
        score += len(top_wins[w]) * 2
        score -= len(bad_losses[w]) * 2
        scores[w] = round(score, 3)
    return scores

def compute_rankings(power_scores):
    return sorted(power_scores.items(), key=lambda x: x[1], reverse=True)

def _method_label(pts):
    if pts >= 6: return "Fall"
    if pts >= 5: return "Tech Fall"
    if pts >= 4: return "Major Dec"
    if pts >= 3: return "Decision"
    return "-"

def compare_breakdown(a, b, history, sos, top_wins, bad_losses, power_scores=None):
    adv = 0
    reasons = []
    breakdown = {
        "head_to_head": 0,
        "win_quality": 0,
        "record": 0,
        "common": 0,
        "sos": 0,
        "top_wins": 0,
        "bad_losses": 0,
    }
    if b in history[a]["wins"]:
        method_pts = history[a]["win_methods"].get(b, 3.0)
        pts = 3 + round(method_pts * 0.75)
        adv += pts
        breakdown["head_to_head"] = pts
        reasons.append(f"{a} beat {b} ({_method_label(method_pts)})")
    elif a in history[b]["wins"]:
        method_pts = history[b]["win_methods"].get(a, 3.0)
        pts = 3 + round(method_pts * 0.75)
        adv -= pts
        breakdown["head_to_head"] = -pts
        reasons.append(f"{b} beat {a} ({_method_label(method_pts)})")
    aq_a = avg_win_quality(a, history)
    aq_b = avg_win_quality(b, history)
    aq_diff = round(aq_a - aq_b, 2)
    if aq_diff >= 0.5:
        adv += 2
        breakdown["win_quality"] = 2
        reasons.append(f"{a} wins by better methods (avg {aq_a:.1f} vs {aq_b:.1f})")
    elif aq_diff <= -0.5:
        adv -= 2
        breakdown["win_quality"] = -2
        reasons.append(f"{b} wins by better methods (avg {aq_b:.1f} vs {aq_a:.1f})")
    r1 = len(history[a]["wins"]) - len(history[a]["losses"])
    r2 = len(history[b]["wins"]) - len(history[b]["losses"])
    if r1 > r2:
        adv += 2
        breakdown["record"] = 2
        reasons.append(f"{a} has better record")
    elif r2 > r1:
        adv -= 2
        breakdown["record"] = -2
        reasons.append(f"{b} has better record")
    common_rows = []
    shared = set(history[a]["wins"] + history[a]["losses"]) & \
             set(history[b]["wins"] + history[b]["losses"])
    for opp in shared:
        a_won = opp in history[a]["wins"]
        b_won = opp in history[b]["wins"]
        a_pts = history[a]["win_methods"].get(opp, 0) if a_won else history[a]["loss_methods"].get(opp, 0)
        b_pts = history[b]["win_methods"].get(opp, 0) if b_won else history[b]["loss_methods"].get(opp, 0)
        common_rows.append({
            "opponent": opp,
            "w1": "W" if a_won else "L",
            "w2": "W" if b_won else "L",
            "w1_pts": a_pts,
            "w2_pts": b_pts,
        })
        if a_won and not b_won:
            method_bonus = round(a_pts / 6.0, 1)
            pts = round(1.5 + method_bonus)
            adv += pts
            breakdown["common"] += pts
            reasons.append(f"{a} beat {opp} ({_method_label(a_pts)}), {b} lost")
        elif b_won and not a_won:
            method_bonus = round(b_pts / 6.0, 1)
            pts = round(1.5 + method_bonus)
            adv -= pts
            breakdown["common"] -= pts
            reasons.append(f"{b} beat {opp} ({_method_label(b_pts)}), {a} lost")
        elif a_won and b_won:
            if a_pts > b_pts + 0.5:
                adv += 1
                breakdown["common"] += 1
                reasons.append(f"{a} beat {opp} more impressively ({_method_label(a_pts)} vs {_method_label(b_pts)})")
            elif b_pts > a_pts + 0.5:
                adv -= 1
                breakdown["common"] -= 1
                reasons.append(f"{b} beat {opp} more impressively ({_method_label(b_pts)} vs {_method_label(a_pts)})")
    diff = sos[a] - sos[b]
    if diff > 0.03:
        adv += 1
        breakdown["sos"] = 1
        reasons.append(f"{a} stronger schedule (SOS {sos[a]:.3f} vs {sos[b]:.3f})")
    elif diff < -0.03:
        adv -= 1
        breakdown["sos"] = -1
        reasons.append(f"{b} stronger schedule (SOS {sos[b]:.3f} vs {sos[a]:.3f})")
    tw_diff = len(top_wins[a]) - len(top_wins[b])
    if tw_diff > 0:
        adv += 1
        breakdown["top_wins"] = 1
        reasons.append(f"{a} more top wins ({len(top_wins[a])} vs {len(top_wins[b])})")
    elif tw_diff < 0:
        adv -= 1
        breakdown["top_wins"] = -1
        reasons.append(f"{b} more top wins ({len(top_wins[b])} vs {len(top_wins[a])})")
    bl_diff = len(bad_losses[a]) - len(bad_losses[b])
    if bl_diff < 0:
        adv += 1
        breakdown["bad_losses"] = 1
        reasons.append(f"{a} fewer bad losses ({len(bad_losses[a])} vs {len(bad_losses[b])})")
    elif bl_diff > 0:
        adv -= 1
        breakdown["bad_losses"] = -1
        reasons.append(f"{b} fewer bad losses ({len(bad_losses[b])} vs {len(bad_losses[a])})")
    if adv == 0 and power_scores:
        ps_diff = power_scores.get(a, 0) - power_scores.get(b, 0)
        if ps_diff > 0:
            adv = 1
            reasons.append(f"{a} edges on power score ({power_scores[a]:.1f} vs {power_scores[b]:.1f})")
        elif ps_diff < 0:
            adv = -1
            reasons.append(f"{b} edges on power score ({power_scores[b]:.1f} vs {power_scores[a]:.1f})")
    winner = a if adv > 0 else b if adv < 0 else "Too close"
    return {
        "winner": winner,
        "advantage": adv,
        "reasons": reasons,
        "breakdown": breakdown,
        "common_rows": sorted(common_rows, key=lambda x: x["opponent"]),
    }

def build_win_method_summary(history):
    summary = {}
    for w in history:
        counts = {"fall": 0, "tech_fall": 0, "major_dec": 0, "decision": 0}
        for opp, pts in history[w]["win_methods"].items():
            if pts >= 6: counts["fall"] += 1
            elif pts >= 5: counts["tech_fall"] += 1
            elif pts >= 4: counts["major_dec"] += 1
            else: counts["decision"] += 1
        summary[w] = counts
    return summary

def build_confidence(seeds, history, sos, top_wins, bad_losses, power_scores):
    conf = {}
    seed_names = [s[0] for s in seeds]
    for i, (name, _) in enumerate(seeds):
        if i == len(seeds) - 1:
            conf[name] = 55
            continue
        nxt = seed_names[i + 1]
        comp = compare_breakdown(name, nxt, history, sos, top_wins, bad_losses, power_scores)
        raw = 50 + comp["advantage"] * 4
        conf[name] = max(10, min(97, int(raw)))
    return conf

def build_alerts(seeds, history, sos, top_wins, bad_losses, confidence, power_scores):
    controversy_flags = []
    upset_alerts = []
    debate_queue = []
    names = [s[0] for s in seeds]
    for i, name in enumerate(names):
        if i >= 5 and (len(top_wins[name]) >= 2 or sos[name] >= 0.55):
            upset_alerts.append(
                f"{name} (seed {i+1}): {len(top_wins[name])} top wins, SOS {sos[name]:.3f}"
            )
        if i < len(names) - 1:
            nxt = names[i + 1]
            comp = compare_breakdown(name, nxt, history, sos, top_wins, bad_losses, power_scores)
            if comp["winner"] == nxt:
                controversy_flags.append(
                    f"Seed {i+1} vs {i+2}: algorithm favors {nxt} over {name} - review recommended"
                )
            if abs(comp["advantage"]) <= 2 or comp["winner"] == "Too close":
                debate_queue.append({
                    "higher": name,
                    "lower": nxt,
                    "reason": comp["reasons"][0] if comp["reasons"] else "No clear separator",
                    "advantage": comp["advantage"],
                })
    return controversy_flags, upset_alerts, debate_queue

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
            win_methods = build_win_method_summary(history)
            confidence = build_confidence(seeds, history, sos, top_wins, bad_losses, power_scores)
            controversy_flags, upset_alerts, debate_queue = build_alerts(
                seeds, history, sos, top_wins, bad_losses, confidence, power_scores
            )
            results[str(weight)] = {
                "seeds": [(i + 1, w[0]) for i, w in enumerate(seeds)],
                "history": history,
                "sos": sos,
                "common": common,
                "confidence": confidence,
                "top_wins": top_wins,
                "bad_losses": bad_losses,
                "power_scores": {k: round(v, 1) for k, v in power_scores.items()},
                "win_methods": win_methods,
                "controversy_flags": controversy_flags,
                "upset_alerts": upset_alerts,
                "debate_queue": debate_queue,
            }
        return JSONResponse(results)
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/session/{weight}")
def create_session(weight: str):
    sessions[weight] = {
        "votes": {},
        "voters": {},
        "current_matchup": None,
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
        "current_matchup": [a, b],
    }
    return sessions[weight]

@app.post("/vote/{weight}")
async def vote(weight: str, request: Request):
    data = await request.json()
    name = data.get("name")
    voter = data.get("voter", "Anonymous")
    role = data.get("role", "coach")
    sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})
    if voter in sessions[weight]["voters"]:
        prev = sessions[weight]["voters"][voter]["name"]
        prev_weight = sessions[weight]["voters"][voter]["weight"]
        if prev in sessions[weight]["votes"]:
            sessions[weight]["votes"][prev] -= prev_weight
    vote_weight = 2 if role == "head_coach" else 1
    sessions[weight]["votes"].setdefault(name, 0)
    sessions[weight]["votes"][name] += vote_weight
    sessions[weight]["voters"][voter] = {
        "name": name,
        "weight": vote_weight,
        "role": role,
    }
    return sessions[weight]

@app.get("/votes/{weight}")
def get_votes(weight: str):
    return sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})
