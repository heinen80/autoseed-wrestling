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

# —————————

# in-memory live meeting store

# —————————

sessions = {}

# —————————

# Win method point values

# Used everywhere: power scores, comparison, confidence

# —————————

WIN_METHOD_POINTS = {
"fall":           6.0,
"pin":            6.0,
"tf":             5.0,
"tech fall":      5.0,
"tech_fall":      5.0,
"md":             4.0,
"major decision": 4.0,
"major_decision": 4.0,
"dec":            3.0,
"decision":       3.0,
"default":        3.0,   # forfeit / default win — treated like a decision
}

def get_method_points(method: str) -> float:
    """Return point value for a win method string. Falls back to decision (3.0)."""
    if not method:
        return 3.0
    return WIN_METHOD_POINTS.get(str(method).strip().lower(), 3.0)

# —————————

# health

# —————————

@app.get("/")
def root():
    return {"status": "ok"}

# —————————

# helpers

# —————————

def sanitize_for_json(obj):
    """
    Recursively replace float NaN/Inf values with 0.0 so JSONResponse never
    raises "Out of range float values are not JSON compliant".
    Also converts pandas NaN (which is float('nan')) coming from to_dict().
    """
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    return obj


def build_history(matches):
    """
    Build per-wrestler win/loss history.
    Tracks win_methods/loss_methods (method points) and win_dates/loss_dates
    (date strings, may be None) for recency weighting.

    CSV columns expected:
        weight, wrestlerA, wrestlerB, winner, method (optional), date (optional)

    method examples: fall, pin, tf, tech fall, md, major decision, dec, decision
    """
    history = {}

    for m in matches:
        w1     = str(m["wrestlerA"]).strip()
        w2     = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()
        method = str(m.get("method", "")).strip()
        date   = m.get("date")

        history.setdefault(w1, {
            "wins": [], "losses": [],
            "win_methods": {}, "loss_methods": {},
            "win_dates": {}, "loss_dates": {},
        })
        history.setdefault(w2, {
            "wins": [], "losses": [],
            "win_methods": {}, "loss_methods": {},
            "win_dates": {}, "loss_dates": {},
        })

        pts = get_method_points(method)

        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w1]["win_methods"][w2] = pts
            history[w1]["win_dates"][w2]   = date
            history[w2]["losses"].append(w1)
            history[w2]["loss_methods"][w1] = pts
            history[w2]["loss_dates"][w1]   = date
        else:
            history[w2]["wins"].append(w1)
            history[w2]["win_methods"][w1] = pts
            history[w2]["win_dates"][w1]   = date
            history[w1]["losses"].append(w2)
            history[w1]["loss_methods"][w2] = pts
            history[w1]["loss_dates"][w2]   = date

    return history

def avg_win_quality(wrestler: str, history: dict) -> float:
    """Average method points across all wins. 0 if no wins."""
    methods = history[wrestler]["win_methods"]
    if not methods:
        return 0.0
    return sum(methods.values()) / len(methods)

_SOS_MIN_OPPONENT_MATCHES = 4  # opponents with fewer matches are excluded as likely backups

def build_sos(history):
    """
    Strength of schedule: average win-rate of opponents faced.
    Excludes opponents with fewer than 4 total matches (backup exclusion).
    """
    sos = {}
    for w in history:
        opps = history[w]["wins"] + history[w]["losses"]
        vals = []
        for o in opps:
            if o not in history:
                continue
            total = len(history[o]["wins"]) + len(history[o]["losses"])
            if total < _SOS_MIN_OPPONENT_MATCHES:
                continue  # exclude backups / sparse opponents
            vals.append(len(history[o]["wins"]) / total)
        sos[w] = round(sum(vals) / len(vals), 3) if vals else 0
    return sos

def build_quality(history):
    top_wins  = {}
    bad_losses = {}

    for w in history:
        top_wins[w]   = []
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

            shared = (
                set(history[w1]["wins"] + history[w1]["losses"]) &
                set(history[w2]["wins"] + history[w2]["losses"])
            )

            rows = []
            for opp in sorted(shared):
                w1_won = opp in history[w1]["wins"]
                w2_won = opp in history[w2]["wins"]
                rows.append({
                    "opponent": opp,
                    "w1":       "W" if w1_won else "L",
                    "w2":       "W" if w2_won else "L",
                    # include method quality so frontend can show it
                    "w1_pts":   history[w1]["win_methods"].get(opp, 0) if w1_won
                                else history[w1]["loss_methods"].get(opp, 0),
                    "w2_pts":   history[w2]["win_methods"].get(opp, 0) if w2_won
                                else history[w2]["loss_methods"].get(opp, 0),
                })
            common[w1][w2] = rows
    return common

_RECENCY_DAYS       = 45   # matches within this many days count as recent
_RECENCY_MULTIPLIER = 1.2  # recent matches weighted 1.2x; older matches 1.0x (no penalty)

def _recency_mult(date_val):
    """Return 1.2 if date_val is within the last 45 days, else 1.0."""
    if not date_val:
        return 1.0
    try:
        if isinstance(date_val, str):
            for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    d = datetime.strptime(date_val, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 1.0
        else:
            d = date_val
        if (datetime.now() - d).days <= _RECENCY_DAYS:
            return _RECENCY_MULTIPLIER
    except Exception:
        pass
    return 1.0

def build_power_scores(history, sos, top_wins, bad_losses, field_match_avg=None):
    """
    Unified power score — identical weighting to compare_breakdown().

    Components (max ~100):
        win %            × 30   → 0–30
        win quality avg  × 4    → 0–24   (fall=6, tf=5, md=4, dec=3)
        opponent quality × 8    per win, recency-weighted (×1.2 if ≤45 days old)
        opponent quality × 8    penalty per loss, recency-weighted
        SOS              × 15   × field-relative confidence weight (0.7 / 0.85 / 1.0)
        top wins         × 2    each
        bad losses       × 2    penalty each

    field_match_avg: average match count across all field wrestlers in this weight.
        Used for field-relative SOS confidence:
        < 50% of avg → SOS weight 0.7x (low confidence)
        50–80% of avg → SOS weight 0.85x
        ≥ 80% of avg → SOS weight 1.0x (full)
    """
    scores = {}

    for w in history:
        wins   = len(history[w]["wins"])
        losses = len(history[w]["losses"])
        total  = wins + losses
        win_pct = wins / total if total else 0

        # Field-relative SOS confidence weight
        if field_match_avg and field_match_avg > 0:
            ratio = total / field_match_avg
            if ratio < 0.5:
                sos_weight = 0.7
            elif ratio < 0.8:
                sos_weight = 0.85
            else:
                sos_weight = 1.0
        else:
            sos_weight = 1.0

        win_dates  = history[w].get("win_dates",  {})
        loss_dates = history[w].get("loss_dates", {})

        score = 0.0

        # 1. Win percentage baseline
        score += win_pct * 30

        # 2. Average win quality (method-based)
        score += avg_win_quality(w, history) * 4

        # 3. Opponent quality on wins — recency-weighted
        for opp in history[w]["wins"]:
            if opp not in history:
                continue
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                method_pts = history[w]["win_methods"].get(opp, 3.0)
                recency = _recency_mult(win_dates.get(opp))
                score += opp_pct * 8 * (method_pts / 6.0) * recency

        # 4. Penalty for losses — recency-weighted
        for opp in history[w]["losses"]:
            if opp not in history:
                continue
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                recency = _recency_mult(loss_dates.get(opp))
                score -= (1 - opp_pct) * 8 * recency

        # 5. Strength of schedule — field-relative confidence weight
        score += sos[w] * 15 * sos_weight

        # 6. Top wins bonus / bad loss penalty
        score += len(top_wins[w]) * 2
        score -= len(bad_losses[w]) * 2

        scores[w] = round(score, 3)

    return scores

def compute_rankings(power_scores):
    return sorted(power_scores.items(), key=lambda x: x[1], reverse=True)

def compare_breakdown(a, b, history, sos, top_wins, bad_losses, power_scores=None):
    """
    Head-to-head comparison using the SAME criteria weights as build_power_scores.
    Returns advantage score, reasons, and per-category breakdown.
    Now includes win method quality in head-to-head and common opponent scoring.
    """
    adv = 0
    reasons = []
    breakdown = {
    "head_to_head": 0,
    "win_quality":  0,
    "record":       0,
    "common":       0,
    "sos":          0,
    "top_wins":     0,
    "bad_losses":   0,
    }

    # 1. Head-to-head — weighted by HOW they won
    if b in history[a]["wins"]:
        method_pts = history[a]["win_methods"].get(b, 3.0)
        # scale: fall(6)→+7, tf(5)→+6, md(4)→+5, dec(3)→+4
        pts = 3 + round(method_pts * 0.75)
        adv += pts
        breakdown["head_to_head"] = pts
        method_label = _method_label(method_pts)
        reasons.append(f"{a} beat {b} ({method_label})")
    elif a in history[b]["wins"]:
        method_pts = history[b]["win_methods"].get(a, 3.0)
        pts = 3 + round(method_pts * 0.75)
        adv -= pts
        breakdown["head_to_head"] = -pts
        method_label = _method_label(method_pts)
        reasons.append(f"{b} beat {a} ({method_label})")

    # 2. Win quality comparison (average method points)
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

    # 3. Season record (win-loss differential)
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

    # 4. Common opponents — method-weighted
    common_rows = []
    shared = (
        set(history[a]["wins"] + history[a]["losses"]) &
        set(history[b]["wins"] + history[b]["losses"])
    )

    for opp in shared:
        a_won = opp in history[a]["wins"]
        b_won = opp in history[b]["wins"]

        a_pts = history[a]["win_methods"].get(opp, 0) if a_won \
                else history[a]["loss_methods"].get(opp, 0)
        b_pts = history[b]["win_methods"].get(opp, 0) if b_won \
                else history[b]["loss_methods"].get(opp, 0)

        common_rows.append({
            "opponent": opp,
            "w1": "W" if a_won else "L",
            "w2": "W" if b_won else "L",
            "w1_pts": a_pts,
            "w2_pts": b_pts,
        })

        if a_won and not b_won:
            # bonus if a won impressively
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
            # both won — compare HOW they won
            if a_pts > b_pts + 0.5:
                adv += 1
                breakdown["common"] += 1
                reasons.append(f"{a} beat {opp} more impressively ({_method_label(a_pts)} vs {_method_label(b_pts)})")
            elif b_pts > a_pts + 0.5:
                adv -= 1
                breakdown["common"] -= 1
                reasons.append(f"{b} beat {opp} more impressively ({_method_label(b_pts)} vs {_method_label(a_pts)})")

    # 5. Strength of schedule
    diff = sos[a] - sos[b]
    if diff > 0.03:
        adv += 1
        breakdown["sos"] = 1
        reasons.append(f"{a} has stronger schedule (SOS {sos[a]:.3f} vs {sos[b]:.3f})")
    elif diff < -0.03:
        adv -= 1
        breakdown["sos"] = -1
        reasons.append(f"{b} has stronger schedule (SOS {sos[b]:.3f} vs {sos[a]:.3f})")

    # 6. Top wins count
    tw_diff = len(top_wins[a]) - len(top_wins[b])
    if tw_diff > 0:
        adv += 1
        breakdown["top_wins"] = 1
        reasons.append(f"{a} has more top wins ({len(top_wins[a])} vs {len(top_wins[b])})")
    elif tw_diff < 0:
        adv -= 1
        breakdown["top_wins"] = -1
        reasons.append(f"{b} has more top wins ({len(top_wins[b])} vs {len(top_wins[a])})")

    # 7. Bad losses
    bl_diff = len(bad_losses[a]) - len(bad_losses[b])
    if bl_diff < 0:
        adv += 1
        breakdown["bad_losses"] = 1
        reasons.append(f"{a} has fewer bad losses ({len(bad_losses[a])} vs {len(bad_losses[b])})")
    elif bl_diff > 0:
        adv -= 1
        breakdown["bad_losses"] = -1
        reasons.append(f"{b} has fewer bad losses ({len(bad_losses[b])} vs {len(bad_losses[a])})")

    # Tiebreaker: power score differential if still tied
    if adv == 0 and power_scores:
        ps_diff = power_scores.get(a, 0) - power_scores.get(b, 0)
        if ps_diff > 0:
            adv = 1
            reasons.append(f"{a} edges on overall power score ({power_scores[a]:.1f} vs {power_scores[b]:.1f})")
        elif ps_diff < 0:
            adv = -1
            reasons.append(f"{b} edges on overall power score ({power_scores[b]:.1f} vs {power_scores[a]:.1f})")

    winner = a if adv > 0 else b if adv < 0 else "Too close"

    return {
        "winner":      winner,
        "advantage":   adv,
        "reasons":     reasons,
        "breakdown":   breakdown,
        "common_rows": sorted(common_rows, key=lambda x: x["opponent"]),
    }

def _method_label(pts: float) -> str:
    """Convert method points back to a human-readable label."""
    if pts >= 6:   return "Fall"
    if pts >= 5:   return "Tech Fall"
    if pts >= 4:   return "Major Dec"
    if pts >= 3:   return "Decision"
    return "—"

def build_win_method_summary(history):
    """
    Returns per-wrestler breakdown of win types for display.
    { wrestler: { "fall": n, "tech_fall": n, "major_dec": n, "decision": n } }
    """
    summary = {}
    for w in history:
        counts = {"fall": 0, "tech_fall": 0, "major_dec": 0, "decision": 0}
        for opp, pts in history[w]["win_methods"].items():
            if pts >= 6:   counts["fall"]      += 1
            elif pts >= 5: counts["tech_fall"] += 1
            elif pts >= 4: counts["major_dec"] += 1
            else:          counts["decision"]  += 1
        summary[w] = counts
    return summary

def build_confidence(seeds, history, sos, top_wins, bad_losses, power_scores):
    """
    Confidence is based on the compare_breakdown advantage score.
    Uses the same unified algorithm — no more divergence.
    """
    conf = {}
    seed_names = [s[0] for s in seeds]

    for i, (name, _) in enumerate(seeds):
        if i == len(seeds) - 1:
            conf[name] = 55  # last seed has no one below to compare
            continue

        nxt = seed_names[i + 1]
        comp = compare_breakdown(name, nxt, history, sos, top_wins, bad_losses, power_scores)

        # Advantage of +10 → 100% confidence, -10 → 0% confidence
        raw = 50 + comp["advantage"] * 4
        conf[name] = max(10, min(97, int(raw)))

    return conf

def build_alerts(seeds, history, sos, top_wins, bad_losses, confidence, power_scores):
    controversy_flags = []
    upset_alerts      = []
    debate_queue      = []
    names = [s[0] for s in seeds]

    for i, name in enumerate(names):
        # Upset potential: low seed with top wins or strong SOS
        if i >= 5 and (len(top_wins[name]) >= 2 or sos[name] >= 0.55):
            upset_alerts.append(
                f"{name} (seed {i+1}) is dangerous: "
                f"{len(top_wins[name])} top wins, SOS {sos[name]:.3f}"
            )

        if i < len(names) - 1:
            nxt  = names[i + 1]
            comp = compare_breakdown(name, nxt, history, sos, top_wins, bad_losses, power_scores)

            # Flag if algorithm prefers the lower seed
            if comp["winner"] == nxt:
                controversy_flags.append(
                    f"Seed {i+1} vs {i+2}: algorithm favors {nxt} over {name} — review recommended"
                )

            # Debate queue: close matchup or algorithm is uncertain
            if abs(comp["advantage"]) <= 2 or comp["winner"] == "Too close":
                debate_queue.append({
                    "higher": name,
                    "lower":  nxt,
                    "reason": comp["reasons"][0] if comp["reasons"] else "No clear separator",
                    "advantage": comp["advantage"],
                })

    return controversy_flags, upset_alerts, debate_queue

def detect_circular_h2h(history, field_names, sos, power_scores):
    """
    Find groups of 3 field wrestlers where A beat B, B beat C, C beat A.
    Resolve each circle by:
      1. Common opponents (most shared-opponent wins)
      2. Win quality (avg method points)
      3. Strength of schedule
    Returns list of dicts: {circle, resolution_order, resolved_by}.
    """
    field = [n for n in field_names if n in history]
    seen  = set()
    circles = []

    for a in field:
        for b in field:
            if b == a or b not in history[a]["wins"]:
                continue
            for c in field:
                if c == a or c == b:
                    continue
                if c not in history[b]["wins"]:
                    continue
                if a not in history[c]["wins"]:
                    continue
                key = frozenset([a, b, c])
                if key in seen:
                    continue
                seen.add(key)

                wrestlers = [a, b, c]

                # 1. Common opponents — count wins against opponents all three faced
                shared = (
                    set(history[a]["wins"] + history[a]["losses"]) &
                    set(history[b]["wins"] + history[b]["losses"]) &
                    set(history[c]["wins"] + history[c]["losses"])
                )
                common_wins = {
                    w: sum(1 for o in shared if o in history[w]["wins"])
                    for w in wrestlers
                }
                if len(set(common_wins.values())) > 1:
                    resolved = sorted(wrestlers, key=lambda w: common_wins[w], reverse=True)
                    method   = "common opponents"
                else:
                    # 2. Win quality
                    wq = {w: avg_win_quality(w, history) for w in wrestlers}
                    if len({round(v, 1) for v in wq.values()}) > 1:
                        resolved = sorted(wrestlers, key=lambda w: wq[w], reverse=True)
                        method   = "win quality"
                    else:
                        # 3. SOS
                        resolved = sorted(wrestlers, key=lambda w: sos.get(w, 0), reverse=True)
                        method   = "strength of schedule"

                circles.append({
                    "circle":           f"{a} beat {b}, {b} beat {c}, {c} beat {a}",
                    "resolution_order": resolved,
                    "resolved_by":      method,
                })

    return circles

def build_field_top_half_matches(full_history, seeds, field_names):
    """
    For each field wrestler, count how many unique opponents they faced this
    season who ranked in the top half of the seeded field.
    Returns {wrestler_name: count}.
    """
    n        = len(seeds)
    top_half = {name for i, (name, _) in enumerate(seeds) if i < (n + 1) // 2}
    result   = {}
    for name in field_names:
        if name not in full_history:
            result[name] = 0
            continue
        opps = set(full_history[name]["wins"] + full_history[name]["losses"])
        result[name] = len(opps & top_half)
    return result

# —————————

# upload

# —————————

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

            history      = build_history(gm)
            sos          = build_sos(history)
            top_wins, bad_losses = build_quality(history)

            match_counts = [
                len(history[n]["wins"]) + len(history[n]["losses"])
                for n in history
            ]
            field_match_avg = sum(match_counts) / len(match_counts) if match_counts else None

            power_scores = build_power_scores(
                history, sos, top_wins, bad_losses,
                field_match_avg=field_match_avg,
            )
            seeds        = compute_rankings(power_scores)
            common       = build_common(history)
            win_methods  = build_win_method_summary(history)

            confidence   = build_confidence(seeds, history, sos, top_wins, bad_losses, power_scores)
            controversy_flags, upset_alerts, debate_queue = build_alerts(
                seeds, history, sos, top_wins, bad_losses, confidence, power_scores
            )
            field_names_csv  = set(history.keys())
            circular_h2h     = detect_circular_h2h(history, field_names_csv, sos, power_scores)
            matches_vs_top_half = build_field_top_half_matches(history, seeds, field_names_csv)

            results[str(weight)] = {
                "seeds":               [(i + 1, w[0]) for i, w in enumerate(seeds)],
                "history":             history,
                "sos":                 sos,
                "common":              common,
                "confidence":          confidence,
                "top_wins":            top_wins,
                "bad_losses":          bad_losses,
                "power_scores":        {k: round(v, 1) for k, v in power_scores.items()},
                "win_methods":         win_methods,
                "controversy_flags":   controversy_flags,
                "upset_alerts":        upset_alerts,
                "debate_queue":        debate_queue,
                "circular_h2h":        circular_h2h,
                "matches_vs_top_half": matches_vs_top_half,
                "field_match_avg":     round(field_match_avg, 1) if field_match_avg else None,
            }

        return JSONResponse(results)

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

# —————————

# live meeting endpoints

# —————————

@app.post("/session/{weight}")
def create_session(weight: str):
    sessions[weight] = {
    "votes":           {},
    "voters":          {},
    "current_matchup": None,
    }
    return {"status": "created"}

@app.post("/session/{weight}/matchup")
async def set_matchup(weight: str, request: Request):
    data = await request.json()
    a = data.get("a")
    b = data.get("b")
    sessions[weight] = {
    "votes":           {a: 0, b: 0},
    "voters":          {},
    "current_matchup": [a, b],
    }
    return sessions[weight]

@app.post("/vote/{weight}")
async def vote(weight: str, request: Request):
    data   = await request.json()
    name   = data.get("name")
    voter  = data.get("voter", "Anonymous")
    role   = data.get("role", "coach")

    sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})

    # one vote per voter per matchup
    if voter in sessions[weight]["voters"]:
        prev        = sessions[weight]["voters"][voter]["name"]
        prev_weight = sessions[weight]["voters"][voter]["weight"]
        if prev in sessions[weight]["votes"]:
            sessions[weight]["votes"][prev] -= prev_weight

    vote_weight = 2 if role == "head_coach" else 1
    sessions[weight]["votes"].setdefault(name, 0)
    sessions[weight]["votes"][name] += vote_weight
    sessions[weight]["voters"][voter] = {
        "name":   name,
        "weight": vote_weight,
        "role":   role,
    }

    return sessions[weight]

@app.get("/votes/{weight}")
def get_votes(weight: str):
    return sessions.setdefault(weight, {"votes": {}, "voters": {}, "current_matchup": None})

# —————————

# FloWrestling Scraper

# —————————

import httpx
import os
from datetime import datetime, timedelta
import re
import asyncio
import json
import math
from bs4 import BeautifulSoup

FLO_BASE = "https://flowrestling.org"
FLO_GRAPHQL = "https://www.flowrestling.org/api/graphql"
FLO_SEARCH_URL = "https://prod-web-api.flowrestling.org/api/search"

WRESTLER_RESULTS_QUERY = """
query GetAthleteResults($personId: String!) {
  person(personId: $personId) {
    id
    firstName
    lastName
    bouts(first: 200) {
      edges {
        node {
          id
          boutResult
          winType
          opponentFirstName
          opponentLastName
          weightClass {
            name
          }
          event {
            name
            startDate
          }
        }
      }
    }
  }
}
"""

# Season runs November through March

# e.g. 2025-26 season = Nov 2025 - Mar 2026

def get_season_range(season_str=None):
    now = datetime.now()
    if season_str:
        # Expect format "2025-26" or "2025"
        try:
            start_year = int(season_str.split("-")[0])
        except:
            start_year = now.year if now.month >= 11 else now.year - 1
    else:
        # Auto-detect current season
        start_year = now.year if now.month >= 11 else now.year - 1
    end_year = start_year + 1
    season_start = datetime(start_year, 11, 1)
    season_end   = datetime(end_year,   3, 31)
    return season_start, season_end

def extract_flo_id(url_or_id):
    url_or_id = url_or_id.strip()
    # Already a bare ID (alphanumeric, no slash)
    if re.match(r'^[A-Za-z0-9_-]+$', url_or_id) and '/' not in url_or_id:
        return url_or_id
    # Extract from URL path: /people/XXXX
    m = re.search(r'/people/([A-Za-z0-9_-]+)', url_or_id)
    if m:
        return m.group(1)
    return None

def map_flo_method(raw):
    if not raw:
        return "dec"
    s = raw.strip().upper()
    if s in ("F", "FALL", "PIN"):
        return "fall"
    if s in ("TF", "TECH", "TECHNICAL FALL", "TECH FALL"):
        return "tf"
    if s in ("MD", "MAJOR", "MAJOR DECISION", "MAJ"):
        return "md"
    if s in ("DEC", "DECISION", "D"):
        return "dec"
    if s in ("DQ", "DEFAULT", "FORFEIT", "FOR"):
        return "dec"
    return "dec"

def parse_weight_from_text(text):
    # Look for standalone numbers that look like weight classes
    m = re.search(r'\b(100|106|110|113|119|120|125|126|132|133|138|139|144|145|150|152|157|160|165|170|175|182|183|189|195|196|215|220|225|235|285|HWT)\b', str(text))
    if m:
        return m.group(1)
    return None

FLO_API_BASE = "https://prod-web-api.flowrestling.org/api"

FLO_API_HEADERS = {
    "accept": "application/json",
    "origin": "https://www.flowrestling.org",
    "referer": "https://www.flowrestling.org/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}

WIN_TYPE_MAP = {
    "F":   "fall",
    "TF":  "tf",
    "MD":  "md",
    "DEC": "dec",
    "FOR": "dec",
}

async def fetch_flo_profile(wrestler_id, season=None):
    season_start, season_end = get_season_range(season)
    matches = []
    wrestler_name = None
    weight_class = None

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # Fetch wrestler name
            athlete_resp = await client.get(
                f"{FLO_API_BASE}/athletes/{wrestler_id}",
                headers=FLO_API_HEADERS,
            )
            athlete_resp.raise_for_status()
            athlete_data = athlete_resp.json()
            profile_data = athlete_data.get("data") or athlete_data
            wrestler_name = (
                profile_data.get("name") or
                " ".join(filter(None, [
                    profile_data.get("firstName"),
                    profile_data.get("lastName"),
                ])).strip() or
                wrestler_id
            )

            # Fetch match results
            results_resp = await client.get(
                f"{FLO_API_BASE}/athletes/{wrestler_id}/results",
                headers=FLO_API_HEADERS,
            )
            results_resp.raise_for_status()
            results_data = results_resp.json()

        for tournament in results_data.get("data") or []:
            tournament_name = tournament.get("name") or "FloWrestling"
            for bout in tournament.get("boutResults") or []:
                opponent = (bout.get("opponent") or {}).get("name") or ""
                if not opponent:
                    continue

                won = bool((bout.get("athlete") or {}).get("isWinner"))
                win_type = str(bout.get("winType") or "").upper()
                method = WIN_TYPE_MAP.get(win_type, "dec")

                wc = bout.get("weight") or ""
                if wc and not weight_class:
                    weight_class = parse_weight_from_text(str(wc))

                date_str = bout.get("date") or ""
                match_date = None
                if date_str:
                    try:
                        match_date = datetime.fromisoformat(date_str.rstrip("Z").split("T")[0])
                    except Exception:
                        pass

                in_season = True
                if match_date:
                    in_season = season_start <= match_date <= season_end

                if in_season:
                    matches.append({
                        "opponent": opponent,
                        "method": method,
                        "won": won,
                        "tournament": tournament_name,
                        "date": match_date.strftime("%b %d, %Y") if match_date else None,
                        "score": None,
                    })

    except Exception as e:
        return {
            "wrestler_id": wrestler_id,
            "wrestler_name": wrestler_name,
            "weight_class": weight_class,
            "matches": [],
            "match_count": 0,
            "error": str(e),
        }

    return {
        "wrestler_id": wrestler_id,
        "wrestler_name": wrestler_name,
        "weight_class": weight_class,
        "matches": matches,
        "match_count": len(matches),
        "error": None,
    }

def build_seedit_matches(profiles, weight_override=None, field_names=None):
    """
    Build match rows from submitted wrestler profiles.

    field_names: if provided, only emit rows where BOTH wrestlers are in
    the set (head-to-head matches only). When None, all matches are emitted
    (used for full SOS / quality calculations).
    """
    seedit_matches = []

    for p in profiles:
        wrestler_name = p.get("wrestler_name") or p.get("wrestler_id")
        weight = weight_override or p.get("weight_class") or "Unknown"

        for m in p.get("matches", []):
            opp = m.get("opponent")
            method = m.get("method", "dec")
            won = m.get("won")
            if not opp or won is None:
                continue
            # When field_names is set, skip matches against non-field opponents
            if field_names is not None and opp not in field_names:
                continue
            winner = wrestler_name if won else opp
            seedit_matches.append({
                "weight":    str(weight).replace(" lbs", "").strip(),
                "wrestlerA": wrestler_name,
                "wrestlerB": opp,
                "winner":    winner,
                "method":    method,
                "date":      m.get("date"),
            })

    return seedit_matches

# —————————

# FloWrestling scrape endpoints

# —————————

@app.post("/scrape/flo")
async def scrape_flo(request: Request):
    """
    Scrape FloWrestling profiles for a group of wrestlers and seed them.

    ```
    Body:
    {
        "wrestlers": [
            {"id": "0QBPq4IRIQuIWK71", "name": "Luke Heinen"},
            {"url": "flowrestling.org/nextgen/people/ABC123", "name": "..."},
            ...
        ],
        "weight": "132",          // optional override
        "season": "2025-26"       // optional, defaults to current season
    }
    """
    try:
        body = await request.json()
        wrestler_entries = body.get("wrestlers", [])
        weight_override = body.get("weight", None)
        season = body.get("season", None)

        if not wrestler_entries:
            return JSONResponse({"error": "No wrestlers provided"}, status_code=400)

        profiles = []
        errors = []

        tasks = []
        for entry in wrestler_entries:
            raw = entry.get("url") or entry.get("id") or ""
            flo_id = extract_flo_id(raw)
            if not flo_id:
                errors.append(f"Could not parse ID from: {raw}")
                continue
            tasks.append(fetch_flo_profile(flo_id, season=season))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_entries = [e for e in wrestler_entries if extract_flo_id(e.get("url") or e.get("id") or "")]
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                errors.append(f"Error fetching profile {i}: {str(r)}")
            else:
                # Use manually provided name if API returned only the ID as fallback
                entry = valid_entries[i] if i < len(valid_entries) else {}
                if entry.get("name") and (
                    not r.get("wrestler_name") or r.get("wrestler_name") == r.get("wrestler_id")
                ):
                    r["wrestler_name"] = entry["name"]
                profiles.append(r)

        # Field wrestlers: only these appear as seeds
        field_names = {
            p.get("wrestler_name") or p.get("wrestler_id")
            for p in profiles
        }

        # All matches: used to build full season history for SOS / quality / common opponents
        all_matches = build_seedit_matches(profiles, weight_override)
        # H2H matches: only submitted-vs-submitted, used for seeding structure
        h2h_matches = build_seedit_matches(profiles, weight_override, field_names=field_names)

        if not all_matches:
            return JSONResponse({
                "error": "No match data found for the current season",
                "profiles": profiles,
                "errors": errors,
                "tip": "Check that the FloWrestling IDs are correct and wrestlers have current season results"
            }, status_code=200)

        results_out = _run_seeding_engine(
            profiles, all_matches, h2h_matches, field_names,
            source="flo", errors=errors,
        )
        return JSONResponse(sanitize_for_json(results_out))

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/scrape/flo/preview/{wrestler_id}")
async def preview_flo_wrestler(wrestler_id: str, season: str = None):
    """
    Preview a single wrestler's scraped data before running full seeding.
    Useful for confirming the right wrestler was found.
    GET /scrape/flo/preview/0QBPq4IRIQuIWK71?season=2025-26
    """
    try:
        profile = await fetch_flo_profile(wrestler_id, season=season)
        return JSONResponse(profile)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# —————————

# Seeding engine helper (shared by /scrape/flo and /scrape/combined)

# —————————

def _run_seeding_engine(profiles, all_matches, h2h_matches, field_names,
                        source="flo", errors=None):
    """
    Run the seeding engine over a set of profiles and return results_out dict.

    all_matches  — every match row (submitted wrestler vs any opponent); used
                   for full-season SOS, quality, and common-opponent data.
    h2h_matches  — only submitted-vs-submitted rows; used for seeding structure.
    field_names  — set of submitted wrestler names; only these appear as seeds.
    """
    if errors is None:
        errors = []

    df_all = pd.DataFrame(all_matches)
    results_out = {}

    for weight, all_group in df_all.groupby("weight"):
        all_gm = all_group.to_dict(orient="records")
        # pandas converts None -> float NaN in object columns; convert back to None
        for row in all_gm:
            if isinstance(row.get("date"), float):
                row["date"] = None

        # Full history: all opponents for accurate SOS / quality / common-opponent.
        full_history = build_history(all_gm)
        full_sos = build_sos(full_history)
        full_top_wins, full_bad_losses = build_quality(full_history)

        # H2H history: only submitted-vs-submitted matches.
        # Guarantee every field wrestler appears even with 0 H2H results.
        h2h_gm = [m for m in h2h_matches
                  if str(m.get("weight", "")).strip() == str(weight).strip()]
        h2h_history = build_history(h2h_gm)
        for fname in field_names:
            h2h_history.setdefault(fname, {
                "wins": [], "losses": [],
                "win_methods": {}, "loss_methods": {},
                "win_dates": {}, "loss_dates": {},
            })

        # Field-relative average match count for SOS confidence weighting.
        field_match_counts = [
            len(h2h_history[n]["wins"]) + len(h2h_history[n]["losses"])
            for n in field_names if n in h2h_history
        ]
        field_match_avg = (
            sum(field_match_counts) / len(field_match_counts)
            if field_match_counts else None
        )

        # Power scores and seeds: only field wrestlers, SOS/quality from full data.
        power_scores = build_power_scores(
            h2h_history, full_sos, full_top_wins, full_bad_losses,
            field_match_avg=field_match_avg,
        )
        seeds = compute_rankings(power_scores)

        # Common opponents from full season; outer/inner keys filtered to field.
        common = build_common(full_history)
        common_out = {
            k: {k2: v2 for k2, v2 in v.items() if k2 in field_names}
            for k, v in common.items() if k in field_names
        }

        win_methods = build_win_method_summary(full_history)

        # Confidence and alerts use full_history so H2H and common-opponent
        # comparisons draw on the complete season record.
        confidence = build_confidence(
            seeds, full_history, full_sos, full_top_wins, full_bad_losses, power_scores
        )
        controversy_flags, upset_alerts, debate_queue = build_alerts(
            seeds, full_history, full_sos, full_top_wins, full_bad_losses,
            confidence, power_scores
        )

        # Circular H2H detection and resolution.
        circular_h2h = detect_circular_h2h(h2h_history, field_names, full_sos, power_scores)

        # Matches vs top half of field (uses full season history).
        matches_vs_top_half = build_field_top_half_matches(full_history, seeds, field_names)

        def fs(d):
            return {k: v for k, v in d.items() if k in field_names}

        results_out[str(weight)] = {
            "seeds":                 [(i + 1, w[0]) for i, w in enumerate(seeds)],
            "history":               fs(full_history),
            "sos":                   fs(full_sos),
            "common":                common_out,
            "confidence":            confidence,
            "top_wins":              fs(full_top_wins),
            "bad_losses":            fs(full_bad_losses),
            "power_scores":          {k: round(v, 1) for k, v in power_scores.items()},
            "win_methods":           fs(win_methods),
            "controversy_flags":     controversy_flags,
            "upset_alerts":          upset_alerts,
            "debate_queue":          debate_queue,
            "circular_h2h":          circular_h2h,
            "matches_vs_top_half":   matches_vs_top_half,
            "field_match_avg":       round(field_match_avg, 1) if field_match_avg else None,
            "source":                source,
            "profiles_scraped":      len(profiles),
            "matches_processed":     len(h2h_matches),
            "errors":                errors,
            "profiles_index":        {
                p["wrestler_name"]: {
                    "flo_record":  p.get("flo_record"),
                    "usab_record": p.get("usab_record"),
                }
                for p in profiles
                if p.get("wrestler_name")
            },
        }

    return results_out

# —————————

# Multi-source helpers

# —————————

USAB_API_BASE = "https://www.usabracketing.com"
USAB_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


async def _usab_login(client):
    """
    Log in to USA Bracketing using the shared httpx client (preserves cookies).
    Returns True if login succeeded (redirected away from /login).
    """
    login_field = os.environ.get("USB_USERNAME", "")
    password    = os.environ.get("USB_PASSWORD", "")
    if not login_field or not password:
        print("=== USAB _usab_login: no credentials configured ===")
        return False

    login_page = await client.get(f"{USAB_API_BASE}/login", headers=USAB_HEADERS)
    html = login_page.text
    token_match = (
        re.search(r'<input[^>]+name="_token"[^>]+value="([^"]+)"', html) or
        re.search(r'<input[^>]+value="([^"]+)"[^>]+name="_token"', html) or
        re.search(r'<meta[^>]+name="csrf-token"[^>]+content="([^"]+)"', html)
    )
    csrf_token = token_match.group(1) if token_match else ""
    print(f"=== USAB _usab_login: login_page_status={login_page.status_code} csrf_token={csrf_token!r} ===")

    login_resp = await client.post(
        f"{USAB_API_BASE}/login",
        data={
            "_token":   csrf_token,
            "login":    login_field,
            "password": password,
            "remember": "on",
        },
        headers=dict(USAB_HEADERS, **{"content-type": "application/x-www-form-urlencoded"}),
    )
    landed    = str(login_resp.url)
    logged_in = "/login" not in landed
    cookies   = list(client.cookies.keys())
    print(f"=== USAB _usab_login: status={login_resp.status_code} landed={landed!r} logged_in={logged_in} cookies={cookies} ===")
    if not logged_in:
        print(f"=== USAB _usab_login: LOGIN FAILED — body (first 300):\n{login_resp.text[:300]} ===")
    return logged_in


def _parse_usab_matches_html(html, event_name, season_start, season_end,
                             wrestler_name=None, debug=False):
    """
    Parse bout results from a USAB Livewire HTML fragment.

    Bout format: "WINNER , TEAM over LOSER , TEAM ( METHOD SCORE )"
    Win/loss is determined by splitting on the last " over " in the bout line
    and checking whether wrestler_name appears before (win) or after (loss).

    Returns a list of match dicts compatible with merge_season_matches.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Extract event date from the header (MM/DD/YYYY or MM/DD - MM/DD/YYYY) ──
    # Find all full dates in the HTML; use the first one as the event date.
    event_date = None
    for dm in re.finditer(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', html):
        try:
            event_date = datetime.strptime(dm.group(1), "%m/%d/%Y")
            break
        except ValueError:
            pass
    print(f"=== USAB _parse_usab_matches_html: event={event_name!r} event_date={event_date} ===")

    # Join all visible text with " | " — used both for debug and as fallback
    # when the DOM walk returns a truncated mat-number prefix.
    all_text = soup.get_text(" | ", strip=True)
    if debug:
        print(f"=== USAB _parse_usab_matches_html DEBUG all_text:\n{all_text[:3000]} ===")
        athlete_links = [a["href"] for a in soup.find_all("a", href=True) if "/athletes/" in a["href"]]
        print(f"=== USAB _parse_usab_matches_html DEBUG athlete hrefs: {athlete_links} ===")

    result_re = re.compile(
        r'\b(Dec|TF|MD|Fall|Forfeit|For\.?|Inj\.?\s*Def\.?)\b'
        r'(?:\s+(\d{1,3}[-–]\d{1,3}))?',
        re.IGNORECASE,
    )
    over_re = re.compile(r'\bover\b', re.IGNORECASE)

    # Pre-compute name tokens for W/L detection
    # USAB may format as "Lastname, Firstname" or "Firstname Lastname" or abbreviate
    wrestler_norm = wrestler_name.lower().strip() if wrestler_name else ""
    if wrestler_norm:
        if "," in wrestler_norm:
            # "Heinen, Luke" → last="heinen", first="luke"
            parts = [p.strip() for p in wrestler_norm.split(",", 1)]
            wname_last  = parts[0]
            wname_first = parts[1].split()[0] if parts[1] else ""
        else:
            # "Luke Heinen" → last="heinen", first="luke"
            tokens = wrestler_norm.split()
            wname_last  = tokens[-1] if tokens else ""
            wname_first = tokens[0]  if len(tokens) > 1 else ""
    else:
        wname_last = wname_first = ""

    # Pre-build ordered list of bout blocks by splitting all_text on "Bout " (literal).
    # Each block is the text of one bout: "965 - Cons. Sub-Quarters: | Tae Berry | , SCP over | Luke Heinen | ..."
    # Using the literal "Bout " (with space) avoids false splits on other words.
    bout_blocks = re.split(r'Bout\s+', all_text, flags=re.IGNORECASE)
    # claimed_blocks tracks which blocks have been matched to prevent double-counting
    # the same bout when an opponent link appears more than once in the HTML.
    claimed_blocks = set()

    # Count all athlete hrefs up front for diagnostics
    all_athlete_hrefs = [a for a in soup.find_all("a", href=True) if "/athletes/" in a["href"]]
    print(f"=== USAB parser: event={event_name!r} bout_blocks={len(bout_blocks)} "
          f"athlete_hrefs={len(all_athlete_hrefs)} ===")

    matches        = []
    season_pass    = 0
    season_fail    = 0
    bout_log_count = 0

    # Iterate over every athlete profile link — each unique unclamed bout = one match.
    # NOTE: seen_opps intentionally removed so the same opponent can appear multiple
    # times (consolation rematches, true-seconds, etc.) as separate bouts.
    for a_tag in soup.find_all("a", href=True):
        if "/athletes/" not in a_tag["href"]:
            continue
        opp = a_tag.get_text(strip=True)
        if not opp or opp.lower() in ("view", "profile", "details", "more", ""):
            continue

        # Skip self-matches: wrestler's own profile link appears on their page
        opp_key = opp.lower()
        if wname_last and wname_last in opp_key:
            if not wname_first or wname_first in opp_key:
                print(f"=== USAB parser: skipping self-match opp={opp!r} (wrestler={wrestler_name!r}) ===")
                continue

        # ── Find the first unclaimed bout block containing this opponent AND "over" ──
        # Each block is already one complete bout, so both names and "over" are present.
        # We claim the block so a second href for the same opponent doesn't reuse it.
        bout_block = ""
        block_idx  = -1
        for i, blk in enumerate(bout_blocks):
            if i in claimed_blocks:
                continue
            if opp_key in blk.lower() and over_re.search(blk):
                bout_block = blk
                block_idx  = i
                break

        if not bout_block:
            # Diagnostic: show exactly what the link text is and where the name
            # appears (if at all) in the raw all_text to reveal format mismatches.
            all_text_lower = all_text.lower()
            idx = all_text_lower.find(opp_key)
            if idx != -1:
                ctx = all_text[max(0, idx - 60): idx + 120]
                print(f"=== USAB parser: no bout block for opp={opp!r} "
                      f"(link_text={opp!r}) but found in all_text: {ctx!r} ===")
            else:
                # Try each word of the name individually
                words = opp_key.split()
                word_hits = {w: all_text_lower.find(w) for w in words if len(w) > 2}
                print(f"=== USAB parser: opp={opp!r} NOT in all_text at all; "
                      f"word hits={word_hits} ===")
            # Also print every bout block that contains any word of the opponent name
            for i, blk in enumerate(bout_blocks):
                blk_lower = blk.lower()
                if any(w in blk_lower for w in opp_key.split() if len(w) > 2):
                    print(f"=== USAB parser: partial-match block[{i}]={blk[:200]!r} ===")
            continue

        claimed_blocks.add(block_idx)

        # ── Method: read from the bout block (no DOM walk needed) ────────────
        rm = result_re.search(bout_block)
        if not rm:
            if debug:
                print(f"=== USAB parser: no result token in bout block for opp={opp!r}, skipping ===")
            continue

        raw_method = rm.group(1).lower().replace(".", "").replace(" ", "")
        score      = rm.group(2) or None
        if "fall" in raw_method:
            method = "fall"
        elif raw_method == "tf":
            method = "tf"
        elif raw_method == "md":
            method = "md"
        elif raw_method in ("for", "forfeit"):
            method = "dec"
        else:
            method = "dec"

        # ── Win/loss: use the FIRST "over" in the bout block ─────────────────
        # Each block covers exactly one bout, so the first "over" is the decisive one.
        # Format: "Winner , Team over Loser , Team ( Method Score )"
        won       = False
        wl_method = "no-over-default-loss"
        if wname_last:
            m_over = over_re.search(bout_block)
            if m_over:
                before = bout_block[:m_over.start()].lower()
                after  = bout_block[m_over.end():].lower()

                if wname_last in before and (not wname_first or wname_first in before):
                    won = True
                    wl_method = "both-names-before"
                elif wname_last in after and (not wname_first or wname_first in after):
                    won = False
                    wl_method = "both-names-after"
                elif wname_last in before:
                    won = True
                    wl_method = "last-name-before"
                elif wname_last in after:
                    won = False
                    wl_method = "last-name-after"

        # Verification logging: print every bout result
        bout_log_count += 1
        print(f"=== USAB bout[{bout_log_count}]: opp={opp!r} won={won} "
              f"wl_method={wl_method!r} bout_block={bout_block[:140]!r} ===")

        # ── Date + season filter ──────────────────────────────────────────────
        # Try ISO date in the bout line first; fall back to the event header date.
        match_date = None
        dm = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', bout_block)
        if dm:
            try:
                match_date = datetime.strptime(dm.group(1), "%Y-%m-%d")
            except Exception:
                pass
        if match_date is None:
            # Also try MM/DD/YYYY in the bout line
            dm2 = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', bout_block)
            if dm2:
                try:
                    match_date = datetime.strptime(dm2.group(1), "%m/%d/%Y")
                except Exception:
                    pass
        if match_date is None:
            match_date = event_date  # use header date extracted at top of function

        if match_date:
            if season_start <= match_date <= season_end:
                season_pass += 1
            else:
                season_fail += 1
                print(f"=== USAB season filter: EXCLUDED {match_date.date()} vs {opp!r} "
                      f"(window {season_start.date()}–{season_end.date()}) ===")
                continue
        # No date at all → include (can't filter what we can't parse)

        matches.append({
            "opponent":   opp,
            "method":     method,
            "won":        bool(won),
            "tournament": str(event_name or "USA Bracketing"),
            "date":       match_date.strftime("%b %d, %Y") if match_date else None,
            "score":      score,
        })

    print(f"=== USAB _parse_usab_matches_html: event={event_name!r} "
          f"kept={len(matches)} season_pass={season_pass} season_fail={season_fail} ===")
    return matches


def _norm_name(s):
    """Lowercase and strip punctuation for fuzzy opponent name matching."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _norm_tournament(s):
    """Lowercase, strip year and punctuation for tournament dedup."""
    if not s:
        return ""
    s = re.sub(r"\b20\d{2}\b", "", s.lower())
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_date_str(s):
    """Parse a date string to datetime. Returns None on failure."""
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            pass
    return None


def _dates_close(d1_str, d2_str, days=2):
    """Return True if two date strings are within `days` of each other."""
    d1 = _parse_date_str(d1_str)
    d2 = _parse_date_str(d2_str)
    if d1 is None or d2 is None:
        return False
    return abs((d1 - d2).days) <= days


def merge_season_matches(flo_matches, usab_matches):
    """
    Merge FloWrestling and USA Bracketing match lists, deduplicating bouts that
    appear on both platforms.

    Dedup key: normalized tournament name overlap + normalized opponent name
               match + same win/loss result + date within 2 days.
    Prefers FloWrestling data when a duplicate is detected.
    """
    merged = [dict(m, source="flo") for m in flo_matches]

    for um in usab_matches:
        u_tourn = _norm_tournament(um.get("tournament", ""))
        u_opp   = _norm_name(um.get("opponent", ""))
        u_won   = bool(um.get("won"))
        u_date  = um.get("date")

        is_dupe = False
        for fm in flo_matches:
            f_tourn = _norm_tournament(fm.get("tournament", ""))
            f_opp   = _norm_name(fm.get("opponent", ""))
            f_won   = bool(fm.get("won"))
            f_date  = fm.get("date")

            opp_match    = u_opp and f_opp and (u_opp == f_opp or
                           u_opp in f_opp or f_opp in u_opp)
            result_match = u_won == f_won
            tourn_match  = u_tourn and f_tourn and (
                u_tourn == f_tourn or u_tourn in f_tourn or f_tourn in u_tourn
            )
            date_match   = _dates_close(u_date, f_date, days=2)

            if opp_match and result_match and (tourn_match or date_match):
                is_dupe = True
                break

        if not is_dupe:
            merged.append(dict(um, source="usab"))

    return merged


async def search_flo_athlete(name, weight=None):
    """
    Search FloWrestling for a wrestler by name.
    POST /api/search — HAR-verified payload: domain, search, offsetPerGroup, limitPerGroup, entities.
    Response: data[].items[] with ofpId, title ("Lastname, Firstname"), metadata2 (location).
    Returns up to 5 matches as {source, id, name, location, weight} dicts.
    Weight is only used as a label — not a search filter.
    """
    headers = dict(FLO_API_HEADERS)
    headers["content-type"] = "application/json"
    payload = {
        "domain": "11FloWrestling000",
        "offsetPerGroup": 0,
        "search": name,
        "limitPerGroup": 20,
        "entities": ["person"],
    }
    print("=== search_flo_athlete DEBUG ===")
    print(f"URL:     {FLO_SEARCH_URL}")
    print(f"PAYLOAD: {json.dumps(payload)}")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(FLO_SEARCH_URL, json=payload, headers=headers)
            print(f"STATUS:  {resp.status_code}")
            print(f"RESPONSE HEADERS: {dict(resp.headers)}")
            print(f"RESPONSE BODY:\n{resp.text}")
            resp.raise_for_status()
            data = resp.json()

        results = []
        for section in (data.get("data") or []):
            for item in (section.get("items") or []):
                ofp_id = item.get("ofpId")
                if not ofp_id:
                    continue
                title = item.get("title") or ""
                # title is "Lastname, Firstname" — convert to "Firstname Lastname"
                parts = [p.strip() for p in title.split(",", 1)]
                display_name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else title
                results.append({
                    "source":   "flo",
                    "id":       str(ofp_id),
                    "name":     display_name,
                    "location": item.get("metadata2") or "",
                    "weight":   str(weight or ""),
                })
                if len(results) >= 5:
                    break
            if len(results) >= 5:
                break
        return results
    except Exception as e:
        return [{"source": "flo", "error": str(e)}]


async def search_usab_athlete(name, weight=None, client=None):
    """
    Search USA Bracketing for a wrestler by name.
    Results are JS-rendered so this is rarely useful; search is skipped in /search/athlete.
    Kept for any internal callers that still need it.
    """
    own_client = client is None
    try:
        if own_client:
            client = httpx.AsyncClient(timeout=15, follow_redirects=True)

        await _usab_login(client)

        parts      = name.strip().split(None, 1)
        first_name = parts[0] if parts else name
        last_name  = parts[1] if len(parts) > 1 else ""
        search_resp = await client.get(
            f"{USAB_API_BASE}/athletes",
            params={"first_name": first_name, "last_name": last_name},
            headers=USAB_HEADERS,
        )
        print(f"=== USAB search status={search_resp.status_code} final_url={search_resp.url} ===")

        soup    = BeautifulSoup(search_resp.text, "html.parser")
        uuid_re = re.compile(r'/athletes/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', re.IGNORECASE)
        seen    = set()
        results = []
        for a in soup.find_all("a", href=True):
            m = uuid_re.search(a["href"])
            if not m:
                continue
            uid = m.group(1)
            if uid in seen:
                continue
            seen.add(uid)
            raw_name = a.get_text(" ", strip=True)
            if not raw_name or raw_name.lower() in ("view", "profile", "details", "more"):
                card = a.find_parent(["div", "article", "li", "tr"])
                if card:
                    heading = card.find(["h1", "h2", "h3", "h4", "strong", "b"])
                    raw_name = heading.get_text(" ", strip=True) if heading else ""
            if not raw_name:
                continue
            location = ""
            card = a.find_parent(["div", "article", "li", "tr"])
            if card:
                for el in card.find_all(["span", "div", "td"]):
                    txt = el.get_text(strip=True)
                    if re.fullmatch(r'[A-Z]{2}', txt):
                        location = txt
                        break
            results.append({"source": "usab", "id": uid, "name": raw_name,
                             "location": location, "weight": str(weight or "")})
            if len(results) >= 5:
                break
        return results
    except Exception as e:
        return [{"source": "usab", "error": str(e)}]
    finally:
        if own_client and client:
            await client.aclose()


async def fetch_usab_profile(wrestler_id, season=None):
    """
    Fetch match results from USA Bracketing for a wrestler by UUID.
    Flow:
      1. Login via _usab_login
      2. GET /athletes/UUID — parse wrestler name and Livewire component snapshot
      3. For each event UUID in the snapshot, POST /livewire/update?showResults
      4. Parse HTML fragments from Livewire for bout rows (opponent link + W/L + method)
      5. Fall back to parsing the profile page HTML directly if no Livewire component found
    Returns the same shape as fetch_flo_profile.
    """
    print(f"=== USAB fetch_usab_profile: wrestler_id={wrestler_id!r} season={season!r} ===")
    season_start, season_end = get_season_range(season)
    matches       = []
    wrestler_name = None
    weight_class  = None

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            logged_in = await _usab_login(client)
            print(f"=== USAB fetch_usab_profile: logged_in={logged_in} ===")

            # Step 1: fetch the athlete profile page
            profile_url = f"{USAB_API_BASE}/athletes/{wrestler_id}"
            print(f"=== USAB fetch_usab_profile: GET {profile_url} ===")
            profile_resp = await client.get(profile_url, headers=USAB_HEADERS)
            print(f"=== USAB fetch_usab_profile: profile status={profile_resp.status_code} final_url={profile_resp.url} ===")
            print(f"=== USAB fetch_usab_profile: profile HTML (first 3000):\n{profile_resp.text[:3000]} ===")

            soup = BeautifulSoup(profile_resp.text, "html.parser")

            # Extract wrestler name from the page heading
            for sel in ["h1", "h2", ".athlete-name", ".profile-name", "[class*='name']"]:
                el = soup.select_one(sel)
                if el:
                    t = el.get_text(strip=True)
                    if t and len(t) > 2:
                        wrestler_name = t
                        break
            print(f"=== USAB fetch_usab_profile: wrestler_name={wrestler_name!r} ===")

            # Step 2: find Livewire component (v3 uses wire:snapshot, v2 uses wire:initial-data)
            lw_el = (
                soup.find(attrs={"wire:snapshot": True}) or
                soup.find(attrs={"wire:initial-data": True})
            )

            if lw_el:
                snapshot_str = lw_el.get("wire:snapshot") or lw_el.get("wire:initial-data") or ""
                wire_id      = lw_el.get("wire:id", "")
                print(f"=== USAB fetch_usab_profile: Livewire wire:id={wire_id!r} snapshot (first 500)={snapshot_str[:500]!r} ===")

                # Event UUIDs are in wire:click="showResults('UUID')" on the raw page HTML
                # (events render lazily, so UUIDs are NOT in the snapshot data itself)
                event_uuids = list(dict.fromkeys(re.findall(
                    r"showResults\('([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'\)",
                    profile_resp.text, re.IGNORECASE,
                )))
                print(f"=== USAB fetch_usab_profile: event UUIDs from wire:click: {event_uuids} ===")

                # Get CSRF token for Livewire AJAX requests
                meta_csrf  = soup.find("meta", attrs={"name": "csrf-token"})
                csrf_token = meta_csrf["content"] if meta_csrf else ""

                # Step 3: call showResults for each event via Livewire
                for ev_idx, event_uuid in enumerate(event_uuids[:15]):
                    is_first = ev_idx == 0
                    try:
                        lw_resp = await client.post(
                            f"{USAB_API_BASE}/livewire/update",
                            json={
                                "_token": csrf_token,
                                "components": [{
                                    "snapshot": snapshot_str,
                                    "updates":  {},
                                    "calls":    [{"path": "", "method": "showResults", "params": [event_uuid]}],
                                }],
                            },
                            headers=dict(USAB_HEADERS, **{
                                "accept":       "text/html, application/xhtml+xml",
                                "content-type": "application/json",
                                "x-csrf-token": csrf_token,
                                "x-livewire":   "true",
                                "referer":      profile_url,
                            }),
                        )
                        print(f"=== USAB Livewire showResults event={event_uuid} status={lw_resp.status_code} ===")
                        lw_text = lw_resp.text
                        if is_first:
                            print(f"=== USAB Livewire first-event raw response (first 3000):\n{lw_text[:3000]} ===")

                        # Extract effects.html from Livewire JSON response
                        # v3: {"components": [{"effects": {"html": "..."}}]}
                        # v2: {"effects": {"html": "..."}}
                        html_frag = ""
                        try:
                            lw_json = lw_resp.json()
                            comps   = lw_json.get("components") or []
                            if comps:
                                html_frag = (comps[0].get("effects") or {}).get("html") or ""
                            if not html_frag:
                                html_frag = (lw_json.get("effects") or {}).get("html") or ""
                        except Exception:
                            pass
                        if not html_frag:
                            html_frag = lw_text  # fallback: treat raw response as HTML

                        if is_first:
                            print(f"=== USAB Livewire first-event effects.html (first 3000):\n{html_frag[:3000]} ===")

                        event_matches = _parse_usab_matches_html(
                            html_frag, event_uuid, season_start, season_end,
                            wrestler_name=wrestler_name, debug=is_first,
                        )
                        print(f"=== USAB Livewire event={event_uuid} parsed {len(event_matches)} matches ===")
                        matches.extend(event_matches)
                    except Exception as lw_err:
                        print(f"=== USAB Livewire error for event {event_uuid}: {lw_err} ===")
            else:
                # No Livewire — try parsing match rows directly from the profile page
                print(f"=== USAB fetch_usab_profile: no Livewire component, parsing profile HTML directly ===")
                page_matches = _parse_usab_matches_html(
                    profile_resp.text, "USA Bracketing", season_start, season_end,
                    wrestler_name=wrestler_name,
                )
                print(f"=== USAB fetch_usab_profile: direct parse found {len(page_matches)} matches ===")
                matches.extend(page_matches)

            print(f"=== USAB fetch_usab_profile: TOTAL matches={len(matches)} ===")

    except Exception as e:
        import traceback
        print(f"=== USAB fetch_usab_profile ERROR: {e}\n{traceback.format_exc()} ===")
        return {
            "wrestler_id":   str(wrestler_id),
            "wrestler_name": wrestler_name,
            "weight_class":  weight_class,
            "matches":       [],
            "match_count":   0,
            "error":         str(e),
        }

    return {
        "wrestler_id":   str(wrestler_id),
        "wrestler_name": wrestler_name,
        "weight_class":  weight_class,
        "matches":       matches,
        "match_count":   len(matches),
        "error":         None,
    }

# —————————

# Multi-source endpoints

# —————————

@app.get("/search/athlete")
async def search_athlete(name: str, weight: str = None):
    """
    Search FloWrestling for a wrestler by name.
    USAB search is skipped (results load via JS on their site); TDs enter USAB UUIDs manually.
    GET /search/athlete?name=Luke+Heinen&weight=132
    """
    try:
        flo_res = await search_flo_athlete(name, weight)
        if isinstance(flo_res, Exception):
            flo_res = [{"source": "flo", "error": str(flo_res)}]
        return JSONResponse({"flo": flo_res, "usab": []})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/search/usab")
async def search_usab(name: str, state: str = None):
    """
    Search USA Bracketing for athletes by name (and optional state) while authenticated.
    Fetches /athletes to inspect the real search form, then submits it with
    first_name / last_name / state fields split from the name parameter.
    GET /search/usab?name=Luke+Heinen&state=MN
    Returns a JSON array of candidates with uuid, name, city, state, club.
    """
    try:
        # Split query name into first / last for the form fields
        parts      = name.strip().split(None, 1)
        first_name = parts[0] if parts else name
        last_name  = parts[1] if len(parts) > 1 else ""

        results = []
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            await _usab_login(client)

            # ── Step 1: fetch the athletes landing page to discover the form ──
            landing = await client.get(f"{USAB_API_BASE}/athletes", headers=USAB_HEADERS)
            print(f"=== /search/usab: landing status={landing.status_code} ===")

            landing_soup = BeautifulSoup(landing.text, "html.parser")
            search_form  = landing_soup.find("form")
            if search_form:
                form_action = search_form.get("action") or "/athletes"
                form_method = (search_form.get("method") or "get").lower()
                form_fields = {
                    inp.get("name"): inp.get("value", "")
                    for inp in search_form.find_all("input")
                    if inp.get("name")
                }
                print(f"=== /search/usab: form action={form_action!r} method={form_method!r} "
                      f"fields={list(form_fields.keys())} ===")
            else:
                form_action = "/athletes"
                form_method = "get"
                form_fields = {}
                print(f"=== /search/usab: no <form> found on landing page ===")

            # Build the full form URL
            if not form_action.startswith("http"):
                form_action = USAB_API_BASE.rstrip("/") + "/" + form_action.lstrip("/")

            # Merge hidden/default fields with the search values
            params = dict(form_fields)
            params["first_name"] = first_name
            params["last_name"]  = last_name
            if state:
                params["state"] = state

            print(f"=== /search/usab: submitting {form_method.upper()} {form_action} params={params} ===")

            # ── Step 2: submit the search form ────────────────────────────────
            if form_method == "post":
                resp = await client.post(form_action, data=params, headers=USAB_HEADERS)
            else:
                resp = await client.get(form_action, params=params, headers=USAB_HEADERS)

            print(f"=== /search/usab: search status={resp.status_code} final_url={resp.url} ===")
            print(f"=== /search/usab: response HTML (first 2000):\n{resp.text[:2000]} ===")

            # ── Step 3: parse athlete cards from the result page ───────────────
            soup    = BeautifulSoup(resp.text, "html.parser")
            uuid_re = re.compile(
                r'/athletes/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
                re.IGNORECASE,
            )
            # Name filter: accept if any query token appears in the candidate name
            name_tokens = [t.lower() for t in name.split() if len(t) > 1]
            seen = set()

            for a in soup.find_all("a", href=True):
                m = uuid_re.search(a["href"])
                if not m:
                    continue
                uid = m.group(1)
                if uid in seen:
                    continue
                seen.add(uid)

                card = a.find_parent(["div", "article", "li", "tr"])

                athlete_name = a.get_text(strip=True)
                if not athlete_name or athlete_name.lower() in ("view", "profile", "details", "more"):
                    if card:
                        h = card.find(["h1", "h2", "h3", "h4", "strong", "b"])
                        athlete_name = h.get_text(strip=True) if h else ""
                if not athlete_name:
                    continue

                # Filter: skip candidates whose name shares no token with the query
                name_lower = athlete_name.lower()
                if name_tokens and not any(tok in name_lower for tok in name_tokens):
                    continue

                # Extract city, state, club from card tokens
                city = cand_state = club = ""
                if card:
                    tokens = [el.get_text(strip=True) for el in card.find_all(["span", "div", "td", "p"])]
                    for tok in tokens:
                        if re.fullmatch(r'[A-Z]{2}', tok):
                            cand_state = tok
                        elif not city and re.match(r'^[A-Za-z][\w\s]*$', tok) and 2 < len(tok) < 40 \
                                and tok.lower() not in (athlete_name.lower(), "view", "profile"):
                            city = tok
                        elif not club and 2 < len(tok) < 60 \
                                and tok not in (athlete_name, city, cand_state):
                            club = tok

                results.append({
                    "uuid":  uid,
                    "name":  athlete_name,
                    "city":  city,
                    "state": cand_state,
                    "club":  club,
                })
                if len(results) >= 20:
                    break

        print(f"=== /search/usab: returned {len(results)} results for {name!r} ===")
        return JSONResponse(results)
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/scrape/combined")
async def scrape_combined(request: Request):
    """
    Multi-source season scrape: search FloWrestling and USA Bracketing by
    wrestler name, merge and deduplicate matches, then seed.

    Body:
    {
        "wrestlers": [
            {"name": "Luke Heinen",  "weight": "132"},
            {"name": "Rein Meitler", "weight": "132"}
        ],
        "weight":  "132",      // optional global override
        "season":  "2025-26"   // optional, defaults to current season
    }
    """
    try:
        body = await request.json()
        wrestler_entries = body.get("wrestlers", [])
        weight_override  = body.get("weight", None)
        season           = body.get("season", None)

        if not wrestler_entries:
            return JSONResponse({"error": "No wrestlers provided"}, status_code=400)

        profiles = []
        errors   = []

        for entry in wrestler_entries:
            name    = (entry.get("name") or "").strip()
            weight  = entry.get("weight") or weight_override
            flo_id  = entry.get("flo_id") or None
            usab_id = entry.get("usab_id") or None
            if not name:
                errors.append("Skipped entry with no name")
                continue

            print(f"=== scrape_combined: processing {name!r} flo_id={flo_id!r} usab_id={usab_id!r} ===")

            # Search Flo if no flo_id supplied
            if not flo_id:
                flo_hits = await search_flo_athlete(name, weight)
                if isinstance(flo_hits, Exception):
                    flo_hits = []
                norm_query = _norm_name(name)
                for hit in (flo_hits or []):
                    if hit.get("id") and _norm_name(hit.get("name", "")) and (
                        norm_query in _norm_name(hit.get("name", "")) or
                        _norm_name(hit.get("name", "")) in norm_query
                    ):
                        flo_id = hit["id"]
                        break
                if not flo_id and flo_hits and flo_hits[0].get("id"):
                    flo_id = flo_hits[0]["id"]
                print(f"=== scrape_combined: {name!r} flo search -> flo_id={flo_id!r} ===")

            # Fetch full season profiles in parallel
            flo_profile, usab_profile = await asyncio.gather(
                fetch_flo_profile(flo_id, season=season) if flo_id else asyncio.sleep(0),
                fetch_usab_profile(usab_id, season=season) if usab_id else asyncio.sleep(0),
                return_exceptions=True,
            )
            if isinstance(flo_profile, Exception) or not isinstance(flo_profile, dict):
                flo_profile = None
            if isinstance(usab_profile, Exception) or not isinstance(usab_profile, dict):
                usab_profile = None

            flo_matches  = (flo_profile  or {}).get("matches", [])
            usab_matches_raw = (usab_profile or {}).get("matches", [])

            # Season-filter USAB matches (fetch_usab_profile parses dates but
            # doesn't always enforce the season window uniformly)
            season_start, season_end = get_season_range(season)
            usab_matches  = []
            usab_filtered = 0
            usab_passed   = 0
            for m in usab_matches_raw:
                d = m.get("date")
                if d:
                    parsed = None
                    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
                        try:
                            parsed = datetime.strptime(d, fmt)
                            break
                        except ValueError:
                            pass
                    if parsed:
                        if season_start <= parsed <= season_end:
                            usab_passed += 1
                        else:
                            usab_filtered += 1
                            continue
                usab_matches.append(m)
            print(f"=== scrape_combined: {name!r} USAB season filter: "
                  f"pass={usab_passed} fail={usab_filtered} no_date={len(usab_matches)-usab_passed} ===")

            merged = merge_season_matches(flo_matches, usab_matches)
            print(f"=== scrape_combined: {name!r} flo={len(flo_matches)} usab={len(usab_matches)} merged={len(merged)} ===")
            # Dedup merged by (tournament, opponent) as a safety net
            seen_keys = set()
            deduped   = []
            for m in merged:
                key = (_norm_tournament(m.get("tournament", "")), _norm_name(m.get("opponent", "")))
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(m)
            if len(deduped) < len(merged):
                print(f"=== scrape_combined: {name!r} dedup removed {len(merged) - len(deduped)} duplicates ===")
            merged = deduped

            # Normalize every match: guarantee all required fields have usable defaults
            clean = []
            for m in merged:
                opp = (m.get("opponent") or "").strip()
                if not opp:
                    continue
                clean.append({
                    "opponent":   opp,
                    "method":     m.get("method") or "dec",
                    "won":        bool(m.get("won")),   # None -> False, never left as None
                    "tournament": (m.get("tournament") or "Unknown").strip(),
                    "date":       m.get("date") or None,
                    "score":      m.get("score") or None,
                    "source":     m.get("source") or "unknown",
                })
            if len(clean) != len(merged):
                print(f"=== scrape_combined: {name!r} dropped {len(merged) - len(clean)} matches with empty opponent ===")
            merged = clean

            wrestler_name = (
                (flo_profile  or {}).get("wrestler_name") or
                (usab_profile or {}).get("wrestler_name") or
                name
            )
            weight_class = (
                (flo_profile  or {}).get("weight_class") or
                (usab_profile or {}).get("weight_class") or
                weight or "Unknown"
            )

            if not merged:
                errors.append(f"No season matches found for {name}")

            flo_w  = sum(1 for m in merged if m.get("source") == "flo"  and m.get("won"))
            flo_l  = sum(1 for m in merged if m.get("source") == "flo"  and not m.get("won"))
            usab_w = sum(1 for m in merged if m.get("source") == "usab" and m.get("won"))
            usab_l = sum(1 for m in merged if m.get("source") == "usab" and not m.get("won"))

            profiles.append({
                "wrestler_id":   flo_id or usab_id or name,
                "wrestler_name": wrestler_name,
                "weight_class":  str(weight_class).replace(" lbs", "").strip(),
                "matches":       merged,
                "match_count":   len(merged),
                "flo_id":        flo_id,
                "usab_id":       usab_id,
                "flo_record":    f"{flo_w}-{flo_l}" if (flo_w or flo_l) else None,
                "usab_record":   f"{usab_w}-{usab_l}" if (usab_w or usab_l) else None,
                "error":         None,
            })

        if not profiles:
            return JSONResponse({"error": "No wrestler profiles could be built", "errors": errors}, status_code=400)

        try:
            import traceback as _tb
            field_names = {p.get("wrestler_name") or p.get("wrestler_id") for p in profiles}
            print(f"=== scrape_combined: field_names={field_names} ===")

            all_matches = build_seedit_matches(profiles, weight_override)
            print(f"=== scrape_combined: all_matches={len(all_matches)} ===")
            for i, m in enumerate(all_matches[:3]):
                print(f"=== scrape_combined: all_matches[{i}]={m} ===")

            h2h_matches = build_seedit_matches(profiles, weight_override, field_names=field_names)
            print(f"=== scrape_combined: h2h_matches={len(h2h_matches)} ===")

            if not all_matches:
                return JSONResponse({
                    "error":    "No match data found for the current season",
                    "profiles": profiles,
                    "errors":   errors,
                    "tip":      "Check wrestler names and season filter",
                }, status_code=200)

            print(f"=== scrape_combined: calling _run_seeding_engine ===")
            results_out = _run_seeding_engine(
                profiles, all_matches, h2h_matches, field_names,
                source="combined", errors=errors,
            )
            print(f"=== scrape_combined: seeding done, weight keys={list(results_out.keys())} ===")
            return JSONResponse(sanitize_for_json(results_out))

        except Exception as seed_err:
            import traceback as _tb
            tb_str = _tb.format_exc()
            print(f"=== scrape_combined SEEDING ERROR: {seed_err}\n{tb_str} ===")
            return JSONResponse({"error": str(seed_err), "trace": tb_str}, status_code=500)

    except Exception as e:
        import traceback as _tb
        tb_str = _tb.format_exc()
        print(f"=== scrape_combined OUTER ERROR: {e}\n{tb_str} ===")
        return JSONResponse({"error": str(e), "trace": tb_str}, status_code=500)
