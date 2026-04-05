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

def build_history(matches):
    """
    Build per-wrestler win/loss history.
    Now also tracks win_methods: {opponent: method_points} for each win.

    ```
    CSV columns expected:
        weight, wrestlerA, wrestlerB, winner, method (optional)

    method examples: fall, pin, tf, tech fall, md, major decision, dec, decision
    """
    history = {}

    for m in matches:
        w1     = str(m["wrestlerA"]).strip()
        w2     = str(m["wrestlerB"]).strip()
        winner = str(m["winner"]).strip()
        method = str(m.get("method", "")).strip()

        history.setdefault(w1, {"wins": [], "losses": [], "win_methods": {}, "loss_methods": {}})
        history.setdefault(w2, {"wins": [], "losses": [], "win_methods": {}, "loss_methods": {}})

        pts = get_method_points(method)

        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w1]["win_methods"][w2] = pts       # how w1 beat w2
            history[w2]["losses"].append(w1)
            history[w2]["loss_methods"][w1] = pts      # how w2 lost to w1
        else:
            history[w2]["wins"].append(w1)
            history[w2]["win_methods"][w1] = pts
            history[w1]["losses"].append(w2)
            history[w1]["loss_methods"][w2] = pts

    return history

def avg_win_quality(wrestler: str, history: dict) -> float:
    """Average method points across all wins. 0 if no wins."""
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

def build_power_scores(history, sos, top_wins, bad_losses):
    """
    Unified power score — identical weighting to compare_breakdown()
    so seeds and comparisons never contradict each other.

    ```
    Components (max ~100):
        win %            × 30   → 0–30
        win quality avg  × 4    → 0–24   (fall=6, tf=5, md=4, dec=3)
        opponent quality × 8    per win  (opponent win %)
        opponent quality × 6    penalty per loss (based on how weak the opp is)
        SOS              × 15   → 0–15
        top wins         × 2    each
        bad losses       × 2    penalty each
    """
    scores = {}

    for w in history:
        wins   = len(history[w]["wins"])
        losses = len(history[w]["losses"])
        total  = wins + losses
        win_pct = wins / total if total else 0

        score = 0.0

        # 1. Win percentage baseline
        score += win_pct * 30

        # 2. Average win quality (method-based)
        score += avg_win_quality(w, history) * 4

        # 3. Opponent quality on wins (beat good people = more points)
        for opp in history[w]["wins"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                # scale by both opponent quality AND how you beat them
                method_pts = history[w]["win_methods"].get(opp, 3.0)
                score += opp_pct * 8 * (method_pts / 6.0)

        # 4. Penalty for losses (losing to weak opponents hurts more)
        for opp in history[w]["losses"]:
            opp_total = len(history[opp]["wins"]) + len(history[opp]["losses"])
            if opp_total:
                opp_pct = len(history[opp]["wins"]) / opp_total
                # losing to a .300 wrestler = big penalty; losing to a .700 = small
                score -= (1 - opp_pct) * 8

        # 5. Strength of schedule
        score += sos[w] * 15

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
            power_scores = build_power_scores(history, sos, top_wins, bad_losses)
            seeds        = compute_rankings(power_scores)
            common       = build_common(history)
            win_methods  = build_win_method_summary(history)

            # confidence and alerts now use power_scores for tiebreaking
            confidence   = build_confidence(seeds, history, sos, top_wins, bad_losses, power_scores)
            controversy_flags, upset_alerts, debate_queue = build_alerts(
                seeds, history, sos, top_wins, bad_losses, confidence, power_scores
            )

            results[str(weight)] = {
                "seeds":              [(i + 1, w[0]) for i, w in enumerate(seeds)],
                "history":            history,
                "sos":                sos,
                "common":             common,
                "confidence":         confidence,
                "top_wins":           top_wins,
                "bad_losses":         bad_losses,
                "power_scores":       {k: round(v, 1) for k, v in power_scores.items()},
                "win_methods":        win_methods,   # NEW — per-wrestler method breakdown
                "controversy_flags":  controversy_flags,
                "upset_alerts":       upset_alerts,
                "debate_queue":       debate_queue,
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
                "weight": str(weight).replace(" lbs", "").strip(),
                "wrestlerA": wrestler_name,
                "wrestlerB": opp,
                "winner": winner,
                "method": method,
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
        return JSONResponse(results_out)

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
                "wins": [], "losses": [], "win_methods": {}, "loss_methods": {}
            })

        # Power scores and seeds: only field wrestlers, SOS/quality from full data.
        power_scores = build_power_scores(
            h2h_history, full_sos, full_top_wins, full_bad_losses
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

        def fs(d):
            return {k: v for k, v in d.items() if k in field_names}

        results_out[str(weight)] = {
            "seeds":             [(i + 1, w[0]) for i, w in enumerate(seeds)],
            "history":           fs(full_history),
            "sos":               fs(full_sos),
            "common":            common_out,
            "confidence":        confidence,
            "top_wins":          fs(full_top_wins),
            "bad_losses":        fs(full_bad_losses),
            "power_scores":      {k: round(v, 1) for k, v in power_scores.items()},
            "win_methods":       fs(win_methods),
            "controversy_flags": controversy_flags,
            "upset_alerts":      upset_alerts,
            "debate_queue":      debate_queue,
            "source":            source,
            "profiles_scraped":  len(profiles),
            "matches_processed": len(h2h_matches),
            "errors":            errors,
        }

    return results_out

# —————————

# Multi-source helpers

# —————————

USAB_API_BASE = "https://www.usabracketing.com"
USAB_HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}


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
    POST /api/search with {"query": name, "entityType": "person"}.
    Response: data[].items[] with ofpId, title ("Lastname, Firstname"), metadata2 (location).
    Returns up to 5 matches as {source, id, name, location, weight} dicts.
    """
    headers = dict(FLO_API_HEADERS)
    headers["content-type"] = "application/json"
    payload = {"query": name, "entityType": "person"}
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(FLO_SEARCH_URL, json=payload, headers=headers)
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
    Search USA Bracketing for a wrestler by name and weight class.
    Logs in first using USB_USERNAME / USB_PASSWORD environment variables.
    Pass an existing httpx.AsyncClient as `client` for connection reuse.
    Returns up to 5 matches as {source, id, name, weight} dicts.
    """
    username = os.environ.get("USB_USERNAME", "")
    password = os.environ.get("USB_PASSWORD", "")
    own_client = client is None
    try:
        if own_client:
            client = httpx.AsyncClient(timeout=15, follow_redirects=True)

        # Authenticate when credentials are configured
        if username and password:
            await client.post(
                f"{USAB_API_BASE}/api/users/login",
                json={"username": username, "password": password},
                headers=USAB_HEADERS,
            )
            # Proceed regardless of login status — cookies captured automatically

        params = {"name": name}
        if weight:
            params["weightClass"] = str(weight)
        resp = await client.get(
            f"{USAB_API_BASE}/athletes/search",
            params=params,
            headers=USAB_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data") or data.get("athletes") or []
        results = []
        for a in raw:
            results.append({
                "source": "usab",
                "id":     a.get("id"),
                "name": (
                    a.get("name") or
                    (a.get("firstName", "") + " " + a.get("lastName", "")).strip()
                ),
                "weight": str(a.get("weightClass") or weight or ""),
            })
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
    Fetch match results from USA Bracketing for a wrestler by ID.
    Returns the same shape as fetch_flo_profile.
    """
    season_start, season_end = get_season_range(season)
    matches = []
    wrestler_name = None
    weight_class = None

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(
                f"{USAB_API_BASE}/athletes/{wrestler_id}/results",
                headers=USAB_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

        profile_root = data.get("data") or data
        if isinstance(profile_root, dict):
            wrestler_name = (
                profile_root.get("name") or
                " ".join(filter(None, [
                    profile_root.get("firstName"),
                    profile_root.get("lastName"),
                ])).strip() or
                str(wrestler_id)
            )

        for event in (data.get("results") or data.get("events") or []):
            event_name = event.get("name") or event.get("eventName") or "USA Bracketing"
            for bout in (event.get("bouts") or event.get("boutResults") or []):
                opp = (
                    (bout.get("opponent") or {}).get("name") or
                    bout.get("opponentName") or ""
                )
                if not opp:
                    continue

                won      = bool((bout.get("athlete") or {}).get("isWinner") or bout.get("isWinner"))
                win_type = str(bout.get("winType") or bout.get("decisionType") or "").upper()
                method   = WIN_TYPE_MAP.get(win_type, "dec")

                wc = str(bout.get("weight") or event.get("weightClass") or "")
                if wc and not weight_class:
                    weight_class = parse_weight_from_text(wc)

                date_str = bout.get("date") or event.get("startDate") or ""
                match_date = None
                if date_str:
                    try:
                        match_date = datetime.fromisoformat(
                            date_str.rstrip("Z").split("T")[0]
                        )
                    except Exception:
                        pass

                in_season = True
                if match_date:
                    in_season = season_start <= match_date <= season_end

                if in_season:
                    matches.append({
                        "opponent":   opp,
                        "method":     method,
                        "won":        won,
                        "tournament": event_name,
                        "date":       match_date.strftime("%b %d, %Y") if match_date else None,
                        "score":      None,
                    })

    except Exception as e:
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
    Search both FloWrestling and USA Bracketing for a wrestler by name.
    Returns candidate profiles from both platforms for the TD to confirm.
    GET /search/athlete?name=Luke+Heinen&weight=132
    """
    try:
        flo_res, usab_res = await asyncio.gather(
            search_flo_athlete(name, weight),
            search_usab_athlete(name, weight),
            return_exceptions=True,
        )
        if isinstance(flo_res, Exception):
            flo_res = [{"source": "flo", "error": str(flo_res)}]
        if isinstance(usab_res, Exception):
            usab_res = [{"source": "usab", "error": str(usab_res)}]
        return JSONResponse({"flo": flo_res, "usab": usab_res})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
            name   = (entry.get("name") or "").strip()
            weight = entry.get("weight") or weight_override
            if not name:
                errors.append("Skipped entry with no name")
                continue

            # Search both platforms in parallel
            flo_hits, usab_hits = await asyncio.gather(
                search_flo_athlete(name, weight),
                search_usab_athlete(name, weight),
                return_exceptions=True,
            )
            if isinstance(flo_hits, Exception):
                flo_hits = []
            if isinstance(usab_hits, Exception):
                usab_hits = []

            # Pick best match: first hit whose name loosely matches the query
            norm_query = _norm_name(name)

            flo_id = None
            for hit in (flo_hits or []):
                if hit.get("id") and _norm_name(hit.get("name", "")) and (
                    norm_query in _norm_name(hit.get("name", "")) or
                    _norm_name(hit.get("name", "")) in norm_query
                ):
                    flo_id = hit["id"]
                    break
            if not flo_id and flo_hits and flo_hits[0].get("id"):
                flo_id = flo_hits[0]["id"]

            usab_id = None
            for hit in (usab_hits or []):
                if hit.get("id") and _norm_name(hit.get("name", "")) and (
                    norm_query in _norm_name(hit.get("name", "")) or
                    _norm_name(hit.get("name", "")) in norm_query
                ):
                    usab_id = hit["id"]
                    break
            if not usab_id and usab_hits and usab_hits[0].get("id"):
                usab_id = usab_hits[0]["id"]

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
            usab_matches = (usab_profile or {}).get("matches", [])
            merged       = merge_season_matches(flo_matches, usab_matches)

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

            profiles.append({
                "wrestler_id":   flo_id or usab_id or name,
                "wrestler_name": wrestler_name,
                "weight_class":  str(weight_class).replace(" lbs", "").strip(),
                "matches":       merged,
                "match_count":   len(merged),
                "flo_id":        flo_id,
                "usab_id":       usab_id,
                "error":         None,
            })

        if not profiles:
            return JSONResponse({"error": "No wrestler profiles could be built", "errors": errors}, status_code=400)

        field_names  = {p.get("wrestler_name") or p.get("wrestler_id") for p in profiles}
        all_matches  = build_seedit_matches(profiles, weight_override)
        h2h_matches  = build_seedit_matches(profiles, weight_override, field_names=field_names)

        if not all_matches:
            return JSONResponse({
                "error":    "No match data found for the current season",
                "profiles": profiles,
                "errors":   errors,
                "tip":      "Check wrestler names and season filter",
            }, status_code=200)

        results_out = _run_seeding_engine(
            profiles, all_matches, h2h_matches, field_names,
            source="combined", errors=errors,
        )
        return JSONResponse(results_out)

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)
