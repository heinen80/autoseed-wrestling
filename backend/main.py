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
from bs4 import BeautifulSoup
from datetime import datetime
import re
import asyncio

FLO_BASE = "https://flowrestling.org"

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

async def fetch_flo_profile(wrestler_id, season=None, client=None):
    url = f"{FLO_BASE}/nextgen/people/{wrestler_id}?tab=results"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    }

    season_start, season_end = get_season_range(season)
    matches = []
    wrestler_name = None
    weight_class = None

    try:
        if client is None:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
                resp = await c.get(url, headers=headers)
                html = resp.text
        else:
            resp = await client.get(url, headers=headers)
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        # Try to get wrestler name from page title or heading
        name_el = soup.find("h1") or soup.find("h2") or soup.find(class_=re.compile(r'name|title|athlete', re.I))
        if name_el:
            wrestler_name = name_el.get_text(strip=True)

        # Try to get weight class from profile header
        for el in soup.find_all(string=re.compile(r'Weight|Class|lbs', re.I)):
            w = parse_weight_from_text(el)
            if w:
                weight_class = w
                break

        # FloWrestling nextgen pages are React-rendered
        # The results data may be in a script tag as JSON
        for script in soup.find_all("script"):
            src = script.string or ""
            if "results" in src.lower() and ("dec" in src.lower() or "fall" in src.lower() or "opponent" in src.lower()):
                # Try to find JSON data blobs
                json_matches = re.findall(r'\{[^{}]*"result"[^{}]*\}', src, re.IGNORECASE)
                for jm in json_matches:
                    try:
                        import json
                        obj = json.loads(jm)
                        # Extract if it looks like a match record
                        opp = obj.get("opponent") or obj.get("opponentName") or obj.get("name")
                        result = obj.get("result") or obj.get("method") or obj.get("decision")
                        won = obj.get("won") or obj.get("win") or obj.get("winner")
                        if opp and result:
                            matches.append({
                                "opponent": str(opp).strip(),
                                "method": map_flo_method(str(result)),
                                "won": bool(won),
                                "tournament": "FloWrestling",
                                "date": None,
                            })
                    except:
                        pass

        # Parse HTML table structure (what we can see in the screenshots)
        # FloWrestling results page groups matches by tournament
        current_tournament = None
        current_date = None

        # Find all tournament sections — they appear as headers with tournament name + date
        # Then match rows follow underneath
        all_elements = soup.find_all(["div", "tr", "li", "span", "p"])

        for el in all_elements:
            text = el.get_text(separator=" ", strip=True)

            # Detect tournament header rows (have a date pattern)
            date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+20\d{2}', text)
            if date_match and len(text) < 120:
                current_date = date_match.group(0)
                tournament_name = text.replace(current_date, "").strip()
                if tournament_name:
                    current_tournament = tournament_name

            # Detect match rows — look for win method keywords
            method_match = re.search(r'\b(DEC|F|TF|MD|FALL|TECH|MAJOR|DEFAULT|FORFEIT|DQ)\b', text, re.I)
            if method_match and current_tournament:
                # Parse the match row
                # Rows appear to have: Name1 / Team1 / Name2 / Team2 / Method / Score
                # From screenshots: two names, two teams, result code, score
                lines = [l.strip() for l in el.get_text(separator="\n").split("\n") if l.strip()]
                names = []
                method = method_match.group(1)
                score = None

                for line in lines:
                    # Score pattern: digits-digits
                    if re.match(r'^\d+[\s\-|]\d+$', line):
                        score = line
                        continue
                    # Skip team abbreviations (short all-caps)
                    if re.match(r'^[A-Z]{2,6}$', line) and len(line) <= 6:
                        continue
                    # Skip pure method labels
                    if line.upper() in ("DEC","F","TF","MD","FALL","TECH","MAJOR","ROUND","RESULTS","NAME","TEAM"):
                        continue
                    # Skip round numbers
                    if re.match(r'^\d+$', line) and int(line) < 20:
                        continue
                    # Likely a wrestler name
                    if len(line) > 3 and re.search(r'[A-Za-z]', line):
                        names.append(line)

                if len(names) >= 2:
                    # First name = wrestler on top row (usually subject wrestler)
                    # Second name = opponent
                    # Determine who won from score context
                    # From screenshots: top score 0 means Luke lost, bottom score is opponent's
                    # Simpler: if wrestler_name in names[0], opponent is names[1]
                    subj = names[0]
                    opp  = names[1]

                    # Try to determine win/loss
                    # Look for score context — from screenshots winner score appears higher
                    won = None
                    if score:
                        parts = re.split(r'[\s\-|]', score)
                        if len(parts) == 2:
                            try:
                                s1, s2 = int(parts[0]), int(parts[1])
                                won = s1 > s2
                            except:
                                pass

                    # Parse date for season filtering
                    match_date = None
                    if current_date:
                        try:
                            match_date = datetime.strptime(
                                re.sub(r',', '', current_date).strip(),
                                "%b %d %Y"
                            )
                        except:
                            try:
                                match_date = datetime.strptime(current_date.strip(), "%b %d, %Y")
                            except:
                                pass

                    # Season filter
                    in_season = True
                    if match_date:
                        in_season = season_start <= match_date <= season_end

                    if in_season:
                        matches.append({
                            "opponent": opp,
                            "method": map_flo_method(method),
                            "won": won,
                            "tournament": current_tournament,
                            "date": current_date,
                            "score": score,
                        })

    except Exception as e:
        return {
            "wrestler_id": wrestler_id,
            "wrestler_name": wrestler_name,
            "weight_class": weight_class,
            "matches": [],
            "error": str(e),
            "match_count": 0,
        }

    return {
        "wrestler_id": wrestler_id,
        "wrestler_name": wrestler_name,
        "weight_class": weight_class,
        "matches": matches,
        "match_count": len(matches),
        "error": None,
    }

def build_seedit_matches(profiles, weight_override=None):
    seedit_matches = []
    name_map = {}

    # Build name->profile map
    for p in profiles:
        name = p.get("wrestler_name") or p.get("wrestler_id")
        if name:
            name_map[p["wrestler_id"]] = name

    for p in profiles:
        wrestler_name = p.get("wrestler_name") or p.get("wrestler_id")
        weight = weight_override or p.get("weight_class") or "Unknown"

        for m in p.get("matches", []):
            opp = m.get("opponent")
            method = m.get("method", "dec")
            won = m.get("won")
            if not opp or won is None:
                continue
            winner = wrestler_name if won else opp
            seedit_matches.append({
                "weight": str(weight).replace(" lbs","").strip(),
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
            {"id": "0QBPq4lRlQuIWK71", "name": "Luke Heinen"},
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

        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            tasks = []
            for entry in wrestler_entries:
                raw = entry.get("url") or entry.get("id") or ""
                flo_id = extract_flo_id(raw)
                if not flo_id:
                    errors.append(f"Could not parse ID from: {raw}")
                    continue
                tasks.append(fetch_flo_profile(flo_id, season=season, client=client))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    errors.append(f"Error fetching profile {i}: {str(r)}")
                else:
                    # Override name if provided
                    entry = wrestler_entries[i] if i < len(wrestler_entries) else {}
                    if entry.get("name") and not r.get("wrestler_name"):
                        r["wrestler_name"] = entry["name"]
                    profiles.append(r)

        # Build SeedIT match format
        all_matches = build_seedit_matches(profiles, weight_override)

        if not all_matches:
            return JSONResponse({
                "error": "No match data found for the current season",
                "profiles": profiles,
                "errors": errors,
                "tip": "Check that the FloWrestling IDs are correct and wrestlers have current season results"
            }, status_code=200)

        # Run through seeding engine
        df = pd.DataFrame(all_matches)
        results_out = {}

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
            results_out[str(weight)] = {
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
                "source": "flo",
                "profiles_scraped": len(profiles),
                "matches_processed": len(all_matches),
                "errors": errors,
            }

        return JSONResponse(results_out)

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/scrape/flo/preview/{wrestler_id}")
async def preview_flo_wrestler(wrestler_id: str, season: str = None):
    """
    Preview a single wrestler's scraped data before running full seeding.
    Useful for confirming the right wrestler was found.
    GET /scrape/flo/preview/0QBPq4lRlQuIWK71?season=2025-26
    """
    try:
        profile = await fetch_flo_profile(wrestler_id, season=season)
        return JSONResponse(profile)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
