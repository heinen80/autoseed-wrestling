@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    matches = df.to_dict(orient="records")

    seeds = compute_rankings(matches)
    bracket = build_bracket(seeds)
    schedule = schedule_matches(bracket)

    # Build match history
    history = {}
    for m in matches:
        for w in [m["wrestlerA"], m["wrestlerB"]]:
            if w not in history:
                history[w] = {"wins": [], "losses": []}

        if m["winner"] == m["wrestlerA"]:
            history[m["wrestlerA"]]["wins"].append(m["wrestlerB"])
            history[m["wrestlerB"]]["losses"].append(m["wrestlerA"])
        else:
            history[m["wrestlerB"]]["wins"].append(m["wrestlerA"])
            history[m["wrestlerA"]]["losses"].append(m["wrestlerB"])

    return JSONResponse({
        "seeds": [(i+1, w[0], round(w[1],3)) for i,w in enumerate(seeds)],
        "bracket": bracket,
        "schedule": schedule,
        "history": history
    })
