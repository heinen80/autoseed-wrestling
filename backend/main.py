@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    matches = df.to_dict(orient="records")

    seeds = compute_rankings(matches)
    bracket = build_bracket(seeds)
    schedule = schedule_matches(bracket)

    # Build wrestler history
    history = {}

    for m in matches:
        w1 = m["wrestlerA"]
        w2 = m["wrestlerB"]
        winner = m["winner"]

        # Initialize
        if w1 not in history:
            history[w1] = {"wins": [], "losses": []}
        if w2 not in history:
            history[w2] = {"wins": [], "losses": []}

        # Assign results
        if winner == w1:
            history[w1]["wins"].append(w2)
            history[w2]["losses"].append(w1)
        else:
            history[w2]["wins"].append(w1)
            history[w1]["losses"].append(w2)

    return JSONResponse({
        "seeds": [(i+1, w[0], round(w[1],3)) for i,w in enumerate(seeds)],
        "bracket": bracket,
        "schedule": schedule,
        "history": history
    })
