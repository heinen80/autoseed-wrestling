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

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(None)):
    try:
        # Handle CSV upload
        if file:
            contents = await file.read()
            from io import StringIO
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
            matches = df.to_dict(orient="records")

        # Handle JSON paste
        else:
            data = await request.json()
            matches = data.get("matches", [])

        if not matches:
            return JSONResponse({"error": "No matches found"})

        # Simple test response (no advanced logic yet)
        return JSONResponse({
            "message": "Upload successful",
            "count": len(matches)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
