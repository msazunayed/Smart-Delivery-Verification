import csv
import os
import shutil
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import detector
from app.schemas import Detection, DetectionResponse

app = FastAPI(title="Smart Delivery Verification API")

# Allow the web UI (opened as a local file or served from another port)
# to call this API during local development/demos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
WEB_DIR = os.path.join(BASE_DIR, "web")
LOG_PATH = os.path.join(BASE_DIR, "detections_log.csv")

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def serve_ui():
    """Serve the demo console at http://localhost:8000/ so the whole
    stack runs from a single `uvicorn app.main:app` command."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Smart Delivery Verification API. POST an image to /detect."}


def _log_result(rider_id, order_id, compliant, detections):
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "rider_id", "order_id", "compliant", "classes"]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                rider_id,
                order_id,
                compliant,
                ";".join(d["cls"] for d in detections),
            ]
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    rider_id: str | None = Form(default=None),
    order_id: str | None = Form(default=None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detections, compliant = detector.predict(tmp_path)
    finally:
        os.remove(tmp_path)

    _log_result(rider_id, order_id, compliant, detections)

    return DetectionResponse(
        detections=[Detection(**d) for d in detections],
        compliant=compliant,
        rider_id=rider_id,
        order_id=order_id,
    )


@app.get("/logs")
def get_logs(limit: int = 50):
    """Return the most recent logged checks, newest first — used by the
    web UI's History panel instead of a separate dashboard app."""
    if not os.path.exists(LOG_PATH):
        return {"rows": []}

    with open(LOG_PATH, newline="") as f:
        reader = list(csv.DictReader(f))

    rows = reader[::-1][:limit]
    return {"rows": rows}
