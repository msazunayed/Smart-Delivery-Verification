# Smart Delivery Verification System

An end-to-end AI/ML pipeline for a courier/delivery company (Pathao-style use case): detects rider safety-gear compliance (helmet / no-helmet) and flags package condition issues from images, then automates alerts and logging with n8n.

```
[Rider App Camera] --image--> [n8n Webhook] --HTTP--> [FastAPI + YOLOv8] --JSON--> [n8n Logic: IF confidence/class]
                                                                                        |
                                                                    ------------------------------------------
                                                                    |                                        |
                                                              PASS -> log to DB/Sheet              FAIL -> Telegram/Slack alert + flag for review
```

## Project structure
```
smart-delivery-verification/
├── app/
│   ├── main.py        # FastAPI app: /detect, /logs, and serves web/index.html at /
│   ├── model.py        # YOLOv8 inference wrapper
│   └── schemas.py      # Pydantic request/response models
├── web/
│   └── index.html       # self-contained demo console (upload, verdict, history)
├── train.py             # fine-tuning script (Colab-friendly)
├── dataset.yaml          # YOLO dataset config template
├── n8n_workflow.json     # importable n8n workflow
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick start — run everything

```bash
# 1. Unzip and enter the folder
unzip smart-delivery-verification.zip
cd smart-delivery-verification

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Start the server (also serves the web UI)
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/** — that's the whole stack: API + UI in one process, one command.

From there:
- Drag and drop any photo, click **Run verification** → see bounding boxes + pass/fail verdict
- Check the **History** panel to see everything logged so far (`/logs` endpoint)
- Out of the box it uses the stock pretrained `yolov8n.pt`, so it'll detect generic objects (people, vehicles, etc.) but not yet "helmet" or "damaged package" — see **Section 1** below to train those classes in
- To wire up n8n automation on top, see **Section 4**
- To run it in Docker instead of locally, see **Section 3**

---

## 1. Data & training
1. Label ~150-300 images (helmet / no_helmet, package_ok / package_damaged) using Roboflow or LabelImg. Start from a public "helmet detection" dataset on Kaggle/Roboflow Universe and mix in your own photos.
2. Point `dataset.yaml` at your labeled data.
3. Run `train.py` (works on free Google Colab GPU):
   ```bash
   python train.py --data dataset.yaml --epochs 50 --imgsz 640
   ```
4. Best weights land in `runs/detect/train/weights/best.pt` — copy that into `app/weights/best.pt`.

## 2. Serving (FastAPI)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Test:
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@sample.jpg"
```
Response:
```json
{
  "detections": [
    {"class": "no_helmet", "confidence": 0.91, "box": [120, 40, 300, 220]}
  ],
  "compliant": false
}
```

## 3. Containerize
```bash
docker build -t delivery-verification .
docker run -p 8000:8000 delivery-verification
```

## 4. Automate with n8n
Import `n8n_workflow.json` into n8n (Workflows → Import from File). It:
1. Receives an image via Webhook (`rider_id`, `order_id`, image).
2. Calls the FastAPI `/detect` endpoint.
3. Checks confidence + class with an IF node.
4. On fail → sends a Telegram/Slack alert and flags for manual review.
5. On pass → logs the result to Google Sheets / Postgres.

Update the `HTTP Request` node URL to point at your deployed FastAPI service, and add your own Telegram/Slack/Sheets credentials.

## 5. Web UI (demo console)
`web/index.html` is a self-contained HTML/CSS/JS page — no build step, no separate server. FastAPI serves it directly, so one command runs the whole stack:
```bash
uvicorn app.main:app --reload --port 8000
```
Then open **http://localhost:8000/** in a browser. The console lets you:
- Drag-and-drop a photo and enter a rider/order ID
- Run verification and see the pass/fail verdict with bounding boxes drawn over the image
- Browse a **History** table (pulled live from the `/logs` endpoint) of every check that's been run, replacing the need for a separate dashboard app

If you open `web/index.html` directly as a local file instead of via the server, it falls back to calling `http://localhost:8000` — just make sure the API is running (CORS is already enabled for this case).

## Talking points for interviews
- Full pipeline: data labeling → transfer learning → model serving → workflow automation → monitoring.
- Precision/recall tradeoff: false negatives (missed no_helmet) are worse than false positives here — tune confidence threshold accordingly.
- Edge cases: low light, partial occlusion, motion blur — discuss augmentation strategy (`train.py` includes basic augments).
- Scalability: batch vs real-time inference, and how n8n lets non-engineers change business rules (thresholds, alert channels) without touching model code.
