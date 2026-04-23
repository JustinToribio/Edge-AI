import asyncio
import hashlib
import io
import os
import pathlib
import shutil
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

NODE_ID          = os.getenv("NODE_ID", "node-a")
MODEL_PATH       = os.getenv("MODEL_PATH", "models/mobilenetv2.onnx")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
SELF_URL         = os.getenv("SELF_URL", "")
PEERS: list[str] = [p for p in os.getenv("PEERS", "").split(",") if p]

lww_store: Dict[str, Any] = {}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["session"] = ort.InferenceSession(MODEL_PATH)
    state["model_version"] = os.getenv("MODEL_VERSION", "v1")
    state["start_time"] = time.time()
    with open("models/imagenet_classes.txt") as f:
        state["classes"] = [line.strip() for line in f]
    if ORCHESTRATOR_URL and SELF_URL:
        async with httpx.AsyncClient() as client:
            for _ in range(5):
                try:
                    await client.post(
                        f"{ORCHESTRATOR_URL}/register",
                        data={"node_id": NODE_ID, "address": SELF_URL},
                        timeout=5,
                    )
                    break
                except Exception:
                    await asyncio.sleep(2)
    yield


app = FastAPI(lifespan=lifespan)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


@app.get("/status")
def status():
    return {
        "node_id": NODE_ID,
        "model_version": state["model_version"],
        "uptime_seconds": round(time.time() - state["start_time"], 1),
    }


@app.post("/update-model")
async def update_model(file: UploadFile = File(...), version: str = Form(...)):
    path = pathlib.Path(f"models/{NODE_ID}_{version}.onnx")
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    state["session"] = ort.InferenceSession(str(path))
    state["model_version"] = version
    return {"status": "updated", "version": version, "path": str(path)}


def merge(remote_store: dict) -> None:
    """LWW-Map merge: keep entry with highest timestamp."""
    for key, remote_val in remote_store.items():
        local_val = lww_store.get(key)
        if local_val is None or remote_val["timestamp"] > local_val["timestamp"]:
            lww_store[key] = remote_val


async def _gossip(key: str, entry: dict) -> None:
    async with httpx.AsyncClient() as client:
        for peer in PEERS:
            try:
                await client.post(f"{peer}/sync", json={key: entry}, timeout=5)
            except Exception:
                pass  # peer offline — will converge on next sync


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    img_hash = hashlib.sha256(data).hexdigest()[:12]
    tensor = preprocess(data)
    input_name = state["session"].get_inputs()[0].name
    logits = state["session"].run(None, {input_name: tensor})[0][0]
    top5_idx = np.argsort(logits)[-5:][::-1]
    predictions = [
        {"class": state["classes"][i], "score": round(float(logits[i]), 4)}
        for i in top5_idx
    ]
    entry = {"predictions": predictions, "timestamp": time.time(), "node_id": NODE_ID}
    key = f"{NODE_ID}:{img_hash}"
    lww_store[key] = entry
    asyncio.create_task(_gossip(key, entry))
    return {
        "node_id": NODE_ID,
        "model_version": state["model_version"],
        "predictions": predictions,
    }


@app.post("/sync")
def sync(remote_store: dict) -> dict:
    merge(remote_store)
    return {"status": "merged", "store_size": len(lww_store)}


@app.get("/store")
def get_store() -> dict:
    return lww_store
