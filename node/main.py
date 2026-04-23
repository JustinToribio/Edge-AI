import asyncio
import io
import os
import pathlib
import shutil
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

NODE_ID          = os.getenv("NODE_ID", "node-a")
MODEL_PATH       = os.getenv("MODEL_PATH", "models/mobilenetv2.onnx")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
SELF_URL         = os.getenv("SELF_URL", "")

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


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    tensor = preprocess(data)
    input_name = state["session"].get_inputs()[0].name
    logits = state["session"].run(None, {input_name: tensor})[0][0]
    top5_idx = np.argsort(logits)[-5:][::-1]
    results = [
        {"class": state["classes"][i], "score": round(float(logits[i]), 4)}
        for i in top5_idx
    ]
    return {
        "node_id": NODE_ID,
        "model_version": state["model_version"],
        "predictions": results,
    }
