import sqlite3
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

DB = "orchestrator/registry.db"


def init_db():
    con = sqlite3.connect(DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS nodes "
        "(node_id TEXT PRIMARY KEY, address TEXT, model_version TEXT)"
    )
    con.commit()
    con.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/register")
def register(node_id: str = Form(...), address: str = Form(...)):
    con = sqlite3.connect(DB)
    con.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,'unknown')", (node_id, address))
    con.commit()
    con.close()
    return {"registered": node_id}


@app.get("/nodes")
def list_nodes():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT * FROM nodes").fetchall()
    con.close()
    return [{"node_id": r[0], "address": r[1], "model_version": r[2]} for r in rows]


@app.post("/deploy")
async def deploy(file: UploadFile = File(...), version: str = Form(...)):
    model_bytes = await file.read()
    con = sqlite3.connect(DB)
    nodes = con.execute("SELECT node_id, address FROM nodes").fetchall()
    con.close()
    results = {}
    async with httpx.AsyncClient() as client:
        for node_id, address in nodes:
            r = await client.post(
                f"{address}/update-model",
                files={"file": (f"{version}.onnx", model_bytes, "application/octet-stream")},
                data={"version": version},
                timeout=30,
            )
            results[node_id] = r.json()
    return {"deployed_version": version, "results": results}
