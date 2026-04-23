# Edge-AI

A minimal but complete edge AI infrastructure system that simulates a fleet of edge devices running local inference, coordinated by a central orchestrator, with peer-to-peer CRDT state sync between nodes — no cloud dependency required.

```
┌─────────────────────────────────────────────┐
│              Orchestrator :8000             │
│  Device registry · Deploy API · Versioning  │
└──────────┬──────────────┬───────────────────┘
           │              │             │
    Node A :8001    Node B :8002   Node C :8003
    ONNX Runtime    ONNX Runtime   ONNX Runtime
    Local infer     Local infer    Local infer
           │              │             │
           └──────────────┴─────────────┘
               Peer-to-peer CRDT sync
```

## What it demonstrates

- **On-device inference** — each node runs MobileNetV2 locally via ONNX Runtime; no external API calls during inference
- **Hot model deployment** — the orchestrator pushes a new ONNX model to all registered nodes without restarting containers
- **Device registry** — nodes auto-register with the orchestrator on startup; the registry tracks node address and current model version
- **LWW-Map CRDT sync** — after each inference, nodes gossip results to peers using a Last-Write-Wins Map; all nodes converge to identical state without a central coordinator
- **Containerized edge simulation** — each node runs as an isolated Docker container; the same image would run on an edge device fleet (i.e. Jetson Nano, Raspberry Pi etc...) with only an execution provider swap

## Tech stack

- Python 3.11, FastAPI, Uvicorn
- ONNX Runtime (CPU)
- MobileNetV2 (ONNX Model Zoo, ~14 MB)
- SQLite (device registry)
- httpx (async HTTP, node-to-node gossip)
- Docker + Docker Compose

## Project structure

```
Edge-AI/
├── docker-compose.yml
├── requirements.txt
├── demo.py                      # end-to-end walkthrough script
├── models/
│   ├── mobilenetv2.onnx         # download separately (see below, excluded from git)
│   └── imagenet_classes.txt     # included in repo
├── node/
│   ├── Dockerfile
│   └── main.py                  # edge node: /infer, /status, /update-model, /sync, /store
├── orchestrator/
│   ├── Dockerfile
│   └── main.py                  # control plane: /register, /nodes, /deploy
└── tests/
    ├── images/                  # test images (car, dog, cat, teddy bear, etc.)
    ├── test_step1_inference.py  # ONNX Runtime loads and infers
    ├── test_step2_node.py       # /status and /infer on node-a
    ├── test_step3_orchestrator.py  # auto-registration and /deploy
    └── test_step4_crdt.py       # LWW-Map convergence across all 3 nodes
```

## Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Python 3.11+ with `httpx` and `pytest` installed locally (for running tests and `demo.py` from the host)

## Setup

### 1. Download the model file

`models/imagenet_classes.txt` is already included in the repository. You only need to download the ONNX model binary (excluded from git due to its 14 MB size):

```bash
curl -L -o models/mobilenetv2.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
```

### 2. Install host-side Python dependencies

```bash
pip install httpx pytest
```

### 3. Build Docker images

```bash
docker compose build
```

## Running the full stack

```bash
docker compose up -d
```

This starts four containers on an isolated bridge network:

| Container | Host port | Role |
|-----------|-----------|------|
| orchestrator | 8000 | Control plane — device registry, model deployment |
| node-a | 8001 | Edge node — local inference, CRDT sync |
| node-b | 8002 | Edge node — local inference, CRDT sync |
| node-c | 8003 | Edge node — local inference, CRDT sync |

Nodes auto-register with the orchestrator on startup. Follow logs with:

```bash
docker compose logs -f
```

## End-to-end demo

With the stack running, execute the demo script from the project root:

```bash
python demo.py
```

The script walks through:
1. Confirm all 3 nodes registered with the orchestrator
2. Deploy model `v1` to all nodes via `/deploy`
3. Run inference on each node with a different image (car, teddy bear, dog)
4. Verify CRDT store convergence — all nodes hold identical results
5. Check model version and uptime on each node
6. Deploy model `v2` to all nodes
7. Run inference with a second image set (cat, dog, cat)
8. Verify CRDT convergence again — stores now hold 6 entries
9. Final status check confirming all nodes report `v2` model

## Running tests

All tests hit the live containers, so the stack must be running first.

```bash
# Step 1 — ONNX Runtime loads the model (runs inside container)
docker compose run --rm node-a python -m pytest tests/test_step1_inference.py -v -s

# Step 2 — node /status and /infer endpoints
pytest tests/test_step2_node.py -v -s

# Step 3 — orchestrator auto-registration and model deploy
pytest tests/test_step3_orchestrator.py -v -s

# Step 4 — CRDT LWW-Map convergence across all 3 nodes
pytest tests/test_step4_crdt.py -v -s

# Run all host-side tests at once
pytest tests/test_step2_node.py tests/test_step3_orchestrator.py tests/test_step4_crdt.py -v -s
```

## Key API endpoints

### Orchestrator (`localhost:8000`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Register a node (`node_id`, `address` form fields) |
| GET | `/nodes` | List all registered nodes and their model versions |
| POST | `/deploy` | Push an ONNX model file to all nodes (`file`, `version` form fields) |

### Node (`localhost:800{1,2,3}`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Node ID, current model version, uptime |
| POST | `/infer` | Run inference on an uploaded image; gossips result to peers |
| POST | `/update-model` | Receive and hot-swap a new ONNX model (`file`, `version`) |
| POST | `/sync` | Receive a partial LWW-Map from a peer and merge it |
| GET | `/store` | Return the full local LWW-Map state |

## Stopping the stack

```bash
docker compose down
```

To also remove the named volume used for model storage:

```bash
docker compose down -v
```
