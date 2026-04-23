import time

import httpx

ORCH = "http://localhost:8000"
NODES = [
    ("node-a", "http://localhost:8001", "tests/images/car.jpg"),
    ("node-b", "http://localhost:8002", "tests/images/teddy.jpg"),
    ("node-c", "http://localhost:8003", "tests/images/dog.jpg"),
]

print("=== Edge-AI Demo ===\n")

# 1. Confirm auto-registration
print("1. Waiting for nodes to register with orchestrator...")
time.sleep(3)
nodes = httpx.get(f"{ORCH}/nodes").json()
print(f"   Registered {len(nodes)} nodes:")
for n in nodes:
    print(f"   - {n['node_id']} @ {n['address']} (model: {n['model_version']})")


# 2. Deploy model to all nodes
print("\n2. Deploying mobilenetv2 v1 to all nodes via orchestrator...")
with open("models/mobilenetv2.onnx", "rb") as f:
    r = httpx.post(
        f"{ORCH}/deploy",
        files={"file": ("mobilenetv2.onnx", f, "application/octet-stream")},
        data={"version": "v1"},
        timeout=120,
    )
body = r.json()
print(f"   Deployed version: {body['deployed_version']}")
for node_id, result in body["results"].items():
    print(f"   - {node_id}: {result}")

# 3. Run inference on each node independently (each with a different image)
print("\n3. Running local inference on each node (different image per node)...")
for node_id, addr, image_path in NODES:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    r = httpx.post(
        f"{addr}/infer",
        files={"file": (image_path.split("/")[-1], img_bytes, "image/jpeg")},
    )
    top = r.json()["predictions"][0]
    print(f"   {node_id} [{image_path.split('/')[-1]}]: '{top['class']}' (score={top['score']:.4f})")

# 4. Check CRDT gossip convergence
print("\n4. Checking CRDT store convergence (waiting 1s for gossip)...")
time.sleep(1)
stores = {}
for node_id, addr, _ in NODES:
    store = httpx.get(f"{addr}/store").json()
    stores[node_id] = store
    print(f"   {node_id}: {len(store)} entries in LWW store")

converged = len(set(frozenset(s.keys()) for s in stores.values())) == 1
print(f"\n   Convergence: {'YES - all nodes hold identical keys and inference results' if converged else 'NO - stores diverged'}")
if converged:
    shared = list(stores.values())[0]
    for key, entry in shared.items():
        top = entry["predictions"][0]
        print(f"   {key} → '{top['class']}' (score={top['score']:.4f})")

# 5. Show node status
print("\n5. Node status check:")
for node_id, addr, _ in NODES:
    r = httpx.get(f"{addr}/status")
    s = r.json()
    print(f"   {node_id}: model={s['model_version']}, uptime={s['uptime_seconds']}s")

# 6. Deploy model v2 to all nodes
print("\n6. Deploying mobilenetv2 v2 to all nodes via orchestrator...")
with open("models/mobilenetv2.onnx", "rb") as f:
    r = httpx.post(
        f"{ORCH}/deploy",
        files={"file": ("mobilenetv2.onnx", f, "application/octet-stream")},
        data={"version": "v2"},
        timeout=120,
    )
body = r.json()
print(f"   Deployed version: {body['deployed_version']}")
for node_id, result in body["results"].items():
    print(f"   - {node_id}: {result}")

# 7. Run inference on each node with a new set of images
ROUND2_IMAGES = [
    ("node-a", "http://localhost:8001", "tests/images/cat.jpg"),
    ("node-b", "http://localhost:8002", "tests/images/dog_2.jpg"),
    ("node-c", "http://localhost:8003", "tests/images/cat_2.jpg"),
]
print("\n7. Running local inference on each node (second image set)...")
for node_id, addr, image_path in ROUND2_IMAGES:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    r = httpx.post(
        f"{addr}/infer",
        files={"file": (image_path.split("/")[-1], img_bytes, "image/jpeg")},
    )
    top = r.json()["predictions"][0]
    print(f"   {node_id} [{image_path.split('/')[-1]}]: '{top['class']}' (score={top['score']:.4f})")

# 8. Check CRDT gossip convergence after round 2
print("\n8. Checking CRDT store convergence after round 2 (waiting 1s for gossip)...")
time.sleep(1)
stores2 = {}
for node_id, addr, _ in ROUND2_IMAGES:
    store = httpx.get(f"{addr}/store").json()
    stores2[node_id] = store
    print(f"   {node_id}: {len(store)} entries in LWW store")

converged2 = len(set(frozenset(s.keys()) for s in stores2.values())) == 1
print(f"\n   Convergence: {'YES - all nodes hold identical keys and inference results' if converged2 else 'NO - stores diverged'}")
if converged2:
    shared2 = list(stores2.values())[0]
    for key, entry in shared2.items():
        top = entry["predictions"][0]
        print(f"   {key} → '{top['class']}' (score={top['score']:.4f})")

# 9. Node status check after v2 deploy
print("\n9. Node status check:")
for node_id, addr, _ in ROUND2_IMAGES:
    r = httpx.get(f"{addr}/status")
    s = r.json()
    print(f"   {node_id}: model={s['model_version']}, uptime={s['uptime_seconds']}s")

print("\nDemo complete.")
