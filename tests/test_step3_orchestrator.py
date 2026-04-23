import time

import httpx

ORCH = "http://localhost:8000"
NODES = [
    ("node-a", "http://localhost:8001"),
    ("node-b", "http://localhost:8002"),
    ("node-c", "http://localhost:8003"),
]


def test_nodes_auto_registered():
    time.sleep(3)  # allow containers to finish startup registration
    r = httpx.get(f"{ORCH}/nodes")
    assert r.status_code == 200
    nodes = r.json()
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}: {nodes}"
    node_ids = {n["node_id"] for n in nodes}
    assert node_ids == {"node-a", "node-b", "node-c"}
    print(f"\nRegistered nodes: {node_ids}")


def test_deploy_model():
    with open("models/mobilenetv2.onnx", "rb") as f:
        r = httpx.post(
            f"{ORCH}/deploy",
            files={"file": ("mobilenetv2.onnx", f, "application/octet-stream")},
            data={"version": "v2"},
            timeout=60,
        )
    assert r.status_code == 200
    body = r.json()
    assert body["deployed_version"] == "v2"
    for node_id, _ in NODES:
        assert node_id in body["results"], f"{node_id} missing from deploy results"
        assert body["results"][node_id]["version"] == "v2", \
            f"{node_id} reported wrong version: {body['results'][node_id]}"
    print(f"\nDeploy results: {body['results']}")

    # Confirm each node reports the new version via /status
    for node_id, addr in NODES:
        r = httpx.get(f"{addr}/status")
        assert r.json()["model_version"] == "v2", \
            f"{node_id} /status still shows old version"
    print("All nodes confirmed v2 via /status")
