import time

import httpx

NODES = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]
IMAGE_PATH = "tests/images/teddy.jpg"


def test_crdt_convergence():
    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    for node_url in NODES:
        r = httpx.post(
            f"{node_url}/infer",
            files={"file": ("teddy.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200, f"Infer failed on {node_url}: {r.text}"

    time.sleep(1)  # allow gossip to propagate

    stores = [httpx.get(f"{node}/store").json() for node in NODES]
    print(f"Stores: {stores}")
    assert all(len(s) == 3 for s in stores), (
        f"Stores not converged: sizes={[len(s) for s in stores]}"
    )
    assert stores[0] == stores[1] == stores[2], (
        f"Stores diverged — CRDT merge broken.\n"
        f"node-a keys: {set(stores[0])}\n"
        f"node-b keys: {set(stores[1])}\n"
        f"node-c keys: {set(stores[2])}"
    )
    print(f"\nAll 3 nodes converged to {len(stores[0])} shared entries")
    for key in stores[0]:
        print(f"  {key}: top={stores[0][key]['predictions'][0]['class']}")
