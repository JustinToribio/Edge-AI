import httpx

BASE = "http://localhost:8001"
IMAGE_PATH = "tests/images/car.jpg"


def test_status():
    r = httpx.get(f"{BASE}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "node-a"
    assert "model_version" in body
    assert "uptime_seconds" in body
    print(f"\nStatus: {body}")


def test_infer_with_real_image():
    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()
    r = httpx.post(f"{BASE}/infer", files={"file": ("image.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "node-a"
    assert len(body["predictions"]) == 5
    assert all("class" in p and "score" in p for p in body["predictions"])
    print(f"\nTop prediction: {body['predictions'][0]}")
    print(f"All predictions: {body['predictions']}")
