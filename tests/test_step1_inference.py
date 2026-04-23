import onnxruntime as ort
import numpy as np


def test_model_loads_and_infers():
    session = ort.InferenceSession("models/mobilenetv2.onnx")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})
    logits = outputs[0]  # shape: (1, 1000)
    assert logits.shape == (1, 1000), f"Unexpected output shape: {logits.shape}"
    top_class = int(np.argmax(logits))
    assert 0 <= top_class < 1000
    print(f"\nTop predicted class index: {top_class}")
