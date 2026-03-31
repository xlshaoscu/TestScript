import base64
import requests

path = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"
with open(path, "rb") as f:
    img = base64.b64encode(f.read()).decode()

res = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "model": "/opt/data/models/IDEA-Research/Rex-Omni",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": "Detect person. Output the bounding box coordinates in [x0, y0, x1, y1] format."}
        ]
    }],
    "temperature": 0,
    "max_tokens": 40000
})

print(res.json())
