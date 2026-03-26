import os
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize


# Initialize model
model = RexOmniWrapper(
    model_path="/opt/data/models/IDEA-Research/Rex-Omni",
    backend="transformers",  # or "vllm" "transformers",
    trust_remote_code=True
)

# Load image
image = Image.open("/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg")

# Object Detection
results = model.inference(
    images=image,
    task="detection",
    categories=["person", "car", "dog"]
)

result = results[0]

# 4) Visualize
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
vis.save("visualize.jpg")

