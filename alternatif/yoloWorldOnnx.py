# Step 1: bake classes (one-time, dev machine)
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-world.pt")
model.set_classes([
    "hard hat", "industrial coverall", "work glove",
    "safety glasses", "safety boot", "harness"
])
model.save("yolov8s-world-ppe.pt")   # bakes class embeddings into the weights