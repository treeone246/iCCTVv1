from ultralytics import YOLOE

# Start with the base YOLOE checkpoint
model = YOLOE("yoloe-11s-seg.pt")   # or 11m / 11l / 11x

# Define your PPE classes with descriptive prompts
classes = [
    "hard hat",
    "industrial coverall jumpsuit",
    "work glove safety glove",
    "safety glasses or goggles protective eyewear",
    "safety boot work boot",
    "hands not using gloves",
    "safety harness used on person",
    "socks not wearing safety boots",
]

# Generate text-prompt embeddings and set as the model's class vocabulary
model.set_classes(classes, model.get_text_pe(classes))

# Save — this writes a regular YOLO checkpoint with classes baked into the weights
model.save("yoloe-ppeRev2.pt")