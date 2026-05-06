BDD100K_COLOR_TO_TRAINID = {
    (220, 20, 60): 11,
    (255, 0, 0): 12,
    (0, 0, 142): 13,
    (119, 11, 32): 18,
    (0, 0, 230): 17,
    (0, 60, 100): 14,
    (0, 0, 70): 15,
    (250, 170, 30): 6,
}

BDD100K_TRAINID_TO_CLASS = {
    11: "person", 12: "person",
    13: "car", 18: "bicycle", 17: "motorcycle",
    14: "bus", 15: "truck", 6: "traffic light",
}

TARGET_CLASSES = [
    "person", "car", "bicycle", "motorcycle", "bus", "truck", "traffic light"
]

CLASS_SPECIFIC_MIN_AREA = {
    "person": 20,
    "bicycle": 50,
    "motorcycle": 50,
    "car": 100,
    "bus": 200,
    "truck": 150,
    "traffic light": 3,
}

YOLO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}
