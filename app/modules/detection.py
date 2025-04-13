# Импорт библиотек
from ultralytics import YOLO
import cv2
import numpy as np


def detect_text_blocks(image, model_path, conf_threshold=0.8, iou_threshold=0.45):
    """
    Детекция текстовых блоков с оптимизацией через NMS и порог уверенности
    :param image: изображение в формате PIL Image
    :param model_path: путь к весам YOLOv8nano
    :param conf_threshold: порог уверенности (по умолчанию 0.8)
    :param iou_threshold: порог для NMS (по умолчанию 0.45)
    :return: список обнаруженных текстовых блоков
    """
    # Конвертация PIL Image в numpy array в формате BGR
    image_np = np.array(image)

    # Если изображение RGBA, конвертируем в RGB
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # Конвертация RGB в BGR (как ожидает OpenCV/YOLO)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Инициализация модели
    model = YOLO(model_path)

    # Детекция с параметрами NMS
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        augment=False
    )

    # Постобработка результатов
    detected_blocks = []
    boxes = results[0].boxes

    for box in boxes:
        bbox = box.xyxy[0].tolist()
        class_id = box.cls[0].item()
        class_name = model.names[int(class_id)]
        confidence = box.conf[0].item()
        detected_blocks.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': bbox
        })

    return detected_blocks, results