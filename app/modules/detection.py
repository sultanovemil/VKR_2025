# Импорт библиотек
from ultralytics import YOLO
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict

def detect_text_blocks(
    image: Image.Image,
    model: YOLO,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4,
    img_size: int = 640
) -> Tuple[List[Dict], object]:
    """
    Детекция текстовых блоков.
    
    Параметры:
        image: PIL Image (рекомендуется после preprocess_image)
        model_path: путь к модели YOLO
        conf_threshold: 0.4-0.6 для текста
        iou_threshold: 0.3-0.5 для текстовых блоков
    
    Возвращает:
        (список блоков, raw_results)
    """
    # Конвертация в numpy array (без лишних преобразований)
    img_np = np.array(image)    
    
    # Детекция с оптимальными параметрами
    results = model.predict(
        source=img_np,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        augment=False,
        verbose=False
    )
    
    # Постобработка результатов
    detected_blocks = []
    for box in results[0].boxes:
        detected_blocks.append({
            'class': model.names[int(box.cls[0].item())],
            'confidence': float(box.conf[0].item()),
            'bbox': [round(x, 2) for x in box.xyxy[0].tolist()]
        })
    
    return detected_blocks, results