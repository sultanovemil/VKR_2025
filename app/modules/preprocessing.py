# Импорт библиотек
import cv2
from PIL import Image


def preprocess_image(image_path, target_width=1528, target_height=1080):
    """
    Предобработка изображения документа:
    1. Загрузка изображения
    2. Конвертация в RGB
    3. Ресайз до 1080x1528 с сохранением пропорций
    4. Возвращает изображение в формате PIL Image (JPEG, RGB)
    """
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

    # Конвертация BGR в RGB (OpenCV загружает в BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Приведение к целевому размеру с сохранением пропорций
    processed_img = resize_with_aspect_ratio(img_rgb, target_width, target_height)

    # Конвертация numpy array в PIL Image
    pil_image = Image.fromarray(processed_img)

    # Принудительное установление формата JPEG
    pil_image.format = 'JPEG'

    return pil_image

def resize_with_aspect_ratio(image, target_width=1528, target_height=1080):
    """Изменение размера с сохранением пропорций до 1080x1528"""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_ratio = target_width / target_height

    # Рассчитываем новые размеры с сохранением пропорций
    if aspect_ratio > target_ratio:
        # Широкое изображение - подгоняем по ширине
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Высокое изображение - подгоняем по высоте
        new_h = target_height
        new_w = int(target_height * aspect_ratio)

    # Ресайз изображения с высоким качеством
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Добавление паддинга до target_width x target_height
    delta_w = target_width - new_w
    delta_h = target_height - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Черный padding (0,0,0) для RGB
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)

    return padded
