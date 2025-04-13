# Импорт библиотек
import cv2
from PIL import Image
from ultralytics import YOLO
import pytesseract
from modules.preprocessing import preprocess_image
from modules.detection import detect_text_blocks
from modules.postprocessing import (
    process_fio,
    process_city_name,
    process_hours,
    extract_and_correct_year,
    process_course_period,
    process_course_topic,
    correct_registration_number,
    OrganizationProcessor
)


# Загрузка изображения
input_image = 'img/kpk2.png'

# Путь к предварительно обученной модели YOLOv8 nano
model_path = "model/pretrained_model_yolo8n.pt"

# Инициализация модели
model = YOLO(model_path)

# Предобработка изображения
preproceded_image = preprocess_image(input_image)

# Детектирование текстовых блоков
detected_blocks, results = detect_text_blocks(preproceded_image, model_path)

# Получаем boxes из результатов детекции
boxes = results[0].boxes

# Создаем экземпляр OrganizationProcessor для обработки названий организаций
org_processor = OrganizationProcessor()

# Проходим по каждому bounding box
for i, box in enumerate(boxes):
    # Получаем координаты bounding box
    bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

    # Получаем метку класса
    class_id = int(box.cls[0].item())  # ID класса
    class_name = model.names[class_id]  # Название класса по ID

    # Обрезаем изображение по bounding box (конвертируем координаты в целые числа)
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_image = preproceded_image.crop((x_min, y_min, x_max, y_max))

    # Распознаем текст с помощью Tesseract
    original_text = pytesseract.image_to_string(cropped_image, lang='rus+eng').strip()

    # Применяем соответствующую постобработку в зависимости от класса
    processed_text = original_text

    if class_name == 'name':
        processed_text = process_fio(original_text)
    elif class_name == 'city':
        processed_text = process_city_name(original_text)
    elif class_name == 'hours':
        processed_text = process_hours(original_text)
    elif class_name == 'year':
        processed_text = extract_and_correct_year(original_text)
    elif class_name == 'course_period':
        processed_text = process_course_period(original_text)
    elif class_name == 'course_topic':
        processed_text = process_course_topic(original_text)
    elif class_name == 'registration_number':
        processed_text = correct_registration_number(original_text)
    elif class_name == 'organization':
        processed_text = org_processor.process_organization(original_text)

    # Выводим результат в требуемом формате
    print(f"\nBox {i + 1}:")
    print(f"Class: {class_name}")
    print(f"Original text: {original_text}")
    print(f"Processed text: {processed_text}")
    print("-" * 50)

    
