# Импорт библиотек
import json
import Levenshtein
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pytesseract
from ultralytics import YOLO
import numpy as np
import time
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

# Конфигурация
TEST_IMAGES_DIR = 'datasets/test/images'
ANNOTATIONS_FILE = 'datasets/test/test_annotations.json'
MODEL_PATH = "model/pretrained_model_yolo8n.pt"
OUTPUT_EXCEL = "recognition_results.xlsx"
TARGET_WIDTH = 1528
TARGET_HEIGHT = 1080

# Загрузка аннотаций
with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Инициализация модели и процессора организаций
model = YOLO(MODEL_PATH)
org_processor = OrganizationProcessor()

def evaluate_mAP():
    """Оценка mAP для модели детекции YOLOv8"""
    print("\nОценка mAP для модели YOLOv8...")
    results = model.val(data="datasets/data.yaml", split='test')
    return results.box.map50

def calculate_cer(predicted, ground_truth, class_name):
    """Вычисление CER (Character Error Rate) с предварительной постобработкой"""
    if predicted is None:
        predicted = ""
    if ground_truth is None:
        ground_truth = ""
    
    # Постобработка обоих текстов
    processed_pred = postprocess_text(class_name, predicted)
    processed_gt = postprocess_text(class_name, ground_truth)
    
    if not processed_gt:
        return 0.0 if not processed_pred else 1.0

    distance = Levenshtein.distance(str(processed_pred), str(processed_gt))
    return distance / max(len(processed_gt), 1)

def postprocess_text(class_name, text):
    """Применение соответствующей постобработки в зависимости от класса"""
    if text is None:
        return ""
    
    text = str(text).strip()
    if not text:
        return ""
    
    if class_name == 'name':
        return process_fio(text)
    elif class_name == 'city':
        return process_city_name(text)
    elif class_name == 'hours':
        return process_hours(text)
    elif class_name == 'year':
        return extract_and_correct_year(text)
    elif class_name == 'course_period':
        return process_course_period(text)
    elif class_name == 'course_topic':
        return process_course_topic(text)
    elif class_name == 'registration_number':
        return correct_registration_number(text)
    elif class_name == 'organization':
        return org_processor.process_organization(text)
    return text

def process_image(image_path, model):
    """Обработка изображения с упрощенным pipeline"""
    try:
        start_time = time.time()
        
        # Предобработка изображения
        img = preprocess_image(image_path)
        
        # Детектирование текстовых блоков
        _, results = detect_text_blocks(img, model)

        recognized_texts = {}
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            bbox = box.xyxy[0].tolist()
            
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            
            # Распознавание текста
            text = pytesseract.image_to_string(
                cropped_img, 
                lang='rus+eng',
                config='--psm 6 --oem 3'
            ).strip()
            recognized_texts[class_name] = text
        
        processing_time = time.time() - start_time
        return recognized_texts, processing_time
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return {}, 0.0

# Оценка mAP для модели детекции
mAP_score = evaluate_mAP()

# Основной цикл оценки CER и времени обработки
results = []
class_stats = {name: {'total_chars': 0, 'errors': 0} for name in model.names.values()}
total_processing_time = 0
processed_images = 0

for item in tqdm(annotations['images'], desc="Оценка качества распознавания"):
    image_path = os.path.join(TEST_IMAGES_DIR, item['file_name'])
    if not os.path.exists(image_path):
        continue
    
    gt_data = item['annotations']
    recognized_texts, processing_time = process_image(image_path, model)
    
    total_processing_time += processing_time
    processed_images += 1
    
    for class_name, gt_text in gt_data.items():
        if gt_text is None:
            gt_text = ""
        
        pred_text = recognized_texts.get(class_name, "")
        cer = calculate_cer(pred_text, gt_text, class_name)
        
        # Получаем постобработанные версии для отчета
        processed_gt = postprocess_text(class_name, gt_text)
        processed_pred = postprocess_text(class_name, pred_text)
        
        # Сохраняем детальные результаты
        results.append({
            'Изображение': item['file_name'],
            'Класс': class_name,
            'Исходный GT': gt_text,
            'Обработанный GT': processed_gt,
            'Распознанный текст': pred_text,
            'Обработанный текст': processed_pred,
            'CER': cer,
            'Длина текста': len(processed_gt),
            'Ошибок': int(cer * len(processed_gt))
        })
        
        # Обновляем статистику
        char_count = len(processed_gt)
        class_stats[class_name]['total_chars'] += char_count
        class_stats[class_name]['errors'] += int(cer * char_count)

# Создаем DataFrame с результатами
df_results = pd.DataFrame(results)

# Создаем сводную статистику
stats_data = []
for class_name, stats in class_stats.items():
    if stats['total_chars'] > 0:
        cer = stats['errors'] / stats['total_chars']
        stats_data.append({
            'Класс': class_name,
            'CER': cer,
            'Ошибки': stats['errors'],
            'Всего_символов': stats['total_chars'],
            'Точность': f"{1 - cer:.2%}"
        })

df_stats = pd.DataFrame(stats_data)

# Общая статистика
if not df_stats.empty and 'Всего_символов' in df_stats.columns:
    total_chars = df_stats['Всего_символов'].sum()
    total_errors = df_stats['Ошибки'].sum()
    overall_cer = total_errors / total_chars if total_chars > 0 else 0.0    
else:
    total_chars = 0
    total_errors = 0
    overall_cer = 0.0

# Вычисляем среднее время обработки
avg_processing_time = total_processing_time / processed_images if processed_images > 0 else 0

# Сохраняем в Excel
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Детальные результаты', index=False)
    df_stats.to_excel(writer, sheet_name='Статистика по классам', index=False)
    
    # Настройка ширины столбцов
    for sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = max_length + 2

# Вывод основных метрик
print("\n=== Основные метрики системы ===")
print(f"mAP@0.5 (YOLOv8): {mAP_score:.4f}")
print(f"Средний CER (Tesseract): {overall_cer:.4f}")
print(f"Общая точность распознавания: {1 - overall_cer:.2%}")
print(f"Среднее время обработки изображения: {avg_processing_time:.2f} сек.")
print(f"Общее время обработки {processed_images} изображений: {total_processing_time:.2f} сек.")
print(f"\nРезультаты сохранены в файл: {OUTPUT_EXCEL}")