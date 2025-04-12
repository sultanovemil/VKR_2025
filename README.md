# DocumentProcessing 
**Система автоматического извлечения данных из удостоверений о повышении квалификации**

Репозиторий содержит код экспериментов для магистерской диссертации: 
*Разработка системы распознавания информации из удостоверений о повышении квалификации с использованием алгоритмов искусственного интеллекта* (2025)
 
## 📁 Структура репозитория
- `experiments/` – Jupyter Notebooks с обучением моделей и тестами OCR:
  - `yolov8n_training.ipynb` – обучение YOLOv8 для детекции текста.
  - `ssd300_training.ipynb` – обучение SSD300 на датасете удостоверений.
  - `tesseract_exp.ipynb` – сравнение качества Tesseract OCR.
  - `EasyOCR_exp.ipynb` – тесты EasyOCR.
- `data/` – примеры изображений.

## 🛠 Зависимости
- Python 3.8+
- Ultralytics (YOLOv8)
- OpenCV, Tesseract, EasyOCR
- Jupyter Notebook

Установка:
```bash
pip install -r requirements.txt

