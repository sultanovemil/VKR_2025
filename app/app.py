# Импорт библиотек
import cv2
from PIL import Image
from ultralytics import YOLO
import pytesseract
import streamlit as st
import pandas as pd
import numpy as np
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

# Настройка конфигурации страницы Streamlit
st.set_page_config(
    page_title="Document processing",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📄 Обработка удостоверений о повышении квалификации")
st.markdown("""
**Автоматическая система распознавания данных** из сканов/фото удостоверений.
Позволяет извлекать: ФИО, период обучения, организацию и другие данные.
""")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.subheader("📋 Инструкция")
    st.markdown("""
    1. **Загрузите** изображение документа
    2. Нажмите **«Распознать»**
    3. Проверьте и **отредактируйте** данные
    4. Добавьте в таблицу (**«Добавить в таблицу»**)
    5. Повторите для других документов
    6. **Скачайте** итоговый файл CSV
    """)
    st.markdown("---")

# Инициализация модели
@st.cache_resource
def load_model():
    model_path = "model/pretrained_model_yolo8n.pt"
    return YOLO(model_path)

model = load_model()

def draw_boxes(image, boxes):
    """Отрисовка bounding boxes на изображении"""
    image_np = np.array(image) if isinstance(image, Image.Image) else image.copy()
    
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    return Image.fromarray(image_np)

# Инициализация session state для хранения данных
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "ФИО", "Период обучения", "Организация", 
        "Название курса", "Учебные часы", "Город", 
        "Год", "Регистрационный номер"
    ])
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = {}
if 'show_download' not in st.session_state:
    st.session_state.show_download = False

# Загрузка изображения
uploaded_file = st.sidebar.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"],
                                         help="Загрузите скан/фото удостоверения")

if uploaded_file is not None:
    # Чтение и отображение изображения
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), 
             caption='Загруженное изображение', 
             use_container_width=True)
    
    if st.sidebar.button("Распознать"):
        with st.spinner("Идет обработка изображения..."):
            try:
                # Предобработка изображения
                preprocessed_img = preprocess_image(input_image)
                preprocessed_image = np.array(preprocessed_img)
                # Детектирование текстовых блоков
                detected_blocks, results = detect_text_blocks(preprocessed_img, model)
                boxes = results[0].boxes
                
                if len(boxes) == 0:
                    st.warning("Не обнаружено текстовых блоков")
                    st.stop()
                
                # Отображение изображения с bounding boxes
                image_with_boxes = draw_boxes(preprocessed_img, boxes)
                st.image(image_with_boxes, caption='Обнаруженные текстовые блоки', use_container_width=True)
                
                # Инициализация словаря для текущих данных
                current_data = {
                    "ФИО": "",
                    "Период обучения": "",
                    "Организация": "",
                    "Название курса": "",
                    "Учебные часы": "",
                    "Город": "",
                    "Год": "",
                    "Регистрационный номер": ""
                }
                
                # Создаем экземпляр OrganizationProcessor
                org_processor = OrganizationProcessor()
                
                # Обработка каждого блока
                for box in boxes:
                    # Получаем координаты и класс
                    bbox = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    
                    # Обрезаем изображение
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cropped_image = Image.fromarray(preprocessed_image[y_min:y_max, x_min:x_max])
                    
                    # Распознаем текст
                    original_text = pytesseract.image_to_string(cropped_image, lang='rus+eng').strip()
                    
                    # Обработка текста в зависимости от класса
                    if class_name == 'name':
                        current_data["ФИО"] = process_fio(original_text)
                    elif class_name == 'city':
                        current_data["Город"] = process_city_name(original_text)
                    elif class_name == 'hours':
                        current_data["Учебные часы"] = process_hours(original_text)
                    elif class_name == 'year':
                        current_data["Год"] = extract_and_correct_year(original_text)
                    elif class_name == 'course_period':
                        current_data["Период обучения"] = process_course_period(original_text)
                    elif class_name == 'course_topic':
                        current_data["Название курса"] = process_course_topic(original_text)
                    elif class_name == 'registration_number':
                        current_data["Регистрационный номер"] = correct_registration_number(original_text)
                    elif class_name == 'organization':
                        current_data["Организация"] = org_processor.process_organization(original_text)
                
                # Сохраняем текущие данные в session state
                st.session_state.current_data = current_data
                st.session_state.edited_data = current_data.copy()
                
                # Показываем кнопку "Добавить в таблицу" в sidebar
                st.session_state.show_add_button = True
            
            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")

            # Отображение формы для редактирования данных
            st.subheader("✏️ Проверка и корректировка данных")
            st.markdown("""
            <div style="background-color:#fff4e6; padding:10px; border-radius:5px; margin-bottom:15px">
            ⚠️ Пожалуйста, проверьте автоматически распознанные данные и при необходимости откорректируйте их
            </div>
            """, unsafe_allow_html=True)
if st.session_state.get('current_data'):
        
    # Создаем поля для редактирования
    for field, value in st.session_state.current_data.items():
        st.session_state.edited_data[field] = st.text_input(
            field, 
            value=st.session_state.edited_data.get(field, value),
            key=f"edit_{field}"
        )

# Кнопка добавления в таблицу
if st.session_state.get('show_add_button', False):
    if st.sidebar.button("Добавить в таблицу"):
        # Добавляем отредактированные данные в таблицу
        new_row = pd.DataFrame([st.session_state.edited_data])
        st.session_state.results_df = pd.concat([st.session_state.results_df, new_row], ignore_index=True)
        
        # Очищаем текущие данные
        st.session_state.current_data = {}
        st.session_state.edited_data = {}
        st.session_state.show_add_button = False
        
        # Показываем кнопку сохранения
        st.session_state.show_download = True
        
        st.success("Данные успешно добавлены в таблицу!")
        st.rerun()

# Отображение общей таблицы с данными
if not st.session_state.results_df.empty:
    st.header("Общая таблица с распознанными данными")
    st.write(st.session_state.results_df)

    # Кнопка сохранения в CSV (остается видимой после первого нажатия)
    if st.session_state.show_download:
        csv = st.session_state.results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.download_button(
            label="Скачать CSV",
            data=csv,
            file_name="recognized_data.csv",
            mime="text/csv; charset=utf-8-sig",
            key="download_csv"
        )
    
    # Кнопка "Начать заново"
    if st.sidebar.button("Начать заново"):
        st.session_state.results_df = pd.DataFrame(columns=[
            "ФИО", "Период обучения", "Организация", 
            "Название курса", "Учебные часы", "Город", 
            "Год", "Регистрационный номер"
        ])
        st.session_state.current_data = {}
        st.session_state.edited_data = {}
        st.session_state.show_add_button = False
        st.session_state.show_download = False
        st.rerun()

    else:
        st.info("Пожалуйста, загрузите изображение удостоверения")


st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 12px;
    color: #777;
    text-align: center;
}
</style>
<div class="footer">
Система разработана для автоматизации обработки документов. Версия 1.0 \n
2025 год
</div>
""", unsafe_allow_html=True)