# Импорт библиотек
import re
import json
from difflib import get_close_matches


def process_fio(raw_fio):
    """Очистка и нормализация ФИО"""
    if not raw_fio:
        return None
    
    # Слова для удаления
    stop_words = {'фамилия', 'имя', 'отчество'}
    
    # 1. Заменяем переносы строк на пробелы и приводим к нижнему регистру
    unified = raw_fio.replace('\n', ' ').lower()
    
    # 2. Оставляем только кириллицу, пробелы и дефисы
    cleaned = re.sub(r'[^а-яё -]', '', unified)

    # 3. Удаляем служебные слова
    words = [word for word in cleaned.split() if word not in stop_words]
    filtered_text = ' '.join(words)    
    
    # 4. Удаляем двойные пробелы и обрезаем краевые
    cleaned = ' '.join(filtered_text.split())
    
    # 5. Капитализируем каждое слово (с учётом двойных фамилий)
    capitalized = ' '.join(
        part.capitalize() 
        for word in cleaned.split(' ')
        for part in word.split('-')
    )
    
    # 6. Удаляем одиночные буквы
    result = ' '.join(word for word in capitalized.split() if len(word) > 1)
    
    return result if len(result) >= 3 else None


def load_city_dictionary(file_path):
    """Загрузка  словаря городов из JSON-файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_city_name(raw_city, similarity_threshold=0.7):
    """
    Очистка, нормализация и проверка названия города
    
    :param raw_city: Исходное название города    
    :param check_dict: Флаг принудительной проверки по словарю
    :param similarity_threshold: Порог схожести для исправления опечаток
    :return: Нормализованное название города или None
    """
    # Загрузка словаря
    city_list = load_city_dictionary('modules\cities.json')

    if not raw_city:
        return None

    # 1. Удаление всех небуквенных символов, кроме дефисов и пробелов
    cleaned = re.sub(r"[^а-яёА-ЯЁ -]", "", raw_city.strip())

    # 2. Удаление префиксов типа "г."
    cleaned = re.sub(r"^г\.\s*", "", cleaned, flags=re.IGNORECASE)

    # 3. Удаление двойных пробелов и обрезка краевых
    cleaned = " ".join(cleaned.split())

    # 4. Приведение к нормальному регистру
    if cleaned:
        parts = []
        for word in cleaned.split():
            if '-' in word:
                word_parts = [part.capitalize() for part in word.split('-')]
                parts.append('-'.join(word_parts))
            else:
                parts.append(word.capitalize())
        cleaned = ' '.join(parts)

    # 5. Проверка по списку городов
    if city_list  and cleaned:
        # Приводим к нижнему регистру для сравнения
        lower_cleaned = cleaned.lower()
        city_names_lower = [city.lower() for city in city_list]
        
        # Проверяем точное совпадение
        if lower_cleaned not in city_names_lower:
            # Ищем похожие названия
            matches = get_close_matches(lower_cleaned, city_names_lower, n=1, cutoff=similarity_threshold)
            if matches:
                # Возвращаем правильное написание из списка
                correct_name = next(city for city in city_list if city.lower() == matches[0])
                return correct_name
                
    return cleaned if cleaned else None


def process_hours(raw_hours):
    """Очистка и нормализация данных об учебных часах"""

    # 1. Удаление всех нецифровых символов, кроме пробелов
    cleaned = re.sub(r"[^0-9 ]", "", raw_hours.strip())

    # 2. Извлечение числового значения часов
    match = re.search(r"\d+", cleaned)
    if not match:
        return None

    hours = match.group()

    # 3. Форматирование в нужный вид
    return f"{hours} ч."


def extract_and_correct_year(text):
    """
    Извлекает и корректирует год из текстовой строки.
    Возвращает год в формате "yyyy г." или "xxxx г.", если год не распознан.
    """
    # Нормализация текста: удаление лишних символов и приведение к нижнему регистру
    cleaned_text = re.sub(r'[«»"\'.]', '', text).strip().lower()
    
    # Замена распространенных OCR-ошибок в цифрах и буквах
    replacements = {
        r'[зЗ]': '3',
        r'о': '0',
        r'[дД]': '2',
        r'[чЧ]': '4',
        r'[бБ]': '6',
        r'[жЖ]': '7',
        r'[лЛ]': '1',
    }
    for pattern, repl in replacements.items():
        cleaned_text = re.sub(pattern, repl, cleaned_text)

    # Паттерны для извлечения года в порядке приоритета
    patterns = [
        r'(?<!\d)(\d{4})(?!\d)\s*г?\b',  # 2021 г.
        r'\b(\d{4})\s*г?\b',             # 2021г
        r'(\d{4})\.$',                   # 2021.
        r'(\d{4})\s*\d*\.?$',            # 2021 2.
        r'\d{1,2}\.\d{1,2}\.(\d{4})',    # 24.07.2020
        r'\d{1,2}\s*[а-я]+\s*(\d{4})',   # 24 июля 2020
    ]

    year = None
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            year_candidate = match.group(1)            
            year_candidate = year_candidate[-4:]
            # Валидация года (допустимый диапазон 1900-2099)
            if year_candidate.isdigit() and 1900 <= int(year_candidate) <= 2099:
                year = year_candidate
                break

    # Возвращаем отформатированный результат
    return f"{year} г." if year else "xxxx г."


def process_course_period(raw_period):
    """
    Очищает и нормализует период обучения, исправляя типичные ошибки OCR.
    Возвращает отформатированную строку вида "с DD.MM.YYYY по DD.MM.YYYY".
    """
    # 1. Предварительная очистка и замена частых ошибок
    replacements = {
        r'сЗ': 'с 3', r'с‹': 'с ', r'с«': 'с ', r'с°': 'с ',
        r'\bno\b': 'по', r'\brona\b': 'года', r'\bг\.|\bг\b': '',
        r'\bmas\b|\bmaa\b|\bmast\b': 'мая', r'\bene\b': '',
        r'„|‚|»|«|"|\.\.+': ' ', r'\s+': ' ', r'\.\s*года': '',
        r'с(\d)': r'с \1', r'по(\d)': r'по \1'  # разделяем слитные написания
    }
    
    cleaned = raw_period
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # 2. Извлечение дат по расширенным паттернам
    date_patterns = [
        # Стандартные форматы
        r'(?P<day>\d{1,2})[\.\s]*(?P<month>\d{1,2}|[а-яё]+)[\.\s]*(?P<year>\d{4})',
        # Форматы с опечатками и без года
        r'(?P<day>\d{1,2})\s*(?P<month>[а-яё]+)(?P<year>\s*\d{4})?',
        # Форматы с разделителями
        r'(?P<day>\d{1,2})[\.\,\-\s]+(?P<month>\d{1,2}|[а-яё]+)[\s\-\.\,]*(?P<year>\d{4})?'
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, cleaned, flags=re.IGNORECASE)
        for match in matches:
            day = match.group('day').zfill(2)
            month = match.group('month')
            year = match.group('year') or ''  # Если год не указан
            
            # Нормализация месяца
            if month.isalpha():
                month = month_to_num(month.strip())
            elif month.isdigit():
                month = month.zfill(2)
            
            # Если год не указан, используем из предыдущей даты
            if not year.strip() and dates:
                year = dates[-1].split('.')[-1] if '.' in dates[-1] else ''
            
            if year.strip():
                dates.append(f"{day}.{month}.{year.strip()}")
            else:
                dates.append(f"{day}.{month}")

    # 3. Форматирование результата
    if len(dates) >= 2:
        # Добавляем год ко второй дате, если его нет
        if '.' not in dates[1] and '.' in dates[0]:
            year = dates[0].split('.')[-1]
            dates[1] = f"{dates[1]}.{year}"
        
        # Проверяем, что обе даты имеют год
        if all('.' in date for date in dates[:2]):
            return f"с {dates[0]} по {dates[1]}"
    
    return raw_period


def month_to_num(month):
    """Преобразует название месяца в число (январь → 01)"""
    months = {
        'января': '01', 'февраля': '02', 'марта': '03',
        'апреля': '04', 'мая': '05', 'июня': '06',
        'июля': '07', 'августа': '08', 'сентября': '09',
        'октября': '10', 'ноября': '11', 'декабря': '12'
    }
    return months.get(month.lower(), month)


def process_course_topic(raw_topic):
    """
    Очищает и нормализует тему курса, исправляя типичные ошибки OCR.
    Возвращает исправленную строку с темой курса.
    """
    # 1. Замена частых OCR-ошибок
    replacements = {
        r'[«»"”“]': '"',  # Стандартизация кавычек
        r'\.\s*\"': '"',   # Удаление точек перед кавычками
        r'\s+': ' ',       # Удаление лишних пробелов
        r'Руйоп': 'Python',
        r'ЕГерБапТ': 'ELephanT',
        r'ВОХИ': 'BOXit',
        r'ЗТЕМ': 'STEM'
    }

    cleaned = raw_topic
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned)

    # 2. Удаление лишних символов
    cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9"\s\-:.,]', '', cleaned)

    # 3. Коррекция пунктуации
    cleaned = re.sub(r'\s([,.:])', r'\1', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    return cleaned.strip()


def correct_registration_number(text):
    """Функция для постобработки рег. номера"""
    # Удаление лишних символов (точки, пробелы)
    text = re.sub(r'[.;\s]', '', text)

    # Удаление лишних пробелов вокруг дефисов и слэшей
    text = re.sub(r'\s*([-/])\s*', r'\1', text)

    return text


class OrganizationProcessor:
    def __init__(self, json_file='modules/organisations.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.organizations = self.data['organisations']
        self.abbreviations = self.data['abbreviations']

        # Создаем словарь для быстрого поиска полных названий
        self.org_dict = {self._normalize_name(org): org for org in self.organizations}

        # Словарь для обратного поиска сокращений
        self.reverse_abbr = {v: k for k, v in self.abbreviations.items()}

    def _normalize_name(self, text):
        """Нормализация названия для сравнения"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
        text = re.sub(r'\s+', ' ', text).strip()  # Удаляем лишние пробелы
        return text

    def _find_similar_organization(self, ocr_text):
        """Поиск наиболее похожей организации в эталонном списке"""
        normalized_ocr = self._normalize_name(ocr_text)

        # Прямое совпадение
        if normalized_ocr in self.org_dict:
            return self.org_dict[normalized_ocr]

        # Поиск похожих названий
        matches = get_close_matches(
            normalized_ocr,
            self.org_dict.keys(),
            n=1,
            cutoff=0.7
        )

        return self.org_dict[matches[0]] if matches else None

    def _shorten_organization_name(self, full_name):
        """Сокращение названия организации"""
        # Сначала проверяем стандартные сокращения
        for full, short in self.reverse_abbr.items():
            if full_name.startswith(full):
                return full_name.replace(full, short, 1)

        # Если нет стандартного сокращения, применяем общие правила
        if full_name.startswith("Федеральное государственное бюджетное образовательное учреждение"):
            return full_name.replace("Федеральное государственное бюджетное образовательное учреждение", "ФГБОУ", 1)
        elif full_name.startswith("Автономная некоммерческая организация"):
            return full_name.replace("Автономная некоммерческая организация", "АНО", 1)
        elif full_name.startswith("Общество с ограниченной ответственностью"):
            return full_name.replace("Общество с ограниченной ответственностью", "ООО", 1)

        return full_name

    def process_organization(self, ocr_text):
        """Основной метод обработки названия организации"""
        # 1. Находим наиболее похожее название из эталонного списка
        matched_org = self._find_similar_organization(ocr_text)

        if not matched_org:
            return ocr_text

        # 2. Сокращаем название по правилам
        shortened_name = self._shorten_organization_name(matched_org)

        return shortened_name
