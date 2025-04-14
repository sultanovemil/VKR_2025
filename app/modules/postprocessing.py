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
    
    # 2. Удаляем служебные слова
    words = [word for word in unified.split() if word not in stop_words]
    filtered_text = ' '.join(words)
    
    # 3. Оставляем только кириллицу, пробелы и дефисы
    cleaned = re.sub(r'[^а-яё -]', '', filtered_text)
    
    # 4. Удаляем двойные пробелы и обрезаем краевые
    cleaned = ' '.join(cleaned.split())
    
    # 5. Капитализируем каждое слово (с учётом двойных фамилий)
    capitalized = ' '.join(
        part.capitalize() 
        for word in cleaned.split(' ')
        for part in word.split('-')
    )
    
    # 6. Удаляем одиночные буквы
    result = ' '.join(word for word in capitalized.split() if len(word) > 1)
    
    return result if len(result) >= 3 else None


def process_city_name(raw_city):
    """Очистка и нормализация названия города"""
    if not raw_city:
        return None

    # 1. Удаление всех небуквенных символов, кроме дефисов и пробелов
    cleaned = re.sub(r"[^а-яёА-ЯЁ -]", "", raw_city.strip())

    # 2. Удаление префиксов типа "г."
    cleaned = re.sub(r"^г\.\s*", "", cleaned, flags=re.IGNORECASE)

    # 3. Удаление двойных пробелов и обрезка краевых
    cleaned = " ".join(cleaned.split())

    # 4. Приведение к нормальному регистру (каждое слово и часть после дефиса с заглавной)
    if cleaned:
        # Разбиваем на слова и части через дефис
        parts = []
        for word in cleaned.split():
            if '-' in word:
                word_parts = [part.capitalize() for part in word.split('-')]
                parts.append('-'.join(word_parts))
            else:
                parts.append(word.capitalize())
        cleaned = ' '.join(parts)

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
    Возвращает год в формате "yyyy г." или исходный текст, если год не распознан.
    """
    # Нормализация текста: удаление лишних символов и приведение к нижнему регистру
    cleaned_text = re.sub(r'[«»"\'.]', '', text).strip().lower()

    # Замена распространенных OCR-ошибок в цифрах
    cleaned_text = re.sub(r'[зЗ]', '3', cleaned_text)

    # Паттерны для извлечения года в порядке приоритета
    patterns = [
        r'(?<!\d)(\d{4})(?!\d)\s*г?\b',
        r'\b(\d{4})\s*г?\b',
        r'\d{1,2}\s*[а-я]+\s*(\d{4})\s*г?\b',
        r'\d{1,2}\.\d{1,2}\.(\d{4})\b',
        r'\d{1,2}\s+\d{1,2}\s+(\d{4})\b',
    ]

    year = None
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            year_candidate = match.group(1)
            # Валидация года (допустимый диапазон 1900-2099)
            if 1900 <= int(year_candidate) <= 2099:
                year = year_candidate
                break

    # Возвращаем отформатированный результат
    return f"{year} г." if year else text


def process_course_period(raw_period):
    """
    Очищает и нормализует период обучения, исправляя типичные ошибки OCR.
    Возвращает отформатированную строку вида "с DD.MM.YYYY по DD.MM.YYYY".
    """
    # 1. Удаление лишних символов (кроме цифр, точек, дефисов и ключевых слов)
    cleaned = re.sub(r'[^0-9а-яёА-ЯЁa-zA-Z\s«»\".-]', '', raw_period)

    # 2. Замена частых OCR-ошибок
    replacements = {
        r'сЗ': 'с 3', r'maa': 'мая', r'ene': 'в период с',
        r'\"': '', r'«\s*': '"', r'\s*»': '"',
        r'\bг\.|\bг\b': '',
    }
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # 3. Извлечение дат по паттернам
    date_patterns = [
        r'(\d{1,2})[\.\s]+(\d{1,2})[\.\s]+(\d{4})',
        r'(\d{1,2})\s+([а-я]+)\s+(\d{4})',
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, cleaned)
        for match in matches:
            day, month, year = match.groups()
            # Конвертация месяца из текста в число
            if month.isalpha():
                month = month_to_num(month)
            dates.append(f"{day}.{month}.{year}")

    # 4. Форматирование результата
    if len(dates) >= 2:
        return f"с {dates[0]} по {dates[1]}"
    return raw_period  # Если не удалось извлечь даты


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
