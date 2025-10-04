# ============================================================================
# 7-LAYER PROCESSING (PART 1: Layers 1-4)
# ============================================================================
import re
import pandas as pd
from typing import Dict, List, Any
SCHEMA_DICT = {}
def normalize_arabic(text: str) -> str:
    """Normalization of the Arabic text"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    text = text.replace("ى", "ي").replace("ـ", "")
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    return text

def layer1_text_normalization(question: str) -> Dict[str, Any]:
    """Layer 1: Text normalization"""
    normalized = normalize_arabic(question)
    tokens = [w.strip() for w in normalized.split() if w.strip()]
    numbers = [int(n) for n in re.findall(r'\d+', normalized)]

    return {
        'original': question,
        'normalized': normalized,
        'tokens': tokens,
        'numbers': numbers
    }

def layer2_intent_classification(processed_text: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 2: Intent Classification"""
    text = processed_text['normalized'].lower()

    patterns = {
        'COUNT_QUERY': (['كم', 'عدد', 'احص', 'احسب', 'كام'], 3.0),
        'LIST_QUERY': (['اسماء', 'اعرض', 'اظهر', 'من هم', 'اطبع', 'قائمه', 'اسم'], 2.5),
        'AGGREGATE_QUERY': (['متوسط', 'اعلى', 'اقل', 'مجموع', 'اكثر', 'اصغر', 'اكبر علامه', 'اعلى درجه'], 3.0),
        'AGE_QUERY': (['عمر', 'عمرهم', 'مواليد', 'اكبر من', 'اقل من', 'سنه', 'سنوات'], 3.5),
        'EXAM_QUERY': (['علامه', 'علامات', 'درجه', 'درجات', 'امتحان', 'اختبار', 'نجح', 'رسب'], 3.5),
        'INSTITUTE_QUERY': (['معهد', 'معاهد', 'مركز', 'مراكز'], 3.0),
        'LOCATION_QUERY': (['منطقة', 'اقليم', 'موقع', 'شمال', 'جنوب', 'وسط', 'مكان'], 3.0),
        'NAME_QUERY': (['اسم', 'اسماء', 'من اسمه', 'الاسم', 'اب', 'جد', 'عائله'], 2.5),
    }

    best_intent = 'GENERAL_QUERY'
    max_score = 0.0

    for intent, (keywords, weight) in patterns.items():
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            score = matches * weight
            if score > max_score:
                best_intent = intent
                max_score = score

    return {
        'intent': best_intent,
        'confidence': min(max_score / 5.0, 1.0)
    }

def layer3_entity_recognition(processed_text: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 3: Entity Extraction"""
    text = processed_text['normalized'].lower()
    entities = []

    # Region patterns
    region_patterns = {
        'الشمالي': 'الإقليم الشمالي', 'الشمال': 'الإقليم الشمالي', 'شمالي': 'الإقليم الشمالي',
        'الجنوبي': 'الإقليم الجنوبي', 'الجنوب': 'الإقليم الجنوبي', 'جنوبي': 'الإقليم الجنوبي',
        'الاوسط': 'الإقليم الأوسط', 'الوسط': 'الإقليم الأوسط', 'وسط': 'الإقليم الأوسط'
    }

    for pattern, value in region_patterns.items():
        if pattern in text:
            entities.append({'type': 'REGION', 'value': value, 'pattern': pattern})
            break

    # Gender patterns
    if 'ذكر' in text or 'ذكور' in text:
        entities.append({'type': 'GENDER', 'value': 'ذكر'})
    elif 'انثى' in text or 'اناث' in text or 'بنات' in text:
        entities.append({'type': 'GENDER', 'value': 'أنثى'})

    # Age patterns
    age_patterns = [
        (r'اكبر من (\d+)', 'GREATER_THAN'),
        (r'اقل من (\d+)', 'LESS_THAN'),
        (r'بين (\d+) و ?(\d+)', 'BETWEEN'),
        (r'فوق (\d+)', 'GREATER_THAN'),
        (r'تحت (\d+)', 'LESS_THAN'),
        (r'من (\d+) الى (\d+)', 'BETWEEN'),
        (r'عمره (\d+)', 'EQUAL'),
        (r'عمرهم (\d+)', 'EQUAL'),
    ]

    for pattern, op_type in age_patterns:
        match = re.search(pattern, text)
        if match:
            if op_type == 'BETWEEN':
                entities.append({
                    'type': 'AGE_RANGE',
                    'operator': op_type,
                    'value': [int(match.group(1)), int(match.group(2))]
                })
            else:
                entities.append({
                    'type': 'AGE_RANGE',
                    'operator': op_type,
                    'value': int(match.group(1))
                })
            break

    # Exam patterns
    exam_patterns = [
        (r'علامته? (اكبر من|اكثر من|فوق) (\d+)', 'GREATER_THAN'),
        (r'علامته? (اقل من|تحت) (\d+)', 'LESS_THAN'),
        (r'علامته? بين (\d+) و ?(\d+)', 'BETWEEN'),
        (r'نجح', 'PASS'),
        (r'رسب', 'FAIL'),
        (r'اعلى علامه', 'MAX_SCORE'),
        (r'اقل علامه', 'MIN_SCORE'),
        (r'متوسط العلامات', 'AVG_SCORE'),
    ]

    for pattern, op_type in exam_patterns:
        match = re.search(pattern, text)
        if match:
            if op_type == 'BETWEEN':
                entities.append({
                    'type': 'EXAM_SCORE',
                    'operator': 'BETWEEN',
                    'value': [int(match.group(2)), int(match.group(3))]
                })
            elif op_type in ['GREATER_THAN', 'LESS_THAN']:
                entities.append({
                    'type': 'EXAM_SCORE',
                    'operator': op_type,
                    'value': int(match.group(2))
                })
            elif op_type == 'PASS':
                entities.append({
                    'type': 'EXAM_SCORE',
                    'operator': 'GREATER_THAN_EQUAL',
                    'value': 50
                })
            elif op_type == 'FAIL':
                entities.append({
                    'type': 'EXAM_SCORE',
                    'operator': 'LESS_THAN',
                    'value': 50
                })
            elif op_type in ['MAX_SCORE', 'MIN_SCORE', 'AVG_SCORE']:
                entities.append({
                    'type': 'EXAM_AGGREGATE',
                    'function': op_type
                })
            break

    # Name patterns
    name_patterns = [
        (r'اسمه ([ا-ي]+)', 'FIRST_NAME'),
        (r'اسم الاب ([ا-ي]+)', 'FATHER_NAME'),
        (r'العائله ([ا-ي]+)', 'FAMILY_NAME'),
        (r'من اسمه ([ا-ي]+)', 'FIRST_NAME'),
    ]

    for pattern, name_type in name_patterns:
        match = re.search(pattern, text)
        if match:
            entities.append({
                'type': 'NAME_FILTER',
                'name_type': name_type,
                'value': match.group(1)
            })
            break

    # Areas
    if 'اربد' in text:
        entities.append({'type': 'AREA', 'value': 'اربد'})
    elif 'عمان' in text:
        entities.append({'type': 'AREA', 'value': 'عمان'})
    elif 'الزرقاء' in text:
        entities.append({'type': 'AREA', 'value': 'الزرقاء'})

    return {'entities': entities}

def layer4_context_analysis(text_data: Dict, intent: Dict, entities: Dict) -> Dict:
    """Layer 4: Context Analysis"""
    text = text_data['normalized'].lower()

    indicators = {
        'Students': ['طالب', 'طلاب', 'عمر', 'جنس', 'اسم', 'مهنه', 'مؤهل'],
        'Locations': ['منطقة', 'اقليم', 'موقع', 'شمال', 'جنوب', 'وسط'],
        'Institutes': ['معهد','مركز','مراكز', 'معاهد'],
        'Exams': ['علامة', 'درجة', 'امتحان', 'اختبار']
    }

    required_tables = set()
    for table, words in indicators.items():
        if any(w in text for w in words):
            required_tables.add(table)

    if intent['intent'] in ['COUNT_QUERY', 'LIST_QUERY', 'AGE_QUERY']:
        required_tables.add('Students')

    if any(e['type'] == 'REGION' for e in entities['entities']):
        required_tables.add('Students')
        required_tables.add('Locations')

    if not required_tables:
        required_tables.add('Students')

    return {'required_tables': list(required_tables)}

print("✓ Layers 1-4 defined")

# ============================================================================
# 7-LAYER PROCESSING (PART 2: Layers 5-7)
# ============================================================================

def layer5_schema_mapping(context: Dict) -> Dict:
    """Layer 5: Schema binding""
    return {
        'relationships': [
            "Students.residence_location_id = Locations.id",
            "Students.institute_id = Institutes.id",
            "Exams.student_id = Students.national_id"
        ]
    }

def get_simplified_schema(required_tables: List[str]) -> str:
    """Extract schema for required tables only"""
    simplified = ""

    for table in required_tables:
        if table in SCHEMA_DICT:
            simplified += f"\nCREATE TABLE {table} (\n"
            col_defs = []
            for col in SCHEMA_DICT[table]:
                col_def = f"    {col['name']} {col['type']}"
                if col['desc']:
                    col_def += f"  -- {col['desc']}"
                col_defs.append(col_def)
            simplified += ",\n".join(col_defs) + "\n);\n"

    return simplified

def extract_entities(text: str) -> Dict[str, Any]:
    """Wrapper للطبقة 3"""
    return layer3_entity_recognition(text)


def determine_intent(text: str, entities: Dict) -> str:
    """Wrapper للطبقة 2"""
    processed = layer1_text_normalization(text)
    return layer2_intent_classification(processed, entities)


def analyze_context(text: str, entities: Dict, intent: str) -> Dict[str, Any]:
    """Wrapper للطبقة 4"""
    processed = layer1_text_normalization(text)
    return layer4_context_analysis(processed, intent, entities)


