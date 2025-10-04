"""
SQL Processing - Layer 6 & 7
"""
import re
from typing import Dict, Any
from .models import get_sqlcoder_pipeline
from .utils import clean_sql

def build_prompt(question: str, entities: Dict, intent: str, context: Dict) -> str:
    """Layer 6: بناء Prompt لـ SQLCoder"""
    
    # Schema 
    schema = """CREATE TABLE Students (
  national_id TEXT PRIMARY KEY,
  first_name TEXT,
  father_name TEXT,
  grandfather_name TEXT,
  family_name TEXT,
  birth_date DATE,
  gender TEXT,
  phone TEXT,
  age INTEGER,
  profession TEXT,
  qualification TEXT,
  institute_id INTEGER,
  residence_location_id INTEGER
);

CREATE TABLE Locations (
  id INTEGER PRIMARY KEY,
  region TEXT,
  area TEXT
);

CREATE TABLE Institutes (
  id INTEGER PRIMARY KEY,
  institute_name TEXT,
  location_id INTEGER
);

CREATE TABLE Exams (
  id INTEGER PRIMARY KEY,
  student_id TEXT,
  exam_score INTEGER,
  exam_date DATE
);"""

    # أمثلة واقعية - مهمة لتعليم النموذج
    examples = """Question: كم عدد الطلاب؟
SQL: SELECT COUNT(*) FROM Students;

Question: اعرض أسماء الطلاب
SQL: SELECT first_name, father_name, family_name FROM Students LIMIT 50;

Question: كم طالب ذكر؟
SQL: SELECT COUNT(*) FROM Students WHERE gender = 'ذكر';

Question: كم طالب من الإقليم الشمالي؟
SQL: SELECT COUNT(*) FROM Students WHERE residence_location_id IN (SELECT id FROM Locations WHERE region LIKE '%الشمالي%');

Question: من الطلاب فوق 25 سنة؟
SQL: SELECT first_name, father_name, family_name, age FROM Students WHERE age > 25 LIMIT 50;

Question: ما متوسط علامات الطلاب؟
SQL: SELECT AVG(exam_score) FROM Exams;

Question: اعرض علامات الطلاب
SQL: SELECT S.first_name, S.father_name, E.exam_score FROM Students S JOIN Exams E ON S.national_id = E.student_id LIMIT 50;"""

    # بناء الـ Prompt النهائي
    prompt = f"""You are a SQL expert. Generate a SQLite query for the Arabic question.

Database Schema:
{schema}

Rules:
1. Use LIKE not ILIKE
2. Gender: 'ذكر' or 'أنثى'
3. Region: 'الإقليم الشمالي', 'الإقليم الجنوبي', 'الإقليم الأوسط'
4. For regions: residence_location_id IN (SELECT id FROM Locations WHERE region LIKE '%..%')
5. JOIN: Students.national_id = Exams.student_id
6. End with semicolon

Examples:
{examples}

Question: {question}

SQL Query:
SELECT"""

    return prompt


def extract_sql(response: str) -> str:
    """Layer 7: Extract SQL from the response"""
    if not response:
        return None

    # محاولة 1: من code blocks
    match = re.search(r'```(?:sql)?\s*(SELECT.*?);?\s*```', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # محاولة 2: من SELECT مباشرة
    match = re.search(r'(SELECT\s+.*?)(?:;|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # محاولة 3: أخذ كل شيء من SELECT
    if 'SELECT' in response.upper():
        idx = response.upper().find('SELECT')
        sql = response[idx:].strip()
        if ';' in sql:
            sql = sql.split(';')[0]
        return sql

    return None


def generate_sql(question: str, entities: Dict, intent: str, context: Dict) -> Dict[str, Any]:
    """Generate SQL using SQLCoder""
    
    print(f"\nProcessing: {question}")
    print(f"Intent: {intent}")
    print(f"Tables: {context['required_tables']}")
    
    # Layer 6: build  Prompt
    prompt = build_prompt(question, entities, intent, context)
    
    # import model 
    pipeline = get_sqlcoder_pipeline()
    if not pipeline:
        return {'sql': None, 'method': 'no_model'}
    
    try:
        response = pipeline(prompt, max_new_tokens=200)
        raw = response[0].get('generated_text', '') if isinstance(response, list) else str(response)
        
        # Layer 7:extract  SQL
        sql = extract_sql(raw)
        
        if sql:
            sql = clean_sql(sql)
            print(f"Generated SQL: {sql}")
            return {'sql': sql, 'method': 'sqlcoder'}
        else:
            print("Failed to extract SQL")
            return {'sql': None, 'method': 'extraction_failed'}
    
    except Exception as e:
        print(f"Error: {e}")
        return {'sql': None, 'method': 'error', 'error': str(e)}


def build_fallback_sql(question: str, entities: Dict, intent: str) -> str:
    """Building a simple SQL backup"""
    text = question.lower()
    conditions = []
    
    # add condition 
    if entities.get('gender'):
        conditions.append(f"gender = '{entities['gender']}'")
    
    if entities.get('profession'):
        conditions.append(f"profession LIKE '%{entities['profession']}%'")
    
    if entities.get('age_filter'):
        age = entities['age_filter']
        if age['op'] == 'gt':
            conditions.append(f"age > {age['value']}")
        elif age['op'] == 'lt':
            conditions.append(f"age < {age['value']}")
        elif age['op'] == 'between':
            conditions.append(f"age BETWEEN {age['values'][0]} AND {age['values'][1]}")
    
    where = " AND ".join(conditions) if conditions else "1=1"
    
    # add area
    if entities.get('region'):
        where += f" AND residence_location_id IN (SELECT id FROM Locations WHERE region LIKE '%{entities['region']}%')"
    
    # Construct the query by type
    if intent == 'COUNT' or 'كم' in text or 'عدد' in text:
        return f"SELECT COUNT(*) FROM Students WHERE {where};"
    elif intent == 'LIST' or 'اسماء' in text or 'اعرض' in text:
        return f"SELECT first_name, father_name, family_name, age FROM Students WHERE {where} LIMIT 50;"
    elif intent == 'AVG_EXAM':
        return "SELECT AVG(exam_score) FROM Exams;"
    elif intent == 'INSTITUTE':
        return "SELECT I.institute_name, COUNT(S.national_id) FROM Institutes I LEFT JOIN Students S ON I.id = S.institute_id GROUP BY I.id, I.institute_name;"
    else:
        return f"SELECT COUNT(*) FROM Students WHERE {where};"
