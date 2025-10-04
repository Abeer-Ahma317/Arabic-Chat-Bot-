"""
Database initialization and management
"""
import sqlite3
import pandas as pd
from datetime import datetime, date
import os
from .config import DB_PATH

# Global connection
CONN = None
SCHEMA_DICT = {}

def add_age_column():
    """إضافة وحساب عمود العمر"""
    try:
        cursor = CONN.cursor()
        cursor.execute("PRAGMA table_info(Students)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'age' not in columns:
            cursor.execute("ALTER TABLE Students ADD COLUMN age INTEGER")
            print("  - Age column added")

        cursor.execute("SELECT national_id, birth_date FROM Students WHERE birth_date IS NOT NULL")
        today = date.today()

        for national_id, birth_date_str in cursor.fetchall():
            try:
                birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
                age = today.year - birth_date.year - (
                    (today.month, today.day) < (birth_date.month, birth_date.day)
                )
                cursor.execute("UPDATE Students SET age = ? WHERE national_id = ?", (age, national_id))
            except:
                pass

        CONN.commit()
        print("  - Ages calculated successfully")
    except Exception as e:
        print(f"  - Age calculation error: {e}")

def build_schema():
    """بناء Schema Dictionary"""
    global SCHEMA_DICT

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", CONN)
    SCHEMA_DICT = {}

    for table in tables['name']:
        if table == 'sqlite_sequence':
            continue
        cols = pd.read_sql(f"PRAGMA table_info([{table}]);", CONN)
        SCHEMA_DICT[table] = {
            'columns': [col['name'] for col in cols.to_dict('records')],
            'types': {col['name']: col['type'] for col in cols.to_dict('records')}
        }

    print(f"  - Schema built for {len(SCHEMA_DICT)} tables")

def initialize_database():
    """تهيئة قاعدة البيانات"""
    global CONN

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        return False

    try:
        CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        print("  - Database connected")
        add_age_column()
        build_schema()
        return True
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")
        return False

def get_connection():
    """الحصول على الاتصال الحالي"""
    return CONN

def get_schema():
    """الحصول على Schema"""
    return SCHEMA_DICT
