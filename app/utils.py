"""
Utility functions -Text normalization and auxiliary tools
"""
import re
import pandas as pd

def normalize_arabic(text: str) -> str:
    
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    text = text.replace("ى", "ي").replace("ـ", "")
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    return text

def extract_numbers(text: str) -> list:
    
    return [int(n) for n in re.findall(r'\d+', text)]

def clean_sql(sql: str) -> str:
    
    if not sql:
        return None
    
    # Remove text before SELECT
    sql = re.sub(r'^.*?(SELECT)', r'\1', sql, flags=re.IGNORECASE | re.DOTALL)
    
    #  ILIKE → LIKE
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    
    # Cleaning spaces
    sql = re.sub(r'\s+', ' ', sql)
    
    #Remove multiple semicolons
    sql = re.sub(r';+', ';', sql)
    
    # Verify;
    if not sql.endswith(';'):
        sql = sql.rstrip(';') + ';'
    
    return sql.strip()
