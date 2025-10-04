# Database
DB_PATH = "Student.db"

# Models
PHI2_MODEL_NAME = "microsoft/phi-2"
SQLCODER_MODEL_NAME = "defog/sqlcoder-7b-2"

# SQLCoder Generation Settings
SQLCODER_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.2,
    "return_full_text": False
}

# Column Descriptions
COL_DESC = {
    "national_id": "الرقم الوطني للطالب (PK).",
    "first_name": "الاسم الأول للطالب.",
    "father_name": "اسم الأب.",
    "grandfather_name": "اسم الجد.",
    "family_name": "اسم العائلة.",
    "birth_date": "تاريخ الميلاد (yyyy-mm-dd).",
    "gender": "الجنس (ذكر/انثى).",
    "phone": "رقم هاتف الطالب.",
    "age": "العمر (محسوب تلقائياً).",
    "profession": "المهنة أو البرنامج التدريبي.",
    "qualification": "المؤهل العلمي.",
    "institute_id": "معرف المعهد (FK).",
    "residence_location_id": "الموقع السكني (FK).",
    "registration_date": "تاريخ التسجيل.",
    "region": "الإقليم (الشمالي/الجنوبي/الأوسط).",
    "area": "المنطقة داخل الإقليم.",
    "institute_name": "اسم المعهد.",
    "location_id": "معرف الموقع (FK).",
    "student_id": "الرقم الوطني للطالب (FK).",
    "exam_score": "علامة الامتحان (0-100).",
    "exam_date": "تاريخ الامتحان.",
}
