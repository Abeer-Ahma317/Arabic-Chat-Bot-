"""
AI Models - Phi-2 and SQLCoder-7B
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from .config import PHI2_MODEL_NAME, SQLCODER_MODEL_NAME, SQLCODER_CONFIG

# Global model instances
PHI2_MODEL = None
PHI2_TOKENIZER = None
SQLCODER_PIPELINE = None

def load_phi2_model():
    """تحميل Phi-2 للمحادثة والتصنيف"""
    global PHI2_MODEL, PHI2_TOKENIZER
    
    try:
        print("  - Loading Phi-2 tokenizer...")
        PHI2_TOKENIZER = AutoTokenizer.from_pretrained(PHI2_MODEL_NAME, trust_remote_code=True)
        
        print("  - Loading Phi-2 model...")
        PHI2_MODEL = AutoModelForCausalLM.from_pretrained(
            PHI2_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        PHI2_MODEL.eval()
        print("  - Phi-2 loaded successfully")
        return True
    except Exception as e:
        print(f"  - Phi-2 loading failed: {e}")
        return False

def load_sqlcoder_model():
    """تحميل SQLCoder-7B"""
    global SQLCODER_PIPELINE
    
    try:
        print("  - Loading SQLCoder tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(SQLCODER_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("  - Loading SQLCoder model (~13GB)...")
        model = AutoModelForCausalLM.from_pretrained(
            SQLCODER_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_8bit=True
        )
        model.eval()
        
        print("  - Creating pipeline...")
        SQLCODER_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **SQLCODER_CONFIG
        )
        print("  - SQLCoder loaded successfully")
        return True
    except Exception as e:
        print(f"  - SQLCoder loading failed: {e}")
        return False

def get_phi2_model():
    """الحصول على Phi-2 model"""
    return PHI2_MODEL, PHI2_TOKENIZER

def get_sqlcoder_pipeline():
    """الحصول على SQLCoder pipeline"""
    return SQLCODER_PIPELINE

def classify_intent(user_text: str) -> str:
    """تصنيف نية المستخدم"""
    lower_text = user_text.lower()
    
    chat_keywords = ['مرحبا', 'اهلا', 'السلام', 'صباح', 'مساء', 'كيف حالك',
                    'شو اخبارك', 'هاي', 'هلا', 'شكرا', 'باي', 'وداعا',
                    'من انت', 'شو انت', 'كيفك']
    
    if any(kw in lower_text for kw in chat_keywords):
        return "chat"
    
    sql_keywords = ['كم', 'عدد', 'اسماء', 'اعرض', 'اظهر', 'من', 'ما',
                   'طالب', 'طلاب', 'علامة', 'معهد', 'منطقة', 'عمر']
    
    if any(kw in lower_text for kw in sql_keywords):
        return "sql"
    
    return "general"

def chat_response(user_text: str) -> str:
    """ردود المحادثة"""
    lower_text = user_text.lower()
    
    if any(w in lower_text for w in ['مرحبا', 'اهلا', 'السلام', 'هاي', 'هلا']):
        return """مرحباً! أنا بوت ذكي للاستعلام عن قاعدة بيانات الطلاب.

أقدر أساعدك بـ:
- أعداد الطلاب (حسب العمر، الجنس، المنطقة)
- أسماء وبيانات الطلاب
- معلومات المعاهد والمناطق
- إحصائيات العلامات

جرب: "كم عدد الطلاب؟" أو "من الطلاب اللي عمرهم أكبر من 25؟" """
    
    if any(w in lower_text for w in ['كيف حالك', 'شو اخبارك', 'كيفك']):
        return "الحمد لله تمام! شكراً لسؤالك. كيف أقدر أساعدك؟"
    
    if any(w in lower_text for w in ['من انت', 'شو انت', 'ايش وظيفتك']):
        return """أنا بوت ذكي متخصص في الاستعلامات!

تم تطويري باستخدام:
- Phi-2 للمحادثة والتصنيف
- SQLCoder-7B لتوليد SQL من العربية
- 7 طبقات معالجة للغة العربية"""
    
    if any(w in lower_text for w in ['شكرا', 'مشكور', 'يعطيك العافية']):
        return "العفو! أنا موجود لخدمتك."
    
    if any(w in lower_text for w in ['باي', 'مع السلامة', 'وداعا']):
        return "مع السلامة! ارجع لي متى ما تحتاج."
    
    return "أهلاً! كيف أقدر أساعدك بقاعدة بيانات الطلاب؟"