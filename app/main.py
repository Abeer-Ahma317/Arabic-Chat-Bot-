"""
FastAPI Application - نقطة الدخول الرئيسية
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .database import initialize_database
from .models import load_phi2_model, load_sqlcoder_model
from .responses import execute_query

app = FastAPI(title="Smart Chatbot - Phi-2 + SQLCoder")

@app.on_event("startup")
async def startup_event():
    """تشغيل عند بدء التطبيق"""
    print("\n" + "="*70)
    print("SMART CHATBOT STARTUP")
    print("="*70)
    
    print("\n[1/3] Initializing Database...")
    if not initialize_database():
        print("ERROR: Database initialization failed!")
        return
    
    print("\n[2/3] Loading Phi-2...")
    load_phi2_model()
    
    print("\n[3/3] Loading SQLCoder-7B...")
    if load_sqlcoder_model():
        print("\n" + "="*70)
        print("STARTUP COMPLETE - SERVER READY")
        print("="*70)
        print("\nExample Questions:")
        print("  • مرحباً!")
        print("  • كم عدد الطلاب؟")
        print("  • من الطلاب اللي عمرهم أكبر من 25؟")
        print("  • كم عدد الإناث من الشمال؟")
        print("  • اعرض أسماء الطلاب بين 20 و 30 سنة")
        print("="*70 + "\n")

@app.get("/")
def home():
    """الصفحة الرئيسية"""
    with open("templates/chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/ping")
def ping():
    """فحص حالة السيرفر"""
    return {"status": "ok", "message": "Bot is running"}

@app.post("/ask")
async def ask(request: dict):
    """معالجة السؤال"""
    question = request.get('question', '').strip()
    
    if not question:
        return JSONResponse({
            'success': False,
            'answer': 'الرجاء إدخال سؤال',
            'sql': None,
            'results': None
        })
    
    try:
        result = execute_query(question)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return JSONResponse({
            'success': False,
            'answer': f'خطأ: {str(e)}',
            'sql': None,
            'results': None
        })

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """معالج الأخطاء العام"""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)