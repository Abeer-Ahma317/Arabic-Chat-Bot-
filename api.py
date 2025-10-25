from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os
from pathlib import Path
import atexit
from ai_query import AIDataQuery

# تطبيق الـ Nest AsyncIO
nest_asyncio.apply()

# التأكد من وجود المجلدات المطلوبة
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# إعداد قاعدة البيانات
mysql_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306))
}

# إنشاء الـ AI Agent
agent = AIDataQuery(db_config=mysql_config, db_type="mysql")

# نموذج الطلب
class QuestionRequest(BaseModel):
    question: str

# إنشاء تطبيق FastAPI
app = FastAPI(title="AI DataQuery Agent")

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    """الصفحة الرئيسية"""
    return {
        "status": "✅ running",
        "message": "AI DataQuery Agent is active"
    }

@app.post("/ask")
async def ask(req: QuestionRequest):
    """معالجة الأسئلة"""
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        result = agent.query(req.question)
        return {
            "question": req.question,
            "intent": result.get("intent"),
            "answer": result.get("answer"),
            "sql": result.get("sql"),
            "results": result.get("results")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """واجهة المستخدم"""
    try:
        template_path = TEMPLATES_DIR / "index.html"
        if not template_path.exists():
            raise HTTPException(
                status_code=500,
                detail="UI template not found. Please ensure templates/index.html exists."
            )
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading UI template: {str(e)}"
        )

def cleanup():
    """تنظيف الموارد عند إيقاف التطبيق"""
    try:
        if agent and agent.conn:
            agent.close()
            print("✅ Database connection closed successfully")
    except Exception as e:
        print(f"⚠️ Error closing database connection: {e}")

def start():
    """تشغيل التطبيق"""
    try:
        # التأكد من وجود الملفات المطلوبة
        if not (TEMPLATES_DIR / "index.html").exists():
            raise FileNotFoundError(
                "templates/index.html not found! Please ensure all required files exist."
            )
        
        # تسجيل دالة التنظيف
        atexit.register(cleanup)
        
        # إعداد ngrok
        try:
            ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
            if not ngrok_token:
                raise ValueError("NGROK_AUTH_TOKEN not found in environment variables")
            ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(8000)
            public_url = tunnel.public_url
            
            print(f"\n{'='*60}")
            print(f"✅ Server starting successfully!")
            print(f"🌐 Web UI: {public_url}/ui")
            print(f"🚀 API Endpoint: {public_url}/ask")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"⚠️ Warning: Ngrok setup failed: {e}")
            print("🔄 Continuing with local server only...")
            public_url = "http://localhost:8000"
        
        # تشغيل السيرفر
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )
    
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        cleanup()
        raise

if __name__ == "__main__":
    start()