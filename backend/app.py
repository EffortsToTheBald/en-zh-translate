# backend/app.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.translator import EN2ZHTranslator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Custom EN→ZH Translation Service")

# CORS（允许 React 开发服务器访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局加载一次模型（启动时）
try:
    translator = EN2ZHTranslator()
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    translator = None

class TranslateRequest(BaseModel):
    text: str
    temperature: float = 0.8  # 可选参数

class TranslateResponse(BaseModel):
    translation: str

@app.get("/")
async def root():
    return {"message": "EN → ZH Translation API"}

@app.post("/translate", response_model=TranslateResponse)
async def translate_api(req: TranslateRequest):
    if translator is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    
    try:
        result = translator.translate(
            sentence=req.text.strip(),
            temperature=max(0.1, min(2.0, req.temperature))  # 限制范围
        )
        print(result)
        return TranslateResponse(translation=result)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
# backend/app.py 的最底部
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)