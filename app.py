import asyncio
import logging
from fastapi import FastAPI, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List

# 這邊匯入原本的程式
from rag import (
    search_by_vector, summarize_text, ask_gemma,
    download_files, classify_and_extract, chunk_texts,
    create_es_index, embed_and_index
)

app = FastAPI(title="SCU 秘書室 API", version="1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("startup")

# 背景初始化任務
async def init_data():
    logger.info("🚀 背景初始化資料中...")
    await asyncio.to_thread(download_files)
    await asyncio.to_thread(classify_and_extract)
    await asyncio.to_thread(chunk_texts)
    await asyncio.to_thread(create_es_index)
    await asyncio.to_thread(embed_and_index)
    logger.info("✅ 資料初始化完成！")

@app.on_event("startup")
async def startup_event():
    # 在背景啟動，不阻塞 FastAPI 啟動
    asyncio.create_task(init_data())
    logger.info("FastAPI 已啟動，資料初始化在背景進行")

# 查詢參數資料結構
class SearchResponse(BaseModel):
    question: str
    answer: str
    top_context: str
    summarized_context: str

@app.get("/search", response_model=SearchResponse)
def search(question: str = Query(..., description="要詢問的問題")):
    hits = search_by_vector(question, size=3)
    if not hits:
        return {
            "question": question,
            "answer": "查無相關資料",
            "top_context": "",
            "summarized_context": ""
        }

    top_hit = hits[0]
    top_content = top_hit["_source"].get("content", "")
    summarized_context = summarize_text(top_content)
    answer = ask_gemma(question, summarized_context)

    return {
        "question": question,
        "answer": answer,
        "top_context": top_content,
        "summarized_context": summarized_context
    }

@app.post("/reload")
def reload_data(background_tasks: BackgroundTasks):
    """重新抓取資料與建立索引（背景執行）"""
    background_tasks.add_task(init_data)
    return {"status": "資料重新載入已開始，請稍後"}

