from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import logging

# 這邊匯入你原本的功能
from rag import (
    search_by_vector, summarize_text, ask_gemma,
    download_files, classify_and_extract, chunk_texts,
    create_es_index, embed_and_index
)

app = FastAPI(title="SCU 秘書室 API", version="1.0")

# 啟動時先建立索引與資料


@app.on_event("startup")
def startup_event():
    logging.info("🚀 啟動 API 並初始化資料...")
    download_files()
    classify_and_extract()
    chunk_texts()
    create_es_index()
    embed_and_index()

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

    # 取分數最高的段落
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
def reload_data():
    """重新抓取資料與建立索引"""
    download_files()
    classify_and_extract()
    chunk_texts()
    create_es_index()
    embed_and_index()
    return {"status": "資料已重新載入並建立索引完成"}
