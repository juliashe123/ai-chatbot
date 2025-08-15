import os
import shutil
import hashlib
import requests
import certifi
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
from odf import text, teletype
from odf.opendocument import load
from elasticsearch import Elasticsearch
import time
import re
import json
import jieba.analyse
import opencc
from sentence_transformers import SentenceTransformer
import logging

# === logging 設定 ===
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# === 參數設定 ===
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
DOWNLOAD_FOLDER = "docs"
CHUNK_FOLDER = os.path.join(DOWNLOAD_FOLDER, "chunks")
TEXT_FOLDER = os.path.join(DOWNLOAD_FOLDER, "text")
SCU_URL = "https://web-ch.scu.edu.tw/secretary/web_page/1655"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ES_INDEX = "mydocs"

for folder in [DOWNLOAD_FOLDER, CHUNK_FOLDER, TEXT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

es = Elasticsearch("http://localhost:9200")


# === 下載文件 ===
def download_files():
    if any(fname.startswith("doc_") for fname in os.listdir(DOWNLOAD_FOLDER)):
        logging.info("本地已有下載檔案，跳過下載階段。")
        return
    logging.info(f"開始從 {SCU_URL} 下載文件...")
    try:
        response = requests.get(
            SCU_URL, headers=HEADERS, verify=certifi.where())
        if response.status_code != 200:
            logging.error(f"下載網頁失敗，狀態碼: {response.status_code}")
            return
    except Exception as e:
        logging.error(f"下載網頁失敗，錯誤: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    extensions = [".pdf", ".doc", ".docx", ".odt"]
    file_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"].lower()
        if any(ext in href for ext in extensions) or ".ashx" in href:
            full_url = href if href.startswith(
                "http") else "https://web-ch.scu.edu.tw" + href
            file_links.append(full_url)

    logging.info(f"找到 {len(file_links)} 筆文件連結，開始下載...")
    for i, url in enumerate(file_links, 1):
        ext = os.path.splitext(url)[1]
        if ".ashx" in ext or not ext:
            ext = ".pdf"
        filename = f"doc_{i}{ext}"
        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            logging.info(f"檔案已存在，跳過下載：{filename}")
            continue
        try:
            resp = requests.get(url, headers=HEADERS, verify=False)
            with open(filepath, "wb") as f:
                f.write(resp.content)
            logging.info(f"⬇️ 已下載：{filename}")
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"下載失敗：{url}，錯誤：{e}")
    logging.info("下載完成。\n")


# === 分類與轉文字 ===
seen_hashes = set()
folders = {
    'pdf': os.path.join(DOWNLOAD_FOLDER, 'pdf'),
    'docx': os.path.join(DOWNLOAD_FOLDER, 'docx'),
    'odt': os.path.join(DOWNLOAD_FOLDER, 'odt'),
    'others': os.path.join(DOWNLOAD_FOLDER, 'others'),
    'text': TEXT_FOLDER
}
for p in folders.values():
    os.makedirs(p, exist_ok=True)


def get_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_text_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        return ''.join([page.extract_text() or '' for page in reader.pages]).strip()
    except Exception as e:
        logging.error(f"PDF 解析錯誤 ({filepath}): {e}")
        return f"【PDF 解析錯誤】: {e}"


def extract_text_docx(filepath):
    try:
        doc = Document(filepath)
        return '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        logging.error(f"DOCX 解析錯誤 ({filepath}): {e}")
        return f"【DOCX 解析錯誤】: {e}"


def extract_text_odt(filepath):
    try:
        odt_doc = load(filepath)
        texts = [teletype.extractText(p)
                 for p in odt_doc.getElementsByType(text.P)]
        return '\n'.join(texts)
    except Exception as e:
        logging.error(f"ODT 解析錯誤 ({filepath}): {e}")
        return f"【ODT 解析錯誤】: {e}"


def classify_and_extract():
    if len(os.listdir(TEXT_FOLDER)) > 0:
        logging.info("本地已有轉文字檔，跳過分類轉文字階段。")
        return
    logging.info("開始分類並轉文字...")
    for filename in os.listdir(DOWNLOAD_FOLDER):
        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
        if not os.path.isfile(filepath):
            continue
        ext = filename.lower().split('.')[-1]
        file_hash = get_file_hash(filepath)
        if file_hash in seen_hashes:
            logging.info(f"🗑️ 重複檔案略過：{filename}")
            continue
        seen_hashes.add(file_hash)
        try:
            text_content = None
            if ext == 'pdf':
                dest = os.path.join(folders['pdf'], filename)
                shutil.move(filepath, dest)
                text_content = extract_text_pdf(dest)
            elif ext == 'docx':
                dest = os.path.join(folders['docx'], filename)
                shutil.move(filepath, dest)
                text_content = extract_text_docx(dest)
            elif ext == 'odt':
                dest = os.path.join(folders['odt'], filename)
                shutil.move(filepath, dest)
                text_content = extract_text_odt(dest)
            else:
                shutil.move(filepath, os.path.join(
                    folders['others'], filename))
            if text_content:
                txt_path = os.path.join(
                    folders['text'], os.path.splitext(filename)[0] + '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                logging.info(f"📄 轉文字完成：{txt_path}")
        except Exception as e:
            logging.warning(f"解析錯誤：{filename} → {e}")
    logging.info("分類轉文字完成。\n")
# === 切段 ===


def load_texts_from_folder(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts


def split_text(text, max_len=500, overlap=50):
    raw_sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_len:
            chunks.append(current.strip())
            current = current[-overlap:]
        current += sentence
    if current:
        chunks.append(current.strip())

    return chunks


def chunk_texts():
    if len(os.listdir(CHUNK_FOLDER)) > 0:
        logging.info("本地已有切段檔，跳過切段階段。")
        return
    logging.info("開始切段...")
    all_texts = load_texts_from_folder(TEXT_FOLDER)
    all_chunks = []
    for text in all_texts:
        all_chunks.extend(split_text(text))
    for i, chunk in enumerate(all_chunks, 1):
        with open(f"{CHUNK_FOLDER}/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)
    logging.info(f"已切出 {len(all_chunks)} 個段落。\n")


# === 建立 Elasticsearch 索引及 Mapping ===
def create_es_index():
    logging.info("🔧 建立 Elasticsearch 向量索引...")
    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "keywords": {"type": "text"},
                "summary": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        }
    }

    if es.indices.exists(index=ES_INDEX):
        logging.info("Elasticsearch 索引已存在，跳過建立索引階段。")
        return
    es.indices.create(index=ES_INDEX, body=mapping)
    logging.info("✅ Elasticsearch 索引建立完成。\n")


def get_embedding(text):
    try:
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"⚠️ 產生 embedding 失敗：{e}")
        return None


def extract_keywords(text, top_k=5):
    return jieba.analyse.extract_tags(text, topK=top_k)


def extract_summary(text, ratio=0.3):
    try:
        from gensim.summarization import summarize
        return summarize(text, ratio=ratio)
    except Exception:
        return ""


def embed_and_index():
    chunk_files = sorted(os.listdir(CHUNK_FOLDER))
    existing_count = 0
    if es.indices.exists(index=ES_INDEX):
        existing_count = es.count(index=ES_INDEX)['count']
    if existing_count >= len(chunk_files) and len(chunk_files) > 0:
        logging.info("Elasticsearch 已有足夠資料，跳過產生向量與索引階段。")
        return
    logging.info("🚀 開始產生向量並寫入 Elasticsearch...")
    for i, fname in enumerate(chunk_files, 1):
        path = os.path.join(CHUNK_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logging.warning(f"⚠️ 空白段落跳過：{fname}")
            continue
        embedding = get_embedding(text)
        if embedding is None:
            logging.error(f"❌ 向量產生失敗，跳過：{fname}")
            continue
        keywords = extract_keywords(text)
        summary = extract_summary(text)

        doc = {
            "content": text,
            "keywords": ", ".join(keywords),
            "summary": summary,
            "embedding": embedding
        }
        es.index(index=ES_INDEX, id=i, document=doc)
        logging.info(f"✅ 已索引段落 {i}/{len(chunk_files)}")
    logging.info("🎉 全部段落完成索引。\n")


# === 向量檢索函式 ===
def search_by_vector(query, size=5):
    query_vec = get_embedding(query)
    if query_vec is None:
        logging.error("❌ 無法產生查詢向量")
        return None

    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": query
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": """
                        double semantic = cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                        double keyword = _score; 
                        return 0.7 * semantic + 0.3 * keyword;
                    """,
                    "params": {
                        "query_vector": query_vec
                    }
                }
            }
        }
    }

    res = es.search(index=ES_INDEX, body=body)
    hits = res["hits"]["hits"]

    logging.info(f"🔍 混合檢索「{query}」結果：")
    for i, hit in enumerate(hits, 1):
        score = hit["_score"]
        content = hit["_source"].get("content", "")
        logging.info(f"  {i}. 分數: {score:.4f}，內容前30字: {content[:30]}...")
    return hits


# === 中文關鍵詞抽取函式
def extract_chinese_keywords(text, top_k=10):
    if not text:
        return []
    try:
        return jieba.analyse.extract_tags(text, topK=top_k)
    except Exception as e:
        logging.error(f"❌ 關鍵字提取失敗：{e}")
        return []


# 建立簡轉繁轉換器
cc = opencc.OpenCC('s2t')


def to_traditional(text):
    return cc.convert(text)


# === 改良版關鍵詞匹配 ===
def contains_keywords_loose(answer, context, match_threshold=0.6):
    answer = to_traditional(answer)
    context = to_traditional(context)

    ans_clean = re.sub(r'\s+', '', answer.lower())
    ctx_clean = re.sub(r'\s+', '', context.lower())
    ans_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', ans_clean)
    ctx_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', ctx_clean)

    tokens = re.findall(r'[\u4e00-\u9fff]+', ans_clean)
    match_count = sum(1 for t in tokens if t and t in ctx_clean)

    en_tokens = re.findall(r'[A-Za-z0-9]+', ans_clean)
    en_match_count = sum(1 for t in en_tokens if t and t in ctx_clean)

    total_tokens = len(tokens) + len(en_tokens)
    if total_tokens == 0:
        return False

    return (match_count + en_match_count) / total_tokens >= match_threshold


# === 使用 Gemma 摘要單段文字（150±10字，自然斷句版） ===
def summarize_text(text):
    if not text:
        return ""

    prompt = f"請用簡潔中文摘要以下段落，控制在140~160字之間，且不要中斷句子：\n{text}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt},
            stream=True
        )

        full_answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    full_answer += data.get("response", "")
                except:
                    continue

        summary = full_answer.strip()
        if len(summary) > 160:
            cut_pos = summary.rfind("。", 140, 160)
            if cut_pos == -1:
                cut_pos = summary.rfind("，", 140, 160)
            if cut_pos == -1:
                cut_pos = 150
            summary = summary[:cut_pos].rstrip("，。")

    except Exception as e:
        logging.error(f"❌ 摘要失敗：{e}")
        return ""

    if len(summary) < 140 and summary != "":
        prompt2 = f"請補充剛剛的摘要，使內容更完整，仍然保持簡潔且長度接近150字，不要中斷句子：\n{summary}"
        try:
            response2 = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:1b", "prompt": prompt2},
                stream=True
            )
            full_answer2 = ""
            for line in response2.iter_lines():
                if line:
                    try:
                        data2 = json.loads(line.decode("utf-8"))
                        full_answer2 += data2.get("response", "")
                    except:
                        continue
            summary2 = full_answer2.strip()
            if len(summary2) > 160:
                cut_pos = summary2.rfind("。", 140, 160)
                if cut_pos == -1:
                    cut_pos = summary2.rfind("，", 140, 160)
                if cut_pos == -1:
                    cut_pos = 150
                summary2 = summary2[:cut_pos].rstrip("，。")
            if len(summary2) > len(summary):
                summary = summary2
        except Exception as e:
            logging.error(f"❌ 補充摘要失敗：{e}")

    return summary


# === 多段摘要合併（保留未改）===
def summarize_multiple_hits(hits, top_n=3):
    summaries = []
    for hit in hits[:top_n]:
        content = hit.get("_source", {}).get("content", "")
        if content:
            summary = summarize_text(content)
            if summary:
                summaries.append(summary)
    return "\n".join(summaries)


# === 使用 Gemma 回答問題 ===
def ask_gemma(question, context):
    context = context or ""

    prompt = f"""你是一位嚴謹的秘書室問答助手，
只能根據提供的文件內容回答問題。
若文件中沒有明確提到問題的答案，請直接回答「查無相關資料」，不要猜測或虛構答案。
以下是參考文件內容：
========
{context}
========

請回答下面這個問題：
「{question}」
回答要求：
- 只能根據上面文件內容作答
- 若找不到明確答案，直接回答「查無相關資料」
- 不要加入個人推測、意見或額外說明
- 回答務必簡短扼要，字數請嚴格限制 50 字以內

"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt},
            stream=True
        )

        full_answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("response", "")
                    full_answer += chunk
                except Exception:
                    continue

        full_answer = full_answer.strip()
        if len(full_answer) > 50:
            cut_pos = full_answer.find("。", 40, 60)
            if cut_pos == -1:
                cut_pos = full_answer.find("，", 40, 60)
            if cut_pos != -1:
                full_answer = full_answer[:cut_pos + 1]
            else:
                full_answer = full_answer[:60]

        if not full_answer or full_answer == "":
            final_answer = "不好意思，請您更具體提問"
        elif full_answer == "查無相關資料":
            final_answer = "查無相關資料"
        else:
            final_answer = full_answer

        logging.info("\n🤖 模型回答：%s", full_answer)
        logging.info("💡 最終回答：%s", final_answer)
        return final_answer

    except Exception as e:
        logging.error(f"❌ Gemma 回答錯誤：{e}")
        return "不好意思，請您更具體"


# === 主聊天迴圈（單段摘要版）===
def chat_loop():
    print("歡迎使用東吳大學秘書室小幫手，請輸入問題，輸入 q 離開。\n")
    while True:
        question = input("請輸入您的問題 (q 離開): ").strip()
        if question.lower() == 'q':
            print("再見！")
            break
        if not question:
            continue

        hits = search_by_vector(question, size=3)
        if not hits:
            print("查無相關資料，不好意思，請您更具體提問。")
            continue

        # 只摘要分數最高的第一筆段落內容
        top_hit = hits[0]
        top_content = top_hit["_source"].get("content", "")
        summarized_context = summarize_text(top_content)

        answer = ask_gemma(question, summarized_context)
        print(f"\n【秘書室回覆】\n{answer}\n")


# === 主流程 ===
def main():
    download_files()
    classify_and_extract()
    chunk_texts()
    create_es_index()
    embed_and_index()
    chat_loop()


if __name__ == "__main__":
    main()
