#!/usr/bin/env python3
import os
from rank_bm25 import BM25Okapi
import numpy as np
import sys
import json
import time
import queue
import pathlib
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. HỆ THỐNG QUẢN LÝ POOL (ALIVE / DEAD)
# ==========================================
class DynamicPoolManager:
    def __init__(self):
        self.embed_pool = queue.Queue()
        self.llm_pool = queue.Queue()
        self._init_embed_pool()
        self._init_llm_pool()

    def _init_embed_pool(self):
        print("--- ĐANG QUÉT KEY CHO EMBEDDING (OpenRouter - 1536d) ---")
        or_keys = [val for key, val in os.environ.items() if key.startswith("OPENROUTER_KEY_")]
        for idx, key in enumerate(or_keys):
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                # Test nhẹ 1 request nhúng
                client.embeddings.create(input="test", model="openai/text-embedding-3-small")
                self.embed_pool.put((f"OR_Embed_{idx+1}", client))
                print(f"  [+] OpenRouter Embed Key {idx+1}: SỐNG")
            except Exception as e:
                print(f"  [-] OpenRouter Embed Key {idx+1}: CHẾT ({str(e)[:60]})")
        
        if self.embed_pool.empty():
            print("[!] LỖI NGHIÊM TRỌNG: Không có key OpenRouter nào sống để nhúng vector!")
            sys.exit(1)
        print(f"=> Tổng cộng có {self.embed_pool.qsize()} key Embedding sẵn sàng.\n")

    def _init_llm_pool(self):
        print("--- ĐANG QUÉT KEY CHO LLM (ChatGPT & Gemini) ---")
        
        # 1. Quét OpenRouter cho LLM (ChatGPT)
        or_keys = [val for key, val in os.environ.items() if key.startswith("OPENROUTER_KEY_")]
        for idx, key in enumerate(or_keys):
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5
                )
                self.llm_pool.put({
                    'provider': 'openrouter',
                    'client': client,
                    'name': f"ChatGPT_OR_{idx+1}"
                })
                print(f"  [+] ChatGPT (OpenRouter) Key {idx+1}: SỐNG")
            except Exception as e:
                print(f"  [-] ChatGPT (OpenRouter) Key {idx+1}: CHẾT ({str(e)[:60]})")

        # 2. Quét Gemini cho LLM
        gemini_keys = [val for key, val in os.environ.items() if key.startswith("GEMINI_KEY_")]
        for idx, key in enumerate(gemini_keys):
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                model.generate_content("hi")
                self.llm_pool.put({
                    'provider': 'gemini',
                    'key': key,
                    'name': f"Gemini_{idx+1}"
                })
                print(f"  [+] Gemini Key {idx+1}: SỐNG")
            except Exception as e:
                print(f"  [-] Gemini Key {idx+1}: CHẾT ({str(e)[:60]})")

        if self.llm_pool.empty():
            print("[!] LỖI NGHIÊM TRỌNG: Không có key LLM nào sống để trả lời!")
            sys.exit(1)
        print(f"=> Tổng cộng có {self.llm_pool.qsize()} key LLM sẵn sàng luân phiên.\n")

    def embed_query(self, query_text):
        """Lấy 1 key sống từ hàng đợi, nhúng xong lại nhét vào cuối hàng đợi"""
        name, client = self.embed_pool.get()
        try:
            res = client.embeddings.create(input=query_text, model="openai/text-embedding-3-small")
            return res.data[0].embedding
        finally:
            self.embed_pool.put((name, client))

    def generate_answer(self, prompt):
        """Lấy 1 key sống luân phiên ChatGPT/Gemini, sinh text xong lại nhét vào"""
        item = self.llm_pool.get()
        provider = item['provider']
        name = item['name']
        
        print(f"    -> Đang dùng model LLM: {name}")
        try:
            if provider == 'openrouter':
                res = item['client'].chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                text = res.choices[0].message.content
            else:
                genai.configure(api_key=item['key'])
                model = genai.GenerativeModel('gemini-1.5-flash')
                res = model.generate_content(
                    prompt, 
                    generation_config={"response_mime_type": "application/json"}
                )
                text = res.text
                
            # Đảm bảo làm sạch code block markdown nếu có
            text = text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            
            return json.loads(text.strip())
            
        except Exception as e:
            print(f"    [!] Lỗi khi gọi {name}: {e}")
            return {"answer": "Lỗi hệ thống từ LLM", "citations": []}
        finally:
            self.llm_pool.put(item)
            time.sleep(1) # Tránh bị rate limit


# ==========================================
# 2. KHỞI TẠO CÁC KẾT NỐI CHUNG
# ==========================================
BASE = pathlib.Path(__file__).parent.parent
QPATH = BASE / "benchmark" / "questions_day7.json"
RESULTS_DIR = BASE / "results"

def get_db_connection():
    conn = psycopg2.connect(
        host="127.0.0.1", port="55433",
        user="day6", password="day6_local_only", dbname="day6_lab"
    )
    register_vector(conn)
    return conn

# ==========================================
# 3. HÀM TÌM KIẾM BẰNG PGVECTOR
# ==========================================
def dense_search(conn, pool_manager, query_text, k=3):
    query_vector = pool_manager.embed_query(query_text)
    
    chunks = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, source, page, (embedding <=> %s::vector) AS distance
            FROM chunks_table
            ORDER BY distance ASC
            LIMIT %s;
        """, (query_vector, k))
        
        rows = cur.fetchall()
        for row in rows:
            chunks.append({
                "id": row[0], "text": row[1], "source": row[2], "page": row[3]
            })
    return chunks

def load_all_chunks_for_bm25(conn):
    chunks = []
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text, source, page FROM chunks_table;")
        rows = cur.fetchall()
        for row in rows:
            chunks.append({"id": row[0], "text": row[1], "source": row[2], "page": row[3]})
    return chunks

def bm25_search(query_text, all_chunks, k=3):
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_text.lower().split()
    
    scores = bm25.get_scores(tokenized_query)
    top_n_idx = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_n_idx:
        if scores[idx] > 0:
            results.append(all_chunks[idx])
    return results

# ==========================================
# HÀM TÌM KIẾM HYBRID (RRF)
# ==========================================
def hybrid_search(conn, pool_manager, query_text, all_chunks, k=3):
    # Lấy top 10 từ cả 2 phương pháp để trộn
    dense_results = dense_search(conn, pool_manager, query_text, k=10)
    bm25_results = bm25_search(query_text, all_chunks, k=10)
    
    # Tính điểm Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    
    for rank, chunk in enumerate(dense_results):
        fused_scores[chunk["id"]] = fused_scores.get(chunk["id"], 0) + 1 / (60 + rank + 1)
        
    for rank, chunk in enumerate(bm25_results):
        fused_scores[chunk["id"]] = fused_scores.get(chunk["id"], 0) + 1 / (60 + rank + 1)
        
    # Sắp xếp và lấy top k
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:k]
    
    # Lấy data gốc từ id
    id_to_chunk = {c["id"]: c for c in all_chunks}
    return [id_to_chunk[cid] for cid in sorted_ids if cid in id_to_chunk]

# ==========================================
# CẬP NHẬT LUỒNG CHẠY CHÍNH
# ==========================================
def run_agent(method="dense"):
    raw_data = json.loads(QPATH.read_text(encoding="utf-8"))
    questions = raw_data["questions"]
    RESULTS_DIR.mkdir(exist_ok=True)
    
    pool_manager = DynamicPoolManager()
    conn = get_db_connection()
    
    all_chunks = []
    if method in ["bm25", "hybrid"]:
        print("Đang nạp dữ liệu cho BM25...")
        all_chunks = load_all_chunks_for_bm25(conn)
    
    print(f"--- BẮT ĐẦU CHẠY BENCHMARK ({method.upper()}) ---")
    traces = []
    
    for q in questions:
        print(f"Câu: {q['id']} - {q['question']}")
        t0 = time.time()
        
        # Chọn phương pháp Retrieval
        if method == "dense":
            retrieved_chunks = dense_search(conn, pool_manager, q["question"], k=3)
        elif method == "bm25":
            retrieved_chunks = bm25_search(q["question"], all_chunks, k=3)
        else:
            retrieved_chunks = hybrid_search(conn, pool_manager, q["question"], all_chunks, k=3)
            
        context_ids = [c["id"] for c in retrieved_chunks]
        context_text = "\n\n".join([f"[ID: {c['id']}] {c['text']}" for c in retrieved_chunks])
        
        prompt = f"""Dựa vào tài liệu sau:
{context_text}
Hãy trả lời câu hỏi: "{q['question']}"
Quy tắc:
1. CHỈ dùng thông tin có trong tài liệu.
2. Nếu không đủ bằng chứng, trả lời ĐÚNG CỤM TỪ: "Không đủ bằng chứng".
3. Trả về JSON: {{"answer": "...", "citations": ["id_1"]}}"""
        
        llm_response = pool_manager.generate_answer(prompt)
        answer = llm_response.get("answer", "")
        citations = llm_response.get("citations", [])
        
        status = "abstain" if "không đủ bằng chứng" in answer.lower() or not citations else "answered"
        rt = time.time() - t0
        
        traces.append({
            "query_id": q["id"], "split": q["split"], "question": q["question"],
            "status": status, "answer": answer, "citations": citations,
            "events": [{"event": "search", "method": method, "top_k": 3, "context_ids": context_ids, "context_chars": len(context_text)}],
            "elapsed_ms": round(rt * 1000, 2)
        })

    conn.close()
    out_file = RESULTS_DIR / f"traces_{method}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    print(f"[OK] Đã chạy xong {method}! Log: {out_file}")

if __name__=="__main__":
    run_agent("dense")
    run_agent("bm25")
    run_agent("hybrid")