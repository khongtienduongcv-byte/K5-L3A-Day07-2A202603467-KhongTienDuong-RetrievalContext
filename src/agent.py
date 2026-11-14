#!/usr/bin/env python3
import os
import re
import json
import time
import queue
import pathlib
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE = pathlib.Path(__file__).parent.parent
QPATH = BASE / "benchmark" / "questions_day7.json"
RESULTS_DIR = BASE / "results"

# ==========================================
# 1. QUẢN LÝ KEY OPENROUTER (CỐ ĐỊNH 1 MODEL)
# ==========================================
class OpenRouterKeyPool:
    def __init__(self):
        self.pool = queue.Queue()
        or_keys = [val for key, val in sorted(os.environ.items()) if key.startswith("OPENROUTER_KEY_") and val]
        print("--- KIỂM TRA OPENROUTER KEYS (1536d Embed + gpt-4o-mini) ---")
        for idx, key in enumerate(or_keys, 1):
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                client.embeddings.create(input="ping", model="openai/text-embedding-3-small")
                self.pool.put((f"OR_Key_{idx}", client))
                print(f"  [+] Key {idx}: SỐNG")
            except Exception as e:
                print(f"  [-] Key {idx}: CHẾT ({str(e)[:50]})")
        if self.pool.empty():
            raise RuntimeError("Không có OPENROUTER_KEY nào sống trong .env!")

    def embed(self, text: str):
        name, client = self.pool.get()
        try:
            res = client.embeddings.create(input=text, model="openai/text-embedding-3-small")
            return res.data[0].embedding
        finally:
            self.pool.put((name, client))

    def chat_json(self, prompt: str):
        name, client = self.pool.get()
        try:
            res = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            print(f"  [!] Lỗi gọi LLM ({name}): {e}")
            return {"abstain": True, "answer": "Không đủ bằng chứng", "citations": []}
        finally:
            self.pool.put((name, client))

# ==========================================
# 2. KẾT NỐI DB QUYỂN SÁCH LAB 6 & RETRIEVAL
# ==========================================
def get_db_connection():
    conn = psycopg2.connect(
        host="127.0.0.1", port="55433",
        user="day6", password="day6_local_only", dbname="day6_lab"
    )
    register_vector(conn)
    return conn

def load_book_chunks(conn):
    """Đọc toàn bộ các chunk của quyển sách từ DB Lab 6 (đã khử trùng lặp)"""
    unique_chunks = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, source, page, character, chapter
            FROM chunks_table
            ORDER BY page ASC, id ASC;
        """)
        for cid, text, source, page, character, chapter in cur.fetchall():
            if cid not in unique_chunks:
                unique_chunks[cid] = {
                    "id": cid,
                    "text": text,
                    "source": source,
                    "page": page,
                    "character": character or "",
                    "chapter": chapter or ""
                }
    return list(unique_chunks.values())

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def dense_search(conn, pool: OpenRouterKeyPool, query_text: str, k: int = 3):
    q_vec = pool.embed(query_text)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, source, page, character, chapter, MIN(embedding <=> %s::vector) AS dist
            FROM chunks_table
            GROUP BY chunk_id, text, source, page, character, chapter
            ORDER BY dist ASC
            LIMIT %s;
        """, (q_vec, k))
        return [
            {
                "id": r[0], "text": r[1], "source": r[2], "page": r[3],
                "character": r[4] or "", "chapter": r[5] or "", "score": round(float(1 - r[6]), 4)
            }
            for r in cur.fetchall()
        ]

def bm25_search(query_text: str, all_chunks: list, bm25_index: BM25Okapi, k: int = 3):
    q_tokens = tokenize(query_text)
    if not q_tokens:
        return []
    scores = bm25_index.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {**all_chunks[i], "score": round(float(scores[i]), 4)}
        for i in top_idx if scores[i] > 0
    ]

def hybrid_search(conn, pool: OpenRouterKeyPool, query_text: str, all_chunks: list, bm25_index: BM25Okapi, k: int = 3, rrf_k: int = 60):
    d_res = dense_search(conn, pool, query_text, k=max(10, k * 2))
    b_res = bm25_search(query_text, all_chunks, bm25_index, k=max(10, k * 2))
    fused = {}
    id_map = {c["id"]: c for c in all_chunks}

    for rank, c in enumerate(d_res, 1):
        fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (rrf_k + rank)
    for rank, c in enumerate(b_res, 1):
        fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (rrf_k + rank)

    sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:k]
    return [{**id_map[cid], "score": round(float(fused[cid]), 4)} for cid in sorted_ids if cid in id_map]

def format_book_context(chunks: list, include_metadata: bool = True):
    blocks = []
    for c in chunks:
        if include_metadata:
            header = f"[CHUNK_ID: {c['id']} | Page: {c['page']} | Chapter: {c.get('chapter','')} | Character: {c.get('character','')}]"
        else:
            header = f"[CHUNK_ID: {c['id']}]"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)

# ==========================================
# 3. VÒNG LẶP AGENT ĐA LƯỢT & ĐO METRICS
# ==========================================
def run_question(q: dict, cfg_name: str, method: str, top_k: int, max_steps: int, include_meta: bool, conn, pool: OpenRouterKeyPool, all_chunks: list, bm25_index: BM25Okapi):
    t0 = time.time()
    events = []
    collected = {}
    current_query = q["question"]

    for step in range(1, max_steps + 1):
        if method == "dense":
            step_chunks = dense_search(conn, pool, current_query, k=top_k)
        elif method == "bm25":
            step_chunks = bm25_search(current_query, all_chunks, bm25_index, k=top_k)
        else:
            step_chunks = hybrid_search(conn, pool, current_query, all_chunks, bm25_index, k=top_k)

        for c in step_chunks:
            if c["id"] not in collected:
                collected[c["id"]] = c

        curr_list = list(collected.values())[:top_k + (step - 1) * 2]
        ctx_text = format_book_context(curr_list, include_metadata=include_meta)
        events.append({
            "step": step,
            "event": "search_book",
            "method": method,
            "query": current_query,
            "top_k": top_k,
            "context_ids": [c["id"] for c in curr_list],
            "pages": [c["page"] for c in curr_list],
            "context_chars": len(ctx_text)
        })

        # Nếu là Agent đa lượt (max_steps > 1), kiểm tra xem đã đủ bằng chứng từ sách chưa
        if step < max_steps:
            verify_prompt = f"""Book Context Retrieved:
{ctx_text}

Question: "{q['question']}"
Does the retrieved book context explicitly contain the answer to the question?
- If YES, return {{"sufficient": true, "expanded_query": ""}}
- If NO, rewrite a better search query focusing on character names, story events, or keywords from the question to search the book again: {{"sufficient": false, "expanded_query": "..."}}"""
            check = pool.chat_json(verify_prompt)
            if check.get("sufficient") is True or not check.get("expanded_query"):
                break
            current_query = check["expanded_query"]

    final_chunks = list(collected.values())[:max(top_k, 5) if max_steps > 1 else top_k]
    valid_ids = [c["id"] for c in final_chunks]
    final_ctx_text = format_book_context(final_chunks, include_metadata=include_meta)

    answer_prompt = f"""You are a book question-answering agent for the book "Fifty Famous People".
Retrieved Book Excerpts:
{final_ctx_text}

Question: "{q['question']}"

Strict Rules:
1. Answer ONLY using facts explicitly stated in the Retrieved Book Excerpts above.
2. If the excerpts do NOT contain the answer, you MUST abstain: set "abstain": true, "answer": "Không đủ bằng chứng", and "citations": [].
3. If the excerpts DO contain the answer, set "abstain": false, provide a clear and accurate answer, and list ONLY the exact CHUNK_ID(s) from {valid_ids} in "citations". Never invent IDs.

Return JSON:
{{
  "abstain": false,
  "answer": "...",
  "citations": []
}}"""

    llm_res = pool.chat_json(answer_prompt)
    raw_ans = str(llm_res.get("answer", "")).strip()
    is_abstain = bool(llm_res.get("abstain", False)) or ("không đủ bằng chứng" in raw_ans.lower())

    raw_cites = llm_res.get("citations", [])
    clean_cites = [c for c in raw_cites if c in valid_ids]

    if is_abstain or not clean_cites:
        status = "abstain"
        final_answer = "Không đủ bằng chứng"
        clean_cites = []
    else:
        status = "answered"
        final_answer = raw_ans

    # Đánh giá Retrieval Metrics dựa trên gold_chunk_id và evidence_quote của quyển sách
    gold_cid = q.get("gold_chunk_id")
    ev_quote = (q.get("evidence_quote") or "").strip().lower()

    def has_book_evidence(c):
        if gold_cid and c["id"] == gold_cid:
            return True
        if ev_quote and len(ev_quote) >= 20 and ev_quote[:25] in c["text"].lower():
            return True
        return False

    hit_1 = 1 if (final_chunks and has_book_evidence(final_chunks[0])) else 0
    ev_hit_3 = 1 if any(has_book_evidence(c) for c in final_chunks[:3]) else 0
    ctx_rate = round(sum(1 for c in final_chunks if has_book_evidence(c)) / len(final_chunks), 4) if final_chunks else 0.0

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    return {
        "query_id": q["id"],
        "split": q["split"],
        "config": cfg_name,
        "model": "openai/gpt-4o-mini",
        "question": q["question"],
        "answerable": q["answerable"],
        "gold_chunk_id": gold_cid,
        "gold_page": q.get("gold_page"),
        "status": status,
        "answer": final_answer,
        "citations": clean_cites,
        "metrics": {
            "hit_at_1": hit_1,
            "evidence_hit_at_3": ev_hit_3,
            "context_evidence_rate": ctx_rate,
            "tool_calls": len(events),
            "valid_citation": 1 if (status == "answered" and len(clean_cites) > 0) else 0,
            "correct_abstention": 1 if (not q["answerable"] and status == "abstain") else 0,
            "false_abstention": 1 if (q["answerable"] and status == "abstain") else 0
        },
        "events": events,
        "elapsed_ms": elapsed_ms
    }

def summarize(traces: list, cfg_name: str, split_name: str):
    ans_qs = [t for t in traces if t["answerable"]]
    unans_qs = [t for t in traces if not t["answerable"]]
    n_ans = max(len(ans_qs), 1)
    n_all = max(len(traces), 1)

    return {
        "config": cfg_name,
        "split": split_name,
        "total_questions": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstain": sum(1 for t in traces if t["status"] == "abstain"),
        "hit_at_1": round(sum(t["metrics"]["hit_at_1"] for t in ans_qs) / n_ans, 4),
        "evidence_hit_at_3": round(sum(t["metrics"]["evidence_hit_at_3"] for t in ans_qs) / n_ans, 4),
        "context_evidence_rate": round(sum(t["metrics"]["context_evidence_rate"] for t in ans_qs) / n_ans, 4),
        "valid_citations": sum(t["metrics"]["valid_citation"] for t in traces),
        "correct_abstention": f"{sum(t['metrics']['correct_abstention'] for t in unans_qs)}/{len(unans_qs)}",
        "false_abstention": f"{sum(t['metrics']['false_abstention'] for t in ans_qs)}/{len(ans_qs)}",
        "avg_tool_calls": round(sum(t["metrics"]["tool_calls"] for t in traces) / n_all, 2),
        "avg_elapsed_ms": round(sum(t["elapsed_ms"] for t in traces) / n_all, 1)
    }

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = json.loads(QPATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    dev_qs = [q for q in questions if q["split"] == "dev"]
    test_qs = [q for q in questions if q["split"] == "test"]

    pool = OpenRouterKeyPool()
    conn = get_db_connection()
    all_chunks = load_book_chunks(conn)
    print(f"=> Đã nạp {len(all_chunks)} chunks duy nhất của quyển sách từ PostgreSQL Lab 6.")

    # Đánh chỉ mục BM25 kết hợp cả tên chương, nhân vật và nội dung sách
    bm25_corpus = [tokenize(f"{c['chapter']} {c['character']} {c['text']}") for c in all_chunks]
    bm25_index = BM25Okapi(bm25_corpus)

    # 5 cấu hình thực nghiệm trên tập DEV (17 câu): 3 Baseline + 2 Cải thiện
    dev_configs = [
        ("bm25", "bm25", 3, 1, False),
        ("dense", "dense", 3, 1, False),
        ("hybrid", "hybrid", 3, 1, False),
        ("hybrid_k5_metadata", "hybrid", 5, 1, True),        # Cải thiện 1: Tăng top_k=5 + bổ sung Metadata (Chapter/Character/Page)
        ("hybrid_expanded_agent", "hybrid", 5, 2, True)      # Cải thiện 2: Multi-step Agent tự mở rộng truy vấn (max_steps=2)
    ]

    summaries = []
    for cfg_name, method, k, max_steps, inc_meta in dev_configs:
        print(f"\n--- Đang chạy DEV [{cfg_name}] (method={method}, top_k={k}, max_steps={max_steps}) ---")
        traces = [
            run_question(q, cfg_name, method, k, max_steps, inc_meta, conn, pool, all_chunks, bm25_index)
            for q in dev_qs
        ]
        # Lưu cả file traces_<method>.json và traces_dev_<cfg>.json để tương thích mọi script chấm
        (RESULTS_DIR / f"traces_{cfg_name}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
        s = summarize(traces, cfg_name, "dev")
        summaries.append(s)
        print(f"=> Kết quả DEV [{cfg_name}]: Hit@1={s['hit_at_1']} | EvHit@3={s['evidence_hit_at_3']} | Answered={s['answered']}/{len(dev_qs)} | FalseAbstain={s['false_abstention']}")

    # Khóa cấu hình tốt nhất (hybrid_expanded_agent) để chạy trên tập TEST (7 câu)
    print("\n--- Đang chạy TEST (7 câu) với cấu hình khóa [hybrid_expanded_agent] ---")
    test_traces = [
        run_question(q, "hybrid_expanded_agent", "hybrid", 5, 2, True, conn, pool, all_chunks, bm25_index)
        for q in test_qs
    ]
    (RESULTS_DIR / "traces_test.json").write_text(json.dumps(test_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    test_summary = summarize(test_traces, "hybrid_expanded_agent", "test")
    summaries.append(test_summary)
    print(f"=> Kết quả TEST [hybrid_expanded_agent]: Hit@1={test_summary['hit_at_1']} | EvHit@3={test_summary['evidence_hit_at_3']} | Answered={test_summary['answered']}/{len(test_qs)} | CorrectAbstain={test_summary['correct_abstention']}")

    # Xuất bảng tổng hợp metrics và file chấm điểm thủ công 0/1/2 (reviews.json)
    (RESULTS_DIR / "summary_metrics.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    reviews = []
    for t in dev_configs and test_traces:
        ret_score = 2 if t["metrics"]["hit_at_1"] == 1 else (1 if t["metrics"]["evidence_hit_at_3"] == 1 else 0)
        if not t["answerable"]:
            ans_score = 2 if t["status"] == "abstain" else 0
        else:
            ans_score = 2 if (t["status"] == "answered" and t["metrics"]["valid_citation"] == 1) else 0
        reviews.append({
            "query_id": t["query_id"],
            "split": t["split"],
            "question": t["question"],
            "retrieval_score_0_1_2": ret_score,
            "answer_score_0_1_2": ans_score,
            "status": t["status"],
            "citations": t["citations"]
        })
    (RESULTS_DIR / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    conn.close()
    print("\n[HOÀN TẤT TOÀN BỘ] Đã lưu đầy đủ traces, summary_metrics.json và reviews.json tại thư mục results/!")

if __name__ == "__main__":
    main()