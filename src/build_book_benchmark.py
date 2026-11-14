#!/usr/bin/env python3
import os
import json
import pathlib
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE = pathlib.Path(__file__).parent.parent
QPATH = BASE / "benchmark" / "questions_day7.json"

def get_openrouter_client():
    for k, v in sorted(os.environ.items()):
        if k.startswith("OPENROUTER_KEY_") and v:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=v)
                client.models.list()
                return client
            except Exception:
                continue
    raise RuntimeError("Không tìm thấy OPENROUTER_KEY_x nào hoạt động trong .env!")

def main():
    print("1. Đang đọc dữ liệu quyển sách từ bảng chunks_table (DB Lab 6)...")
    conn = psycopg2.connect(
        host="127.0.0.1", port="55433",
        user="day6", password="day6_local_only", dbname="day6_lab"
    )
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, source, page, character, chapter
            FROM chunks_table
            WHERE LENGTH(text) > 200
            ORDER BY page ASC, chunk_id ASC
            LIMIT 25;
        """)
        book_rows = cur.fetchall()
    conn.close()

    if len(book_rows) < 20:
        raise RuntimeError(f"Trong DB chỉ tìm thấy {len(book_rows)} chunks dài > 200 ký tự. Hãy kiểm tra lại container Docker Lab 6!")

    print(f"=> Đã lấy được {len(book_rows)} đoạn trích từ quyển sách (Nguồn: {book_rows[0][2]}).")
    client = get_openrouter_client()

    # Lấy 20 đoạn trích khác nhau trong sách (15 cho dev, 5 cho test)
    selected_book_chunks = book_rows[:20]
    dev_questions = []
    test_questions = []

    for idx, (cid, text, source, page, character, chapter) in enumerate(selected_book_chunks, 1):
        split = "dev" if idx <= 15 else "test"
        meta_info = f"Character: {character or 'N/A'} | Chapter: {chapter or 'N/A'} | Page: {page}"
        prompt = f"""You are building a reading comprehension benchmark for a book stored in a Vector DB.
Book Metadata: {meta_info}
Book Excerpt (Chunk ID: {cid}):
\"\"\"{text}\"\"\"

Task:
Write 1 specific, unambiguous question about an event, character action, or detail explicitly mentioned in this book excerpt.
- Write the question in the SAME language as the book excerpt so both keyword search (BM25) and semantic search (Dense) can match terms directly from the book.
- Provide a concise "reference_answer" based strictly on the excerpt.
- Provide an exact verbatim "evidence_quote" (1-2 sentences copied directly from the excerpt).

Return ONLY valid JSON:
{{
  "question": "...",
  "reference_answer": "...",
  "evidence_quote": "..."
}}"""

        res = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(res.choices[0].message.content)
        q_obj = {
            "id": "TEMP",
            "split": split,
            "question": data["question"],
            "answerable": True,
            "gold_chunk_id": cid,
            "gold_source_id": source,
            "gold_page": page,
            "character": character,
            "chapter": chapter,
            "reference_answer": data["reference_answer"],
            "evidence_quote": data["evidence_quote"]
        }
        if split == "dev":
            dev_questions.append(q_obj)
        else:
            test_questions.append(q_obj)
        print(f"  [+] Đã tạo câu hỏi sách #{idx} ({split}) từ chunk [{cid}] (Trang {page})")

    # Thêm 4 câu hỏi KHÔNG CÓ TRONG SÁCH (answerable = False) nhưng vẫn hỏi xoay quanh quyển sách
    unanswerable_book_questions = [
        ("dev", "In the book Fifty Famous People, what smartphone model and mobile carrier did George Washington use to contact his generals?"),
        ("dev", "According to the book's chapters on Abraham Lincoln, which programming language did he use to write the Emancipation Proclamation?"),
        ("test", "In the book Fifty Famous People, what was the exact Bitcoin price recorded on the day King Alfred burned the cakes?"),
        ("test", "According to the book, which Formula 1 racing team did Paul Revere drive for instead of riding a horse?")
    ]

    for split, uq in unanswerable_book_questions:
        q_obj = {
            "id": "TEMP",
            "split": split,
            "question": uq,
            "answerable": False,
            "gold_chunk_id": None,
            "gold_source_id": None,
            "gold_page": None,
            "character": None,
            "chapter": None,
            "reference_answer": "Không đủ bằng chứng",
            "evidence_quote": ""
        }
        if split == "dev":
            dev_questions.append(q_obj)
        else:
            test_questions.append(q_obj)

    # Đánh lại mã Q001 -> Q017 (dev) và Q018 -> Q024 (test)
    all_questions = dev_questions + test_questions
    for i, q in enumerate(all_questions, 1):
        q["id"] = f"Q{i:03d}"

    QPATH.parent.mkdir(exist_ok=True)
    output_payload = {
        "manifest_version": 2,
        "corpus": "Fifty Famous People (Day 6 PostgreSQL Book Corpus)",
        "total_questions": len(all_questions),
        "dev_count": len(dev_questions),
        "test_count": len(test_questions),
        "questions": all_questions
    }
    QPATH.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[HOÀN TẤT] Đã lưu 24 câu hỏi bám sát quyển sách vào: {QPATH}")

if __name__ == "__main__":
    main()