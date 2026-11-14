# Báo cáo Day 7 – Agentic RAG

## Sinh viên
- **MSSV:** 2A202603467
- **Họ và tên:** Khổng Tiến Dương

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Dữ liệu kế thừa từ Day 6 (bao gồm Fifty Famous People, OpenStax Physics, và tài liệu nội bộ về biến đổi khí hậu). Kho dữ liệu gồm 124 chunks đã được phân tách và xác thực quyền sử dụng.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Cơ sở dữ liệu: PostgreSQL với extension `pgvector` (chạy cục bộ qua Docker).
  - Embedding model: `openai/text-embedding-3-small` (gọi thông qua OpenRouter API).
  - Số chiều (dimensions): 1536 chiều.
- **Agent model, cách chạy và phân chia dev/test:**
  - LLM sử dụng: Luân phiên (Load-balancing) giữa `openai/gpt-4o-mini` (qua OpenRouter) và `gemini-1.5-flash` để tránh lỗi giới hạn tốc độ (Rate Limit).
  - Benchmark: Tổng cộng 24 câu hỏi. Trong đó, tập Dev gồm 17 câu (15 câu có đáp án, 2 câu thiếu bằng chứng); tập Test gồm 7 câu (5 câu có đáp án, 2 câu thiếu bằng chứng).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** Việc đánh giá được thực hiện qua System Prompt khắt khe: Nếu context không chứa đủ bằng chứng, LLM buộc phải trả về chính xác cụm từ "Không đủ bằng chứng". Trạng thái được ghi nhận là `answered` nếu có câu trả lời và trích dẫn hợp lệ; ghi nhận là `abstain` nếu từ chối.

---

## 2. Retrieval và thử nghiệm

Ba phương pháp truy xuất dữ liệu (retrieval) được cấu hình và chạy trên toàn bộ 24 câu hỏi (top_k=3). Kết quả thống kê thời gian và trạng thái trả lời (Answered / Abstain) được ghi nhận trực tiếp từ hệ thống log thực tế.

| Cấu hình | Phương pháp, top-k | Thời gian TB / câu | Đã trả lời (Answered) | Tỷ lệ Từ chối (Abstain) | Số câu có Citation |
|---|---|---|---|---|---|
| **BM25** | Lexical Search (rank-bm25), top_k=3 | 2304.7 ms | 0/24 | 24/24 (100%) | 1 |
| **Dense** | Semantic Search (pgvector), top_k=3 | 3151.7 ms | 1/24 | 23/24 (95.8%) | 5 |
| **Hybrid** | Trộn RRF (Dense + BM25), top_k=3 | 2939.5 ms | 0/24 | 24/24 (100%) | 3 |
| **Baseline** | Chạy kiểm tra nền tảng | 3360.5 ms | 1/24 | 23/24 (95.8%) | 2 |

**Phân tích lỗi (dev) và Bài học:**
1. **Hiện tượng "False Abstention" quá cao:** Mặc dù số câu có đáp án chiếm đa số, nhưng hệ thống từ chối trả lời gần như 100%. 
   - *Nguyên nhân:* Giới hạn `top_k=3` là chưa đủ để gom trọn vẹn ngữ nghĩa của những câu hỏi phức tạp. Kết hợp với việc System Prompt được thiết kế cực kỳ nghiêm ngặt nhằm chống sinh ảo (Hallucination), LLM ưu tiên lựa chọn an toàn (Abstain) thay vì cố gắng suy luận khi thiếu ngữ cảnh.
   - *Giải pháp đề xuất:* Ở vòng tinh chỉnh tiếp theo, cần tăng context budget (ví dụ: `top_k=5` hoặc `top_k=7`) và nới lỏng nhẹ Prompt để LLM dám thực hiện liên kết thông tin (Multi-hop reasoning).
2. **Đánh đổi về thời gian và hiệu quả:** Phương pháp BM25 chạy nhanh nhất (chỉ tốn ~2.3s) nhưng không trả lời được câu nào (0/24). Phương pháp Dense mất nhiều thời gian nhất (~3.15s) do độ trễ khi so khớp vector, nhưng lại mang về lượng trích dẫn (citation) cao nhất (5 citation).
   - *Quyết định chọn cấu hình:* **Cấu hình Dense** được chọn làm giải pháp chính thức cho hệ thống. Mặc dù chậm hơn BM25 khoảng 800ms, Dense bắt ngữ nghĩa chính xác hơn và thu thập được lượng bằng chứng tốt nhất.

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng và xử lý lỗi:** 
  - Hệ thống gọi tool truy xuất 1 lượt duy nhất cho mỗi câu hỏi.
  - Điểm nhấn về kỹ thuật là cơ chế **Dynamic Pool Manager**: Sử dụng hàng đợi (`queue.Queue`) để quản lý và luân phiên các API Key (OpenRouter và Gemini). Hệ thống tự động kiểm tra trạng thái sống/chết (Alive/Dead) của key lúc khởi động, giúp ngăn chặn hoàn toàn lỗi sập do hết hạn mức API hoặc quá tải request.

| Cấu hình | Tool use | Số lượt gọi | Citation hợp lệ | Abstention đúng | Từ chối nhầm (False Abstention) |
|---|---|---|---|---|---|
| **Dense** | 24/24 | 1 lượt/câu | 5 | Cao | Rất cao (Do LLM nhút nhát) |
| **BM25** | 24/24 | 1 lượt/câu | 1 | Cao | Rất cao |
| **Hybrid** | 24/24 | 1 lượt/câu | 3 | Cao | Rất cao |

**Kết luận, Giới hạn và Hướng đi:**
- Dữ liệu log minh chứng tại `results/traces_dense.json` cho thấy việc bảo vệ hệ thống khỏi sinh ảo đang hoạt động quá hiệu quả, dẫn đến tác dụng phụ là model bị giới hạn khả năng suy luận.
- **Hướng phát triển:** Áp dụng Multi-hop retrieval để xử lý các truy vấn cần ghép mảnh thông tin từ nhiều trang khác nhau, và áp dụng cơ chế tự động điều chỉnh `top_k` dựa trên độ phức tạp của câu hỏi.

---

## 4. Thử thách nâng cao (Tùy chọn)

- **Corpus 100 trang:** Do thời gian hạn hẹp, tiến độ tập trung toàn lực vào việc xây dựng cơ chế Load-balancing (luân phiên LLM) và tích hợp hoàn chỉnh PostgreSQL (pgvector). Phần index tài liệu nâng cao chưa được triển khai.

---

### Tick list nộp bài (SUBMISSION.md)

- [x] Code + 3 retrieval methods (BM25, Dense, Hybrid).  
- [x] Embedding model + PostgreSQL vector store info.  
- [x] Benchmark ≥ 24 questions với dev/test split.  
- [x] Experiment log (`traces.json`) với đầy đủ context, status, metrics.  
- [x] Báo cáo chi tiết theo cấu trúc yêu cầu.  
- [x] Đã thiết lập `.gitignore`, hoàn toàn **không** chứa file `.env`, API key, hoặc dữ liệu nhạy cảm.

**Repo cá nhân:** `K5-L3A-DAY07-2A202603467-KhongTienDuong-RetrievalContext`  
**Link nộp:** https://github.com/khongtienduongcv-byte/K5-L3A-Day07-2A202603467-KhongTienDuong-RetrievalContext.git