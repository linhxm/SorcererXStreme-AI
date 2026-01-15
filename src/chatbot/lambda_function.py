import os
import sys
import json
import re
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

# --- [FIX QUAN TRỌNG] THÊM ĐƯỜNG DẪN ĐỂ TÌM THƯ VIỆN CON ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import boto3
from boto3.dynamodb.conditions import Key
from pinecone import Pinecone

# Import thư viện Tử Vi
try:
    from lasotuvi import App, DiaBan
    from lasotuvi.AmDuong import diaChi
    HAS_TUVI = True
except ImportError:
    HAS_TUVI = False
    print("WARNING: Không tìm thấy thư viện lasotuvi.")

# =========================
# I. CONFIGURATION
# =========================
DDB_MESSAGE_TABLE = os.environ.get("DDB_MESSAGE_TABLE", "sorcererxstreme-chatMessages")
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-micro-v1:0")
BEDROCK_EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

# =========================
# II. GLOBAL CLIENTS
# =========================
region = os.environ.get('BEDROCK_REGION', 'ap-southeast-1')
dynamodb = boto3.resource('dynamodb', region_name=region)
ddb_table = dynamodb.Table(DDB_MESSAGE_TABLE)
bedrock = boto3.client("bedrock-runtime")

pc_index = None
if PINECONE_API_KEY and PINECONE_HOST:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pc.Index(host=PINECONE_HOST)
    except Exception as e:
        print(f"INIT ERROR: Pinecone {e}")

# =========================
# III. CALCULATION ENGINES
# =========================

def get_current_date_vn():
    return datetime.now(timezone(timedelta(hours=7)))

def normalize_date(date_str: str) -> Optional[Tuple[int, int, int]]:
    try:
        if not date_str: return None
        if "-" in date_str:
            parts = date_str.split("-")
            return int(parts[2]), int(parts[1]), int(parts[0])
        elif "/" in date_str:
            parts = date_str.split("/")
            return int(parts[0]), int(parts[1]), int(parts[2])
    except: return None
    return None

def calculate_numerology(d: int, m: int, y: int) -> str:
    def sum_digits(n):
        s = sum(int(digit) for digit in str(n))
        if s == 11 or s == 22 or s == 33: return s
        return s if s < 10 else sum_digits(s)
    total = sum_digits(d) + sum_digits(m) + sum_digits(y)
    lp = sum_digits(total)
    if lp == 4 and total == 22: lp = 22
    return str(lp)

def calculate_zodiac(d: int, m: int) -> str:
    zodiacs = [(1, 20, "Ma Kết"), (2, 19, "Bảo Bình"), (3, 21, "Song Ngư"), (4, 20, "Bạch Dương"), (5, 21, "Kim Ngưu"), (6, 22, "Song Tử"), (7, 23, "Cự Giải"), (8, 23, "Sư Tử"), (9, 23, "Xử Nữ"), (10, 24, "Thiên Bình"), (11, 23, "Bọ Cạp"), (12, 22, "Nhân Mã")]
    for month, day, sign in zodiacs:
        if m == month: return sign if d < day else zodiacs[(zodiacs.index((month, day, sign)) + 1) % 12][2]
    return "Ma Kết"

# =========================
# IV. AI & MEMORY FUNCTIONS
# =========================

def call_bedrock_nova(system: str, user: str) -> str:
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.7},
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}]
    })
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_LLM_MODEL_ID, body=body, contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
    except Exception as e: return f"Lỗi kết nối AI: {str(e)}"

def generate_turn_summary(question: str, reply: str) -> str:
    """Tóm tắt lượt chat để làm bộ nhớ dài hạn (Lưu nội bộ)."""
    summary_system = "Bạn là trợ lý ghi nhớ. Hãy tóm tắt lượt chat này thành 1 câu cực ngắn (dưới 20 từ) chứa các keyword chính để theo dõi tiến trình hội thoại."
    summary_user = f"User: {question}\nAI: {reply}"
    try:
        return call_bedrock_nova(summary_system, summary_user)
    except: return f"Trao đổi: {question[:30]}"

def save_turn(session_id: str, question: str, reply: str, summary: str):
    """Lưu cặp câu hỏi/trả lời và tóm tắt vào một dòng duy nhất."""
    try:
        ddb_table.put_item(Item={
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "reply": reply,
            "summary": summary
        })
    except: pass

def load_history(session_id: str) -> str:
    """Load tối đa 50 lượt chat gần nhất dựa trên summary."""
    try:
        items = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id), 
            ScanIndexForward=False, 
            Limit=50
        ).get("Items", [])
        
        history_lines = []
        for h in items[::-1]:
            txt = h.get('summary') or f"Q: {h.get('question')} - A: {h.get('reply')}"
            history_lines.append(f"PAST_TURN: {txt}")
        return "\n".join(history_lines)
    except: return ""

def query_pinecone_rag(keywords: List[str]) -> List[str]:
    if not pc_index or not keywords: return []
    kw = " ".join(list(set(keywords)))
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_EMBED_MODEL_ID, body=json.dumps({"texts": [kw[:2000]], "input_type": "search_query"}), contentType="application/json")
        vector = json.loads(resp["body"].read())["embeddings"][0]
        results = pc_index.query(vector=vector, top_k=3, include_metadata=True)
        return [f"[{m['metadata'].get('entity_name', '')}]: {m['metadata'].get('context_str', m['metadata'].get('content', ''))}" for m in results.get('matches', []) if m['score'] >= 0.35]
    except: return []

# =========================
# V. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    try: body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    except: return {"statusCode": 400, "body": "Invalid JSON"}

    data = body.get("data", {})
    user_ctx = body.get("user_context") or {}
    session_id, question = data.get("sessionId"), data.get("question") or ""
    input_cards = data.get("tarot_cards", [])

    if not session_id or (not question and not input_cards):
         return {"statusCode": 400, "body": json.dumps({"error": "Missing sessionId/question"})}

    # 1. Date & History
    now_vn = get_current_date_vn()
    current_date_str = now_vn.strftime("%d/%m/%Y")
    history_text = load_history(session_id)
    
    # 2. Context Preparation
    context_info = f"- Thời điểm hiện tại: {current_date_str}.\n"
    rag_keywords = []
    if user_ctx.get("birth_date"):
        dmy = normalize_date(user_ctx["birth_date"])
        if dmy:
            lp, zd = calculate_numerology(*dmy), calculate_zodiac(dmy[0], dmy[1])
            context_info += f"- User: {user_ctx.get('name', 'Bạn')}, Số chủ đạo {lp}, Cung {zd}.\n"
            rag_keywords.extend([f"Số {lp}", f"Cung {zd}"])
    
    if input_cards:
        context_info += f"- Bài Tarot đã chọn: {', '.join(input_cards)}.\n"
        rag_keywords.extend(input_cards)

    rag_docs = query_pinecone_rag(rag_keywords)

    # 3. System Prompt: Cân đối, Dí dỏm, Khơi gợi & Formatting
    system_prompt = f"""
# ROLE: SorcererXstreme - Bậc thầy Huyền học thông thái.
# PHONG CÁCH HỘI THOẠI:
1. **ĐỘ CÂN ĐỐI:** Trả lời ngắn gọn cho câu hỏi đơn giản, sâu sắc cho các vấn đề tâm linh/cảm xúc.
2. **TÍNH DÍ DỎM:** Thỉnh thoảng thêm câu đùa duyên dáng (ngẫu nhiên, không lạm dụng).
3. **KHƠI GỢI:** Đưa ra lời gợi mở hoặc câu hỏi tương tác một cách linh hoạt khi thấy phù hợp.
4. **NGỮ CẢNH:** Luôn bám sát [HISTORY SUMMARY] để trò chuyện mạch lạc.

# QUY TẮC TRÌNH BÀY (BẮT BUỘC):
- Sử dụng **dòng trống** giữa các đoạn văn để tạo sự thông thoáng, dễ đọc.
- Sử dụng **bullet points (* hoặc -)** rõ ràng khi liệt kê các ý.
- Sử dụng **in đậm (bold)** cho các từ khóa quan trọng và *in nghiêng (italic)* để nhấn mạnh cảm xúc.
"""

    user_prompt = f"""
[DATA CONTEXT]
{context_info}
[KNOWLEDGE BASE]
{" ".join(rag_docs) if rag_docs else "Kiến thức tổng quát."}
[HISTORY SUMMARY]
{history_text}
[USER QUESTION]
"{question}"
"""

    # 4. AI Response
    reply = call_bedrock_nova(system_prompt, user_prompt)

    # 5. Save & Return
    turn_summary = generate_turn_summary(question, reply)
    save_turn(session_id, question, reply, turn_summary)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)
    }