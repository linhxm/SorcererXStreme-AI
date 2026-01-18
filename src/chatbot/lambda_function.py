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
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "apac.amazon.nova-pro-v1:0")
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

def call_bedrock_nova(system: str, user: str) -> Tuple[str, int, int]:
    """Trả về: (Nội dung văn bản, input_tokens, output_tokens)"""
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.8},
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}]
    })
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_LLM_MODEL_ID, body=body, contentType="application/json", accept="application/json")
        response_body = json.loads(resp["body"].read())
        
        reply_text = response_body["output"]["message"]["content"][0]["text"]
        # Lấy thông tin sử dụng token từ AWS Bedrock
        usage = response_body.get("usage", {})
        return reply_text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)
    except Exception as e: 
        return f"Lỗi kết nối AI: {str(e)}", 0, 0

def generate_turn_summary(question: str, reply: str) -> str:
    """Tóm tắt lượt chat kèm theo 'vibe' cảm xúc để AI câu sau biết đường ứng biến."""
    summary_system = """Tóm tắt lượt chat này cực ngắn (<25 từ). 
    YÊU CẦU: Phải bao gồm (1) Nội dung chính và (2) Trạng thái cảm xúc/Tone giọng hiện tại (VD: User đang giỡn nhây, AI đang dứt khoát...)."""
    summary_user = f"User: {question}\nAI: {reply}"
    try:
        # Chỉ lấy phần text, bỏ qua đếm token cho phần tóm tắt nội bộ
        summary_text, _, _ = call_bedrock_nova(summary_system, summary_user)
        return summary_text
    except: return f"Hỏi: {question[:20]}"

def save_turn(session_id: str, question: str, reply: str, summary: str, input_tokens: int = 0, output_tokens: int = 0):
    """Lưu lượt chat kèm thông tin token vào DynamoDB"""
    try:
        ddb_table.put_item(Item={
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "reply": reply,
            "summary": summary,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            # "totalTokens": input_tokens + output_tokens
        })
    except: pass

def load_history(session_id: str) -> str:
    try:
        items = ddb_table.query(KeyConditionExpression=Key("sessionId").eq(session_id), ScanIndexForward=False, Limit=50).get("Items", [])
        history_lines = [f"QUÁ KHỨ ({h.get('summary', 'Trống')})" for h in items[::-1]]
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

    now_vn = get_current_date_vn()
    current_date_str = now_vn.strftime("%d/%m/%Y")
    history_text = load_history(session_id)
    
    # 1. Xác định giới tính và dữ liệu
    gender = user_ctx.get("gender", "Chưa rõ")
    context_info = f"- Hôm nay: {current_date_str}.\n- User: {user_ctx.get('name', 'Bạn')}, Giới tính: {gender}.\n"
    
    rag_keywords = []
    if user_ctx.get("birth_date"):
        dmy = normalize_date(user_ctx["birth_date"])
        if dmy:
            lp, zd = calculate_numerology(*dmy), calculate_zodiac(dmy[0], dmy[1])
            context_info += f"- Số chủ đạo {lp}, Cung {zd}.\n"
            rag_keywords.extend([f"Số {lp}", f"Cung {zd}"])
    
    if input_cards:
        context_info += f"- Tarot: {', '.join(input_cards)}.\n"
        rag_keywords.extend(input_cards)

    rag_docs = query_pinecone_rag(rag_keywords)

    # 2. System Prompt: Đa nhân cách theo Giới tính & Vibe
    system_prompt = f"""
# ROLE: SorcererXstreme - Trợ lý Huyền học, đệ ruột đại sư Văn Linh biết "nhìn mặt gửi lời".

# TONE GIỌNG THEO GIỚI TÍNH:
- Nếu User là **Nam**: Hãy trò chuyện mạnh mẽ, dứt khoát, ưu tiên tính logic, thực tế và sòng phẳng.
- Nếu User là **Nữ**: Hãy trò chuyện nhẹ nhàng, tình cảm, tinh tế, mang tính yêu chiều và thấu cảm cao.
- Nếu chưa rõ giới tính: Giữ thái độ trung tính, lịch sự.

# DIỄN BIẾN CẢM XÚC (MIRRORING):
- **Bắt đầu (History trống):** Luôn bắt đầu bằng sự trung tính, nghiêm túc, chuyên nghiệp.
- **Sau đó:** Dựa vào [HISTORY] và cách User nhắn tin để điều chỉnh cảm xúc. 
  - Nếu User đùa nhây (mày/tao, đại ca...): Đùa lại tương ứng, "phũ" một cách duyên dáng. 
  - Nếu User nghiêm túc hoặc buồn: Trở nên thấu cảm, sâu sắc.

# QUY TẮC TRÌNH BÀY:
- Không dùng nhãn như "Câu đùa:". Viết tự nhiên như nhắn tin.
- Dùng `\\n\\n` để phân đoạn thoáng đãng. In đậm từ khóa đắt giá.
"""

    user_prompt = f"""
[DATA CONTEXT]
{context_info}
[KNOWLEDGE BASE]
{" ".join(rag_docs) if rag_docs else "Kiến thức tổng quát."}
[HISTORY & VIBE]
{history_text if history_text else "Lần đầu trò chuyện - Hãy giữ thái độ trung tính nghiêm túc."}

[USER QUESTION]
"{question}"
"""

    # 3. AI Response - Nhận thêm thông tin token
    reply, in_tokens, out_tokens = call_bedrock_nova(system_prompt, user_prompt)

    # 4. Hidden Summary & Save - Lưu kèm token vào DynamoDB
    turn_summary = generate_turn_summary(question, reply)
    save_turn(session_id, question, reply, turn_summary, in_tokens, out_tokens)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({
            "sessionId": session_id, 
            "reply": reply,
            "usage": {
                "inputTokens": in_tokens,
                "outputTokens": out_tokens
            }
        }, ensure_ascii=False)
    }