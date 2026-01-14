import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

# Thêm đường dẫn thư viện con (Tử Vi, v.v.)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import boto3
from boto3.dynamodb.conditions import Key

# Giả định các biến môi trường đã được set trên AWS Lambda
DDB_MESSAGE_TABLE = os.environ.get("DDB_MESSAGE_TABLE", "sorcererxstreme-chatMessages")
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-micro-v1:0")

dynamodb = boto3.resource("dynamodb")
ddb_table = dynamodb.Table(DDB_MESSAGE_TABLE)
bedrock = boto3.client("bedrock-runtime")

# =========================
# I. UTILS & CALCULATION
# =========================

def get_vn_now():
    return datetime.now(timezone(timedelta(hours=7)))

def calculate_numerology(date_str: str) -> str:
    """Tính số chủ đạo đơn giản từ chuỗi DD-MM-YYYY"""
    try:
        digits = [int(d) for d in date_str if d.isdigit()]
        total = sum(digits)
        while total > 11 and total not in [22, 33]:
            total = sum(int(d) for d in str(total))
        return str(total)
    except: return "N/A"

# =========================
# II. AI AGENT CORES
# =========================

def call_bedrock(system: str, user: str, tokens=1000) -> str:
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": tokens, "temperature": 0.7},
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}]
    })
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_LLM_MODEL_ID, body=body, contentType="application/json")
        return json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
    except Exception as e: return f"Lỗi AI: {str(e)}"

def get_summarized_content(question: str, answer: str) -> str:
    """Tạo bản tóm tắt cô đọng (Keywords) để lưu vào history"""
    system = "Bạn là trợ lý tóm lược dữ liệu huyền học. Hãy tóm tắt Q&A dưới 50 từ, chỉ giữ lại từ khóa quan trọng."
    user = f"Hỏi: {question}\nĐáp: {answer}"
    return call_bedrock(system, user, tokens=150)

def load_history(session_id: str) -> str:
    """Lấy 5 lượt chat gần nhất từ DynamoDB (chỉ lấy phần summary)"""
    try:
        response = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            ScanIndexForward=False, 
            Limit=5
        )
        items = response.get("Items", [])
        # Format: "USER: [summary] | ASSISTANT: [summary]"
        history = [f"{i['role'].upper()}: {i.get('summary', i['content'])}" for i in items[::-1]]
        return "\n".join(history)
    except: return ""

def save_to_db(session_id: str, role: str, full_content: str, summary: str):
    """Lưu đồng thời bản gốc và bản tóm tắt"""
    try:
        ddb_table.put_item(Item={
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "content": full_content,
            "summary": summary,
            "expire_at": int(time.time()) + (86400 * 7) # Tự xóa sau 7 ngày
        })
    except: pass

# =========================
# III. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    # Trích xuất dữ liệu từ payload BE
    user_ctx = event.get("user_context") or {}
    partner_ctx = event.get("partner_context") or {}
    data = event.get("data") or {}
    
    session_id = data.get("sessionId")
    question = data.get("question", "")
    tarot_cards = data.get("tarot_cards", []) # BE có thể gửi list lá bài nếu có

    if not session_id or not question:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing SessionID or Question"})}

    # 1. Thu thập dữ liệu context
    user_lp = calculate_numerology(user_ctx.get("birth_date", ""))
    partner_lp = calculate_numerology(partner_ctx.get("birth_date", ""))
    
    # 2. Agentic Logic: Quyết định có dùng Tarot hay không
    has_tarot = len(tarot_cards) > 0
    tarot_instruction = f"Giải bài Tarot dựa trên: {', '.join(tarot_cards)}" if has_tarot else "KHÔNG nhắc tới Tarot hay giải bài vì người dùng không bốc bài."

    # 3. Load lịch sử tóm tắt
    history_str = load_history(session_id)

    # 4. Prompt cho AI
    system_prompt = f"""
# ROLE: SorcererXstreme - AI Huyền học chuyên nghiệp. 
# DATE: {get_vn_now().strftime('%d/%m/%Y')}

# QUY TẮC PHẢN HỒI (AGENTIC):
1. Phân tích Thần số học: User (Số {user_lp}), Partner (Số {partner_lp}).
2. {tarot_instruction}
3. Nếu câu hỏi chung chung, hãy dùng dữ liệu ngày sinh để dự báo năng lượng ngày.
4. LUÔN tóm tắt ý chính bằng KEYWORDS ở cuối câu trả lời nếu văn bản dài.

# STYLE: Huyền bí, ngắn gọn, đi thẳng vào vấn đề.
"""

    user_prompt = f"""
[LỊCH SỬ CHAT (TÓM TẮT)]:
{history_str}

[DỮ LIỆU NGƯỜI DÙNG]:
- {user_ctx.get('name')}: Số {user_lp}
- Đối phương: Số {partner_lp}

[CÂU HỎI]:
"{question}"
"""

    # 5. Gọi AI trả lời cho User
    full_reply = call_bedrock(system_prompt, user_prompt)

    # 6. Tóm tắt lượt này và lưu vào DB
    # Tóm tắt cả câu hỏi và câu trả lời để làm history cho lượt sau
    summary_for_history = get_summarized_content(question, full_reply)
    
    # Lưu vào DB (History chỉ cần 1 record tóm tắt cho lượt này là đủ)
    save_to_db(session_id, "user", question, question) # Câu hỏi lưu nguyên gốc hoặc tóm tắt tùy bạn
    save_to_db(session_id, "assistant", full_reply, summary_for_history)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({
            "sessionId": session_id,
            "reply": full_reply
        }, ensure_ascii=False)
    }