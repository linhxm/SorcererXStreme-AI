import os
import sys
import json
import re
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
dynamodb = boto3.resource("dynamodb")
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
    except:
        return None
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
    zodiacs = [
        (1, 20, "Ma Kết"), (2, 19, "Bảo Bình"), (3, 21, "Song Ngư"),
        (4, 20, "Bạch Dương"), (5, 21, "Kim Ngưu"), (6, 22, "Song Tử"),
        (7, 23, "Cự Giải"), (8, 23, "Sư Tử"), (9, 23, "Xử Nữ"),
        (10, 24, "Thiên Bình"), (11, 23, "Bọ Cạp"), (12, 22, "Nhân Mã")
    ]
    for month, day, sign in zodiacs:
        if m == month:
            return sign if d < day else zodiacs[(zodiacs.index((month, day, sign)) + 1) % 12][2]
    return "Ma Kết"

def calculate_tuvi(d: int, m: int, y: int, h_str: str, gender: int) -> dict:
    if not HAS_TUVI or not h_str: return {}
    try:
        hour_val = int(h_str.split(":")[0])
        gio_chi = int((hour_val + 1) / 2) % 12
        if gio_chi == 0: gio_chi = 12
        
        db = App.lapDiaBan(DiaBan.diaBan, d, m, y, gio_chi, gender, True, 7)
        cung_menh = db.thapNhiCung[db.cungMenh]
        chinh_tinh = [s['saoTen'] for s in cung_menh.cungSao if s['saoLoai'] == 1]
        
        return {
            "menh_tai": diaChi[cung_menh.cungSo]['tenChi'],
            "chinh_tinh": ", ".join(chinh_tinh) if chinh_tinh else "Vô Chính Diệu"
        }
    except: return {}

# =========================
# IV. HELPER FUNCTIONS
# =========================

def analyze_intent_and_extract(question: str, input_tarot: List[str]) -> dict:
    intent = {
        "explicit_date": None,
        "has_tarot": False,
        "tarot_cards": list(input_tarot) if input_tarot else [],
        "needs_llm": True
    }
    
    # 1. Detect Explicit Date
    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', question)
    if date_match:
        d, m, y = map(int, [date_match.group(1), date_match.group(2), date_match.group(3)])
        intent["explicit_date"] = (d, m, y)

    # 2. Detect Tarot
    if not intent["tarot_cards"]:
        tarot_keywords = ["Fool", "Magician", "Empress", "Emperor", "Lover", "Chariot", "Strength", "Hermit", "Wheel", "Justice", "Hanged", "Death", "Temperance", "Devil", "Tower", "Star", "Moon", "Sun", "Judgement", "World", "Cup", "Wand", "Sword", "Pentacle"]
        found = [w for w in tarot_keywords if w.lower() in question.lower()]
        if found: intent["tarot_cards"] = found
    
    if intent["tarot_cards"]: intent["has_tarot"] = True

    # 3. Detect Chit-chat
    chit_chat_phrases = ["hi", "hello", "alo", "xin chào", "chào", "chào bạn", "hola", "bắt đầu", "start"]
    cleaned_q = question.strip().lower()
    if cleaned_q in chit_chat_phrases:
        intent["needs_llm"] = False

    return intent

def process_subject_data(intent: dict, user_ctx: dict, partner_ctx: dict) -> dict:
    result = {"rag_keywords": [], "prompt_context": "", "user_calculated": {}, "partner_calculated": {}}

    # Explicit Date
    if intent["explicit_date"]:
        d, m, y = intent["explicit_date"]
        lp = calculate_numerology(d, m, y)
        zd = calculate_zodiac(d, m)
        result["prompt_context"] += f"- [THÔNG TIN ĐƯỢC HỎI - NGÀY {d}/{m}/{y}]: Số chủ đạo {lp}, Cung {zd}.\n"
        result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])

    # Tarot
    if intent["has_tarot"]:
        cards_str = ", ".join(intent["tarot_cards"])
        result["prompt_context"] += f"- [BÀI TAROT]: {cards_str}.\n"
        result["rag_keywords"].extend(intent["tarot_cards"])

    # User Info (Check None trước khi get)
    if user_ctx and user_ctx.get("birth_date"):
        dmy = normalize_date(user_ctx.get("birth_date"))
        if dmy:
            d, m, y = dmy
            lp = calculate_numerology(d, m, y)
            zd = calculate_zodiac(d, m)
            tv = {}
            if user_ctx.get("birth_time"):
                gender = 1 if user_ctx.get("gender") == "Nam" else -1
                tv = calculate_tuvi(d, m, y, user_ctx["birth_time"], gender)
            
            tv_str = f", Mệnh {tv.get('menh_tai')}" if tv else ""
            # Thêm thông tin raw để LLM biết nếu user hỏi "Tôi là ai"
            result["prompt_context"] += f"- [USER DATA - {user_ctx.get('name', 'Bạn')}]: Sinh ngày {user_ctx.get('birth_date')}. Số chủ đạo {lp}, Cung {zd}{tv_str}.\n"
            
            if not intent["explicit_date"] and not intent["has_tarot"]:
                result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])
                if tv: result["rag_keywords"].append(f"Sao {tv.get('chinh_tinh', '')}")

    # Partner Info (Check None trước khi get)
    if partner_ctx and partner_ctx.get("birth_date"):
        dmy = normalize_date(partner_ctx.get("birth_date"))
        if dmy:
            d, m, y = dmy
            lp = calculate_numerology(d, m, y)
            zd = calculate_zodiac(d, m)
            result["prompt_context"] += f"- [PARTNER DATA - Người ấy]: Số chủ đạo {lp}, Cung {zd}.\n"
            
            if not intent["explicit_date"] and not intent["has_tarot"]:
                result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])

    return result

def embed_query(text: str) -> List[float]:
    if not text: return []
    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps({"texts": [text[:2000]], "input_type": "search_query"}),
            contentType="application/json", accept="*/*"
        )
        return json.loads(resp["body"].read())["embeddings"][0]
    except: return []

def query_pinecone_rag(keywords: List[str]) -> List[str]:
    if not pc_index or not keywords: return []
    
    unique_kw = list(set(keywords))
    search_text = " ".join(unique_kw)
    vector = embed_query(search_text)
    if not vector: return []
    try:
        results = pc_index.query(vector=vector, top_k=3, include_metadata=True)
        docs = []
        for match in results.get('matches', []):
            if match['score'] < 0.35: continue
            md = match.get('metadata', {})
            content = md.get('context_str') or md.get('content') or ""
            entity = md.get('entity_name') or ""
            docs.append(f"[{entity}]: {content}")
        return docs
    except: return []

def append_message(session_id: str, role: str, content: str):
    try:
        ddb_table.put_item(Item={
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "content": content,
        })
    except: pass

def load_history(session_id: str) -> str:
    try:
        items = ddb_table.query(KeyConditionExpression=Key("sessionId").eq(session_id), ScanIndexForward=False, Limit=5).get("Items", [])
        return "\n".join([f"{h['role'].upper()}: {h['content']}" for h in items[::-1]])
    except: return ""

def call_bedrock_nova(system: str, user: str) -> str:
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.6},
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}]
    })
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_LLM_MODEL_ID, body=body, contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}"

# =========================
# VI. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    except:
        return {"statusCode": 400, "body": "Invalid JSON Body"}

    data = body.get("data", {})
    
    # [FIX: Xử lý Null Safety cho Input]
    # Dùng 'or {}' để đảm bảo nếu giá trị là None (null trong JSON) thì sẽ thành dict rỗng
    user_ctx = body.get("user_context") or {}
    partner_ctx = body.get("partner_context") or {}
    
    session_id = data.get("sessionId")
    question = data.get("question")
    input_cards = data.get("tarot_cards", [])

    if not session_id or (not question and not input_cards):
         return {"statusCode": 400, "body": json.dumps({"error": "Missing sessionId or question"})}

    # 1. Analyze Intent
    intent = analyze_intent_and_extract(question or "", input_cards) 
    
    # 2. Chit-chat Return
    if not intent["needs_llm"]:
        reply = "Chào bạn, tôi là trợ lý huyền học. Tôi có thể giúp bạn giải bài Tarot, xem Tử vi hoặc Thần số học."
        append_message(session_id, "assistant", reply)
        return {"statusCode": 200, "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)}

    # 3. Calculate Data (Giờ đã an toàn với input null)
    processed_data = process_subject_data(intent, user_ctx, partner_ctx)
    
    # 4. RAG
    rag_docs = query_pinecone_rag(processed_data["rag_keywords"])
    
    # Prompt
    current_date = get_current_date_vn().strftime("%d/%m/%Y")
    rag_text = "\n".join(rag_docs) if rag_docs else ""
    history_text = load_history(session_id)
    
    system_prompt = f"""
# ROLE: AI Huyền Học (SorcererXstreme). Hôm nay: {current_date}.

# STRICT RULES (BẮT BUỘC):
1. **PRIVACY & IDENTITY:**
   - Mặc định KHÔNG tự ý nhắc lại ngày sinh/nơi sinh của User/Partner.
   - **NGOẠI LỆ:** Nếu User hỏi về bản thân (VD: "Tôi là ai?", "Tôi sinh năm mấy?"), hãy dùng dữ liệu trong [CALCULATED CONTEXT] để trả lời.
   - Luôn gọi Partner là "Người ấy" hoặc "Đối phương".

2. **LOGIC TRẢ LỜI:**
   - **Ưu tiên RAG:** Nếu có thông tin tra cứu, dùng nó.
   - **Fallback:** Nếu RAG rỗng, hãy dùng kiến thức tổng quát và [CALCULATED CONTEXT] để trả lời. ĐỪNG nói "Tôi không có thông tin".
   - Nếu có [BÀI TAROT]: Chỉ giải bài, không bịa thêm lá khác.
   - Nếu hỏi Tương Hợp: Tổng hợp thành 1-2 câu súc tích.

3. **TONE:** Ngắn gọn, huyền bí, hữu ích.
"""

    user_prompt = f"""
[CALCULATED CONTEXT]
{processed_data['prompt_context']}

[KNOWLEDGE BASE (RAG)]
{rag_text}

[HISTORY]
{history_text}

[USER QUESTION]
"{question}"
"""

    # 5. Call AI
    reply = call_bedrock_nova(system_prompt, user_prompt)

    append_message(session_id, "user", question)
    append_message(session_id, "assistant", reply)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)
    }