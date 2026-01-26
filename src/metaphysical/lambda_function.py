import json
import boto3
import os
import sys
import traceback
import hashlib
from datetime import datetime, timedelta, timezone

# --- 1. THIẾT LẬP ĐƯỜNG DẪN & IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from lasotuvi.App import lapDiaBan
    from lasotuvi.DiaBan import diaBan as DiaBanClass
    from lasotuvi.ThienBan import lapThienBan
except ImportError:
    print("WARNING: Thư viện lasotuvi không khả dụng.")
    lapDiaBan = DiaBanClass = lapThienBan = None

from prompts import (
    get_tarot_prompt, 
    get_astrology_prompt, 
    get_numerology_prompt, 
    get_horoscope_prompt
)

# --- 2. CẤU HÌNH AWS & DATABASE ---
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-pro-v1:0")

KNOWLEDGE_TABLE = os.environ.get("KNOWLEDGE_TABLE", "SorcererXStreme_Metaphysical_Table")
TAROT_LOG_TABLE = os.environ.get("TAROT_LOG_TABLE", "SorcererXStreme_Tarot_Logs")
CACHE_TABLE = os.environ.get("CACHE_TABLE", "SorcererXStreme_Metaphysical_Cache")

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
dynamodb = boto3.resource("dynamodb", region_name=BEDROCK_REGION)

table_knowledge = dynamodb.Table(KNOWLEDGE_TABLE)
table_tarot_log = dynamodb.Table(TAROT_LOG_TABLE)
table_cache = dynamodb.Table(CACHE_TABLE)

# ==========================================
# 3. CORE HELPER FUNCTIONS
# ==========================================

def get_current_time_vn():
    """Lấy đối tượng datetime hiện tại theo múi giờ Việt Nam (GMT+7)"""
    return datetime.now(timezone.utc) + timedelta(hours=7)

def parse_date(date_str):
    if not date_str: return None
    s = str(date_str).replace('–', '-').replace('—', '-').replace('.', '-').replace('/', '-')
    s = s.strip()
    for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try: return datetime.strptime(s, fmt)
        except ValueError: continue
    return None

def generate_birth_id(dob, tob, gender):
    seed = f"{dob}_{tob}_{gender}"
    return hashlib.md5(seed.encode()).hexdigest()

def get_db_item(category, entity_name):
    try:
        response = table_knowledge.get_item(Key={'category': category, 'entity_name': entity_name})
        item = response.get('Item')
        if not item: return {}
        ctx = item.get('contexts', '{}')
        return json.loads(ctx) if isinstance(ctx, str) else ctx
    except: return {}

def call_bedrock_llm(prompt, temperature=0.6):
    """Gửi prompt và trả về answer cùng token input/output riêng biệt."""
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 2000, "temperature": temperature, "top_p": 0.9},
        "messages": [{"role": "user", "content": [{"text": prompt}]}]
    })
    try:
        res = bedrock.invoke_model(modelId=MODEL_ID, body=body)
        res_body = json.loads(res.get('body').read())
        answer = res_body['output']['message']['content'][0]['text']
        usage = res_body.get('usage', {})
        return answer, usage.get('inputTokens', 0), usage.get('outputTokens', 0)
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Vũ trụ đang bận hiệu chỉnh năng lượng.", 0, 0

# ==========================================
# 4. DOMAIN LOGIC
# ==========================================

# --- CHIÊM TINH (ASTROLOGY) ---
def calculate_zodiac(day, month):
    if (month == 1 and day >= 20) or (month == 2 and day <= 18): return "Bảo Bình"
    if (month == 2 and day >= 19) or (month == 3 and day <= 20): return "Song Ngư"
    if (month == 3 and day >= 21) or (month == 4 and day <= 19): return "Bạch Dương"
    if (month == 4 and day >= 20) or (month == 5 and day <= 20): return "Kim Ngưu"
    if (month == 5 and day >= 21) or (month == 6 and day <= 21): return "Song Tử"
    if (month == 6 and day >= 22) or (month == 7 and day <= 22): return "Cự Giải"
    if (month == 7 and day >= 23) or (month == 8 and day <= 22): return "Sư Tử"
    if (month == 8 and day >= 23) or (month == 9 and day <= 22): return "Xử Nữ"
    if (month == 9 and day >= 23) or (month == 10 and day <= 23): return "Thiên Bình"
    if (month == 10 and day >= 24) or (month == 11 and day <= 22): return "Thiên Yết"
    if (month == 11 and day >= 23) or (month == 12 and day <= 21): return "Nhân Mã"
    return "Ma Kết"

def format_zodiac_context(zodiac_name, context_json):
    """KHÔI PHỤC: Hàm tạo text context chi tiết cho Chiêm tinh."""
    if not context_json: return f"Không có dữ liệu cho {zodiac_name}."
    return f"""
    - Cung: {zodiac_name}
    - Tính cách: {context_json.get('tinh-cach', '')}
    - Tình yêu: {context_json.get('tinh-yeu', '')}
    - Điểm mạnh: {context_json.get('diem-manh', '')}
    - Điểm yếu: {context_json.get('diem-yeu', '')}
    - Cung hợp: {context_json.get('cung-hop', '')}
    """

def handle_astrology(body):
    u_ctx = body.get('user_context', {})
    ft = body.get('feature_type', 'overview')
    bid = generate_birth_id(u_ctx.get('birth_date'), '', u_ctx.get('gender',''))
    fid = f"astro_{ft}"

    try:
        cached = table_cache.get_item(Key={"birth_id": bid, "feature_id": fid})
        if "Item" in cached: return cached["Item"]["answer"]
    except: pass

    u_date = parse_date(u_ctx.get('birth_date'))
    if not u_date: return "Ngày sinh không hợp lệ."
    uz = calculate_zodiac(u_date.day, u_date.month)
    uz_data = get_db_item('cung-hoang-dao', uz)

    if ft == 'overview':
        context_str = format_zodiac_context(uz, uz_data)
        prompt = get_astrology_prompt('overview', uz, u_ctx.get('birth_date'), context_str, f"Phân tích {uz}", u_ctx.get('gender'))
        ans, in_t, out_t = call_bedrock_llm(prompt, 0.5)
    else:
        p_ctx = body.get('partner_context', {})
        p_date = parse_date(p_ctx.get('birth_date'))
        pz = calculate_zodiac(p_date.day, p_date.month)
        pz_data = get_db_item('cung-hoang-dao', pz)
        
        # Logic đánh giá độ hợp từ bản cũ
        match_status = "CẦN CỐ GẮNG"
        if pz in uz_data.get('cung-hop', '') or uz in pz_data.get('cung-hop', ''): match_status = "KHÁ HỢP"
        if pz in uz_data.get('cung-hop', '') and uz in pz_data.get('cung-hop', ''): match_status = "RẤT HỢP"

        comb_ctx = f"USER: {uz}\n{format_zodiac_context(uz, uz_data)}\nPARTNER: {pz}\n{format_zodiac_context(pz, pz_data)}\nKẾT LUẬN: {match_status}"
        prompt = get_astrology_prompt('love', f"{uz}&{pz}", f"{u_ctx.get('birth_date')}", comb_ctx, "Độ hợp", u_ctx.get('gender'))
        ans, in_t, out_t = call_bedrock_llm(prompt, 0.6)

    table_cache.put_item(Item={
        "birth_id": bid, "feature_id": fid, "answer": ans,
        "input_tokens": in_t, "output_tokens": out_t, "ts": datetime.utcnow().isoformat()
    })
    return ans

# --- THẦN SỐ HỌC (NUMEROLOGY) ---
def calculate_life_path(day, month, year):
    """KHÔI PHỤC: Logic Master Numbers 11, 22, 33."""
    full_str = f"{day}{month}{year}"
    total = sum(int(digit) for digit in full_str)
    while total > 9:
        if total in [11, 22, 33, 10]: break
        total = sum(int(digit) for digit in str(total))
    return str(total)

def handle_numerology(body):
    u_ctx = body.get('user_context', {})
    u_date = parse_date(u_ctx.get('birth_date'))
    if not u_date: return "Ngày sinh lỗi."
    
    bid = generate_birth_id(u_ctx.get('birth_date'), '', '')
    fid = "num_path"
    try:
        cached = table_cache.get_item(Key={"birth_id": bid, "feature_id": fid})
        if "Item" in cached: return cached["Item"]["answer"]
    except: pass

    lp = calculate_life_path(u_date.day, u_date.month, u_date.year)
    ctx_data = get_db_item('numerology_number', f"Số {lp}")
    
    # KHÔI PHỤC: Context chi tiết từ bản cũ
    context_str = f"""
    - Số chủ đạo: {lp}
    - Tổng quan: {ctx_data.get('tong-quan', '')}
    - Ưu điểm: {ctx_data.get('uu-diem', '')}
    - Nhược điểm: {ctx_data.get('nhuoc-diem', '')}
    - Sứ mệnh: {ctx_data.get('chi-so-su-menh', '')}
    - Công việc: {ctx_data.get('so-hop-cong-viec', '')}
    """
    
    prompt = get_numerology_prompt(lp, u_ctx.get('birth_date'), context_str, f"Phân tích số {lp}", u_ctx.get('gender'))
    ans, in_t, out_t = call_bedrock_llm(prompt, 0.5)
    
    table_cache.put_item(Item={
        "birth_id": bid, "feature_id": fid, "answer": ans,
        "input_tokens": in_t, "output_tokens": out_t, "ts": datetime.utcnow().isoformat()
    })
    return ans

# --- TAROT ---
def handle_tarot(body):
    # 1. Lấy dữ liệu đầu vào
    feature_type = body.get('feature_type', 'question')
    data = body.get('data', {})
    cards_input = data.get('cards_drawn', [])
    user_context = body.get('user_context', {})
    user_query = data.get('question', '')
    
    if not cards_input:
        return "Vui lòng chọn lá bài."

    vn_now = get_current_time_vn()
    time_str = vn_now.strftime("%H:%M:%S ngày %d/%m/%Y")

    # 2. Nhận diện chủ đề (Intent Topic) - Đã mở rộng từ khóa
    intent_topic = "general"
    if user_query:
        q_low = user_query.lower()
        if any(k in q_low for k in ['yêu', 'tình', 'crush', 'cưới', 'hẹn hò', 'người yêu']): 
            intent_topic = "love"
        elif any(k in q_low for k in ['việc', 'làm', 'nghề', 'lương', 'công ty', 'sự nghiệp']): 
            intent_topic = "work"
        elif any(k in q_low for k in ['khoẻ', 'bệnh', 'thuốc', 'sức khoẻ']): 
            intent_topic = "health"
        elif any(k in q_low for k in ['bạn', 'gia đình', 'quan hệ', 'đồng nghiệp']): 
            intent_topic = "relationship"

    # 3. Ánh xạ vị trí lá bài
    pos_map = {
        "past": "Quá khứ / Nguyên nhân", 
        "present": "Hiện tại / Diễn biến", 
        "future": "Tương lai / Kết quả"
    }
    
    context_parts = [
        f"THỜI GIAN HIỆN TẠI (GMT+7): {time_str}",
        f"Chủ đề: {intent_topic.upper()}", 
        f"Câu hỏi: {user_query}"
    ]
    
    # 4. Xử lý logic lá bài và RAG (Cập nhật cơ chế Backup)
    for card in cards_input:
        # Chuẩn hóa tên lá bài (ví dụ: "the fool" -> "The Fool")
        raw_name = card.get('card_name', '').strip()
        name = raw_name.title() 
        
        is_up = card.get('is_upright', True)
        pos = card.get('position')
        
        # Lấy dữ liệu từ DynamoDB
        card_full_data = get_db_item('tarot_card', name)
        
        suffix = "upright" if is_up else "reversed"
        
        # Chiến thuật lấy nội dung: Ưu tiên chủ đề cụ thể -> Dự phòng chủ đề chung -> Mặc định
        meaning = (card_full_data.get(f"{intent_topic}_{suffix}") or 
                   card_full_data.get(f"general_{suffix}") or 
                   "Không có dữ liệu chi tiết cho lá bài này.")
        
        orientation = "Xuôi" if is_up else "Ngược"
        pos_label = f"[{pos_map.get(pos, 'Vị trí')}]"
        
        context_parts.append(f"- {pos_label} {name} ({orientation}): {meaning}")
        
    # 5. Gọi AI và Log kết quả (Giữ nguyên phần đếm token của bản cũ)
    prompt = get_tarot_prompt(feature_type, "\n".join(context_parts), user_query, user_context, intent_topic)
    ans, in_t, out_t = call_bedrock_llm(prompt, 0.7)

    # Log vào DynamoDB (Giữ nguyên logic của bản cũ)
    try:
        table_tarot_log.put_item(Item={
            "userId": data.get("userId", "anon"), 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": user_query, 
            "answer": ans, 
            "input_tokens": in_t, 
            "output_tokens": out_t, 
            "domain": "tarot"
        })
    except Exception as e:
        print(f"Log Error: {e}")
        
    return ans

# --- TỬ VI (HOROSCOPE) ---
def parse_time_to_chi(time_str):
    """KHÔI PHỤC: Chuyển giờ sinh sang số thứ tự Chi."""
    if not time_str: return 12
    try: hour = int(str(time_str).split(':')[0])
    except: return 1
    return 1 if (hour >= 23 or hour < 1) else (hour + 1) // 2 + 1

def extract_tuvi_metadata(thien_ban, dia_ban):
    """KHÔI PHỤC: Lấy metadata để hiển thị summary."""
    try:
        return {
            "can_chi": f"{thien_ban.canNamTen} {thien_ban.chiNamTen}",
            "ban_menh": thien_ban.banMenh,
            "cuc": thien_ban.tenCuc,
            "menh_chu": thien_ban.menhChu,
            "vi_tri_menh": f"Cung {dia_ban.thapNhiCung[dia_ban.cungMenh].cungTen}"
        }
    except: return {}

def handle_horoscope(body):
    u = body.get('user_context', {})
    dob = parse_date(u.get('birth_date'))
    if not dob or lapDiaBan is None: return "Hệ thống Tử Vi chưa sẵn sàng."

    bid = generate_birth_id(u.get('birth_date'), u.get('birth_time',''), u.get('gender',''))
    fid = "horo_chart"
    try:
        cached = table_cache.get_item(Key={"birth_id": bid, "feature_id": fid})
        if "Item" in cached: return json.loads(cached["Item"]["answer"])
    except: pass

    chi_gio = parse_time_to_chi(u.get('birth_time', '12:00'))
    gender_val = 1 if str(u.get('gender')).lower() in ['male', 'nam', '1'] else -1

    db = lapDiaBan(DiaBanClass, dob.day, dob.month, dob.year, chi_gio, gender_val, True, 7)
    tb = lapThienBan(dob.day, dob.month, dob.year, chi_gio, gender_val, u.get('name','Đương số'), db, True, 7)
    
    # KHÔI PHỤC: Logic tạo context 12 cung chi tiết
    lines = [f"Đương số: {tb.ten}, Mệnh: {tb.banMenh}, Cục: {tb.tenCuc}"]
    for i in range(1, 13):
        c = db.thapNhiCung[i]
        sao_chinh = [s['saoTen'] for s in c.cungSao if s.get('saoLoai') == 1]
        lines.append(f"Cung {getattr(c, 'cungChu', '')} tại {c.cungTen}: {', '.join(sao_chinh)}")
    
    prompt = get_horoscope_prompt("\n".join(lines), u)
    ans, in_t, out_t = call_bedrock_llm(prompt, 0.7)
    
    res = {"summary": extract_tuvi_metadata(tb, db), "analysis": ans}
    table_cache.put_item(Item={
        "birth_id": bid, "feature_id": fid, "answer": json.dumps(res, ensure_ascii=False),
        "input_tokens": in_t, "output_tokens": out_t, "ts": datetime.utcnow().isoformat()
    })
    return res

# ==========================================
# 5. LAMBDA HANDLER
# ==========================================

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str): body = json.loads(body)
        domain = body.get('domain', '').lower()
        
        if domain == 'tarot': ans = handle_tarot(body)
        elif domain == 'astrology': ans = handle_astrology(body)
        elif domain == 'numerology': ans = handle_numerology(body)
        elif domain == 'horoscope': ans = handle_horoscope(body)
        else: return {'statusCode': 400, 'body': 'Invalid domain'}
            
        return {
            'statusCode': 200, 
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'domain': domain, 'answer': ans}, ensure_ascii=False)
        }
    except Exception as e:
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}