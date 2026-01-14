import json
import boto3
import os
import sys
import traceback
from datetime import datetime

# Import thư viện Tử Vi (Giả định đã có trong Layer hoặc package)
# Nếu chạy local mà không có folder này sẽ lỗi, nhưng trong môi trường Test chúng ta sẽ Mock nó hoặc chấp nhận lỗi import nếu không test sâu vào hàm library.
try:
    from lasotuvi.App import lapDiaBan
    from lasotuvi.DiaBan import diaBan as DiaBanClass
    from lasotuvi.ThienBan import lapThienBan
except ImportError:
    # Fallback giả định để code không crash ngay khi import nếu thiếu thư viện (hữu ích khi chạy test local thiếu lib)
    print("WARNING: Không tìm thấy thư viện lasotuvi. Các chức năng Tử Vi sẽ không hoạt động.")
    lapDiaBan = None
    DiaBanClass = None
    lapThienBan = None

from prompts import get_tarot_prompt, get_astrology_prompt, get_numerology_prompt, get_horoscope_prompt

# ==========================================
# 0. CONSTANTS & CONFIGURATION (Đã gộp vào đây)
# ==========================================
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-1")
# Model ID mặc định là Nova Pro như bạn yêu cầu
LLM_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-pro-v1:0") 
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "SorcererXStreme_Metaphysical_Table")

# === KHỞI TẠO CLIENTS ===
# Lưu ý: Khởi tạo global giúp tận dụng connection reuse trong Lambda
try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    dynamodb = boto3.resource('dynamodb', region_name=BEDROCK_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
except Exception as e:
    print(f"INIT ERROR: Không thể khởi tạo AWS Clients. {e}")
    bedrock_client = None
    table = None

# ==========================================
# 1. HELPER FUNCTIONS (UTILITIES)
# ==========================================

def get_db_item(category, entity_name):
    """
    Lấy item từ DynamoDB và tự động parse JSON string trong trường 'contexts'.
    """
    if not table: return {}
    try:
        response = table.get_item(
            Key={
                'category': category,
                'entity_name': entity_name
            }
        )
        item = response.get('Item')
        if not item:
            print(f"WARN: Item not found for {category} - {entity_name}")
            return {}
        
        contexts_str = item.get('contexts', '{}')
        if isinstance(contexts_str, str):
            try:
                return json.loads(contexts_str)
            except json.JSONDecodeError:
                return {}
        return contexts_str 
    except Exception as e:
        print(f"Error getting item from DynamoDB: {str(e)}")
        return {}

def call_bedrock_llm(prompt, temperature=0.5):
    """Gửi prompt tới Model"""
    if not bedrock_client:
        return "Lỗi: Kết nối tới Bedrock chưa được thiết lập."

    # Cấu trúc Body chuẩn của Amazon Nova Pro
    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 2000, 
            "temperature": temperature,
            "top_p": 0.9
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt} 
                ]
            }
        ]
    })

    try:
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=body
        )
        # Đọc stream từ body
        response_body = json.loads(response.get('body').read())
        
        # Parse response của Nova: output -> message -> content -> text
        return response_body['output']['message']['content'][0]['text']

    except Exception as e:
        print(f"Error calling Bedrock ({LLM_MODEL_ID}): {str(e)}")
        return "Xin lỗi, Vũ trụ Nova đang hiệu chỉnh năng lượng. Vui lòng thử lại sau."

def parse_date(date_str):
    if not date_str:
        return None
    s = str(date_str)
    s = s.replace('–', '-').replace('—', '-').replace('.', '-').replace('/', '-')
    s = s.strip()
    
    formats = ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d") 
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    print(f"DEBUG: Không parse được ngày: '{date_str}'")
    return None

# ==========================================
# 2. DOMAIN LOGIC
# ==========================================

# --- ASTROLOGY (CHIÊM TINH) ---
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
    if not context_json:
        return f"Không có dữ liệu chi tiết cho {zodiac_name}."
    return f"""
    - Cung: {zodiac_name}
    - Tính cách: {context_json.get('tinh-cach', '')}
    - Tình yêu: {context_json.get('tinh-yeu', '')}
    - Điểm mạnh: {context_json.get('diem-manh', '')}
    - Điểm yếu: {context_json.get('diem-yeu', '')}
    - Cung hợp: {context_json.get('cung-hop', '')}
    """

def handle_astrology(body):
    feature_type = body.get('feature_type', 'overview')
    user_context = body.get('user_context', {})
    
    dob_str = user_context.get('birth_date')
    user_date = parse_date(dob_str)
    user_gender = user_context.get('gender', 'unknown')
    
    if not user_date:
        return "Ngày sinh không hợp lệ."
        
    user_zodiac = calculate_zodiac(user_date.day, user_date.month)
    user_zodiac_data = get_db_item('cung-hoang-dao', user_zodiac)
    
    if feature_type == 'overview':
        context_str = format_zodiac_context(user_zodiac, user_zodiac_data)
        internal_query = f"Phân tích tổng quan vận mệnh, tính cách cho người cung {user_zodiac} sinh ngày {dob_str}."
        prompt = get_astrology_prompt('overview', user_zodiac, dob_str, context_str, internal_query, user_gender)
        return call_bedrock_llm(prompt, temperature=0.5)
        
    elif feature_type == 'love':
        partner_context = body.get('partner_context', {})
        p_dob_str = partner_context.get('birth_date')
        p_date = parse_date(p_dob_str)
        
        if not p_date:
            return "Thiếu thông tin ngày sinh đối phương."
            
        partner_zodiac = calculate_zodiac(p_date.day, p_date.month)
        partner_zodiac_data = get_db_item('cung-hoang-dao', partner_zodiac)
        
        u_compatible = user_zodiac_data.get('cung-hop', '')
        p_compatible = partner_zodiac_data.get('cung-hop', '')
        
        is_user_match = partner_zodiac in u_compatible
        is_partner_match = user_zodiac in p_compatible
        
        match_status = ""
        if is_user_match and is_partner_match:
            match_status = "RẤT HỢP (Theo sách: Cả hai đều nằm trong danh sách hợp của nhau)."
        elif is_user_match or is_partner_match:
            match_status = "KHÁ HỢP (Theo sách: Có sự thu hút thuận lợi từ một phía)."
        else:
            match_status = "CẦN CỐ GẮNG (Theo sách: Không nằm trong nhóm hợp tự nhiên, cần nỗ lực thấu hiểu)."
            
        combined_context = f"""
        THÔNG TIN NGƯỜI DÙNG (USER): {user_zodiac}
        {format_zodiac_context(user_zodiac, user_zodiac_data)}
        
        THÔNG TIN ĐỐI PHƯƠNG (PARTNER): {partner_zodiac}
        {format_zodiac_context(partner_zodiac, partner_zodiac_data)}
        
        === ĐÁNH GIÁ ĐỘ HỢP TỪ DỮ LIỆU ===
        Kết luận sơ bộ: {match_status}
        """
        
        love_query = f"Phân tích độ hợp nhau giữa {user_zodiac} và {partner_zodiac}. Dựa trên 'Đánh giá độ hợp' đã cung cấp để đưa ra lời khuyên."
        prompt = get_astrology_prompt('love', f"{user_zodiac} & {partner_zodiac}", f"{dob_str} - {p_dob_str}", combined_context, love_query, user_gender)
        return call_bedrock_llm(prompt, temperature=0.6)

# --- NUMEROLOGY (THẦN SỐ HỌC) ---
def calculate_life_path(day, month, year):
    full_str = f"{day}{month}{year}"
    total = sum(int(digit) for digit in full_str)
    while total > 9:
        if total in [11, 22, 33, 10]:
            break
        total = sum(int(digit) for digit in str(total))
    return str(total)

def handle_numerology(body):
    user_context = body.get('user_context', {})
    dob_str = user_context.get('birth_date')
    user_date = parse_date(dob_str)
    
    if not user_date:
        return "Ngày sinh không hợp lệ."
        
    life_path = calculate_life_path(user_date.day, user_date.month, user_date.year)
    entity_name = f"Số {life_path}"
    
    context_data = get_db_item('numerology_number', entity_name)
    
    context_str = f"""
    - Số chủ đạo: {life_path}
    - Tổng quan: {context_data.get('tong-quan', '')}
    - Ưu điểm: {context_data.get('uu-diem', '')}
    - Nhược điểm: {context_data.get('nhuoc-diem', '')}
    - Sứ mệnh: {context_data.get('chi-so-su-menh', '')}
    - Lời khuyên công việc: {context_data.get('so-hop-cong-viec', '')}
    - Lời khuyên tình yêu: {context_data.get('so-hop-tinh-yeu', '')}
    """
    
    internal_query = f"Phân tích chi tiết Thần số học số {life_path} cho người sinh ngày {dob_str}."
    prompt = get_numerology_prompt(life_path, dob_str, context_str, internal_query, user_context.get('gender'))
    return call_bedrock_llm(prompt, temperature=0.5)

# --- TAROT ---
def handle_tarot(body):
    feature_type = body.get('feature_type', 'question')
    data = body.get('data', {})
    cards_input = data.get('cards_drawn', [])
    user_context = body.get('user_context', {})
    user_query = data.get('question', '')
    
    if not cards_input:
        return "Vui lòng chọn lá bài."

    intent_topic = "general"
    if user_query:
        q_lower = user_query.lower()
        if any(k in q_lower for k in ['yêu', 'tình', 'crush', 'cưới', 'hẹn hò']): intent_topic = "love"
        elif any(k in q_lower for k in ['việc', 'làm', 'nghề', 'lương', 'công ty']): intent_topic = "work"
        elif any(k in q_lower for k in ['khoẻ', 'bệnh', 'thuốc', 'sức khoẻ']): intent_topic = "health"
        elif any(k in q_lower for k in ['bạn', 'gia đình', 'quan hệ']): intent_topic = "relationship"

    position_mapping = {
        "past": "Quá khứ / Nguyên nhân",
        "present": "Hiện tại / Diễn biến",
        "future": "Tương lai / Kết quả"
    }

    context_parts = []
    context_parts.append(f"Chủ đề: {intent_topic.upper()}")
    if user_query: context_parts.append(f"Câu hỏi: {user_query}")

    for card in cards_input:
        raw_name = card.get('card_name', '').strip()
        is_upright = card.get('is_upright', True)
        position = card.get('position')
        
        db_entity_name = raw_name.title() 
        card_full_data = get_db_item('tarot_card', db_entity_name)
        
        suffix = "upright" if is_upright else "reversed"
        target_key = f"{intent_topic}_{suffix}"
        backup_key = f"general_{suffix}"
        
        detail_content = card_full_data.get(target_key) or card_full_data.get(backup_key) or "Không có dữ liệu chi tiết."
        
        orientation_str = "Xuôi" if is_upright else "Ngược"
        pos_label = f"[{position_mapping.get(position, 'Vị trí ngẫu nhiên')}]" if position else ""
        
        card_info = f"- {pos_label} Lá bài: {db_entity_name} ({orientation_str})\n  Ý nghĩa ({intent_topic}): {detail_content}"
        context_parts.append(card_info)
        
    full_context_str = "\n".join(context_parts)
    effective_query = user_query if user_query else "Phân tích trải bài tổng quan."
    
    prompt = get_tarot_prompt(feature_type, full_context_str, effective_query, user_context, intent_topic)
    return call_bedrock_llm(prompt, temperature=0.7)

# --- HOROSCOPE (TỬ VI) ---
def parse_time_to_chi(time_str):
    if not time_str: return 12
    try:
        hour = int(str(time_str).split(':')[0])
    except: return 1
    if hour >= 23 or hour < 1: return 1 
    return (hour + 1) // 2 + 1

def map_gender_tuvi(gender_str):
    if not gender_str: return 1
    return 1 if str(gender_str).lower() in ['male', 'nam', '1'] else -1

def extract_tuvi_metadata(thien_ban, dia_ban):
    try:
        ten_cung_menh = dia_ban.thapNhiCung[dia_ban.cungMenh].cungTen
        ten_cung_than = dia_ban.thapNhiCung[dia_ban.cungThan].cungTen
        return {
            "can_chi_nam": f"{thien_ban.canNamTen} {thien_ban.chiNamTen}",
            "ban_menh": thien_ban.banMenh,
            "cuc": thien_ban.tenCuc,
            "menh_chu": thien_ban.menhChu,
            "than_chu": thien_ban.thanChu,
            "vi_tri_menh": f"Cung {ten_cung_menh}",
            "vi_tri_than": f"Cung {ten_cung_than}"
        }
    except: return {}

def generate_tuvi_context_text(thien_ban, dia_ban):
    lines = [f"Đương số: {thien_ban.ten}, Mệnh: {thien_ban.banMenh}, Cục: {thien_ban.tenCuc}"]
    for i in range(1, 13):
        cung = dia_ban.thapNhiCung[i]
        sao_chinh = [s['saoTen'] for s in cung.cungSao if s.get('saoLoai') == 1]
        lines.append(f"Cung {getattr(cung, 'cungChu', '')} tại {cung.cungTen}: {', '.join(sao_chinh)}")
    return "\n".join(lines)

def handle_horoscope(body):
    user_context = body.get('user_context', {})
    name = user_context.get('name', 'Đương số')
    dob_str = user_context.get('birth_date')
    tob_str = user_context.get('birth_time', '12:00')
    gender_str = user_context.get('gender', 'male')
    
    dob_date = parse_date(dob_str)
    if not dob_date: return {"error": "Ngày sinh lỗi"}
    
    dd, mm, yy = dob_date.day, dob_date.month, dob_date.year
    chi_gio = parse_time_to_chi(tob_str)
    gender_input = map_gender_tuvi(gender_str)
    
    try:
        if lapDiaBan is None:
            raise ImportError("Thư viện lasotuvi không khả dụng.")

        db = lapDiaBan(DiaBanClass, dd, mm, yy, chi_gio, gender_input, duongLich=True, timeZone=7)
        tb = lapThienBan(dd, mm, yy, chi_gio, gender_input, name, db, duongLich=True, timeZone=7)
        
        summary_data = extract_tuvi_metadata(tb, db)
        rag_context = generate_tuvi_context_text(tb, db)
        
        prompt = get_horoscope_prompt(rag_context, user_context)
        ai_response = call_bedrock_llm(prompt, temperature=0.7)
        
        return {
            "summary": summary_data,
            "analysis": ai_response,
            "metadata": {
                "name": name,
                "dob_solar": f"{dd}/{mm}/{yy}",
                "dob_lunar": f"{tb.ngayAm}/{tb.thangAm}/{tb.namAm}"
            }
        }
    except Exception as e:
        print(f"TUVI ERROR: {e}")
        return {"error": str(e)}

# === MAIN HANDLER ===
def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)
            
        domain = body.get('domain', '').lower()
        ans = ""
        
        if domain == 'tarot':
            ans = handle_tarot(body)
        elif domain == 'astrology':
            ans = handle_astrology(body)
        elif domain == 'numerology':
            ans = handle_numerology(body)
        elif domain == 'horoscope':
            ans = handle_horoscope(body)
        else:
            return {'statusCode': 400, 'body': json.dumps({'error': f'Invalid domain: {domain}'})}
            
        return {
            'statusCode': 200, 
            'headers': {
                'Content-Type': 'application/json', 
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'domain': domain, 
                'feature': body.get('feature_type'),
                'answer': ans
            }, ensure_ascii=False)
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error', 'details': str(e)})
        }