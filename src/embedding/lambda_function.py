import json
import boto3
import os
import sys
import hashlib
from pinecone import Pinecone

# === CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ===
try:
    # Cấu hình AWS
    S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
    S3_FILE_KEY = os.environ['S3_FILE_KEY']
    BEDROCK_REGION = os.environ.get('BEDROCK_REGION', 'ap-southeast-1')
    DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
    
    # Cấu hình Pinecone
    PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
    PINECONE_HOST = os.environ['PINECONE_HOST'] 
    
except KeyError as e:
    print(f"CRITICAL ERROR: Thiếu biến môi trường: {e}")
    sys.exit(1)

# === KHỞI TẠO CLIENTS ===
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
table = dynamodb.Table(DYNAMODB_TABLE)

# Khởi tạo Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

def get_embedding(text):
    """Gọi Bedrock Cohere để lấy vector (1024 dimensions)"""
    # Cắt ngắn text nếu quá dài (Cohere giới hạn token, mức 2000-4000 ký tự là an toàn)
    if len(text) > 4000: 
        text = text
    body = json.dumps({
        "texts": [text],
        "input_type": "search_document" 
    })
    
    try:
        response = bedrock_client.invoke_model(
            body=body,
            modelId='cohere.embed-multilingual-v3',
            contentType='application/json',
            accept='*/*'
        )
        response_body = json.loads(response['body'].read())
        return response_body['embeddings'][0]
    except Exception as e:
        print(f"BEDROCK ERROR: {e}")
        return None

def flatten_contexts(contexts):
    """
    Biến đổi dict contexts thành chuỗi văn bản cho embedding.
    Giữ nguyên tiếng Việt để AI hiểu ngữ nghĩa tốt nhất.
    """
    text_parts = []
    for key, value in contexts.items():
        if value and isinstance(value, str):
            readable_key = key.replace("-", " ").replace("_", " ") 
            text_parts.append(f"{readable_key}: {value}")
        elif isinstance(value, list):
            readable_key = key.replace("-", " ")
            text_parts.append(f"{readable_key}: {', '.join(value)}")
            
    return ". ".join(text_parts)

def lambda_handler(event, context):
    print(f"BẮT ĐẦU MIGRATE: {S3_BUCKET_NAME}/{S3_FILE_KEY}")
    
    # 1. Đọc file từ S3
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
        dataset_content = s3_object['Body'].read().decode('utf-8')
    except Exception as e:
        return {'statusCode': 500, 'body': f"Lỗi đọc S3: {e}"}

    lines = dataset_content.splitlines()
    print(f"Tìm thấy {len(lines)} items.")

    batch_vectors = []
    BATCH_SIZE = 50 
    success_count = 0
    
    # Dùng dynamodb batch writer để ghi nhanh hơn
    with table.batch_writer() as batch_db:
        for i, line in enumerate(lines):
            if not line.strip(): continue
            
            try:
                item = json.loads(line)
                category = item.get('category', 'unknown')
                entity_name = item.get('entity_name', 'unknown')
                keywords = item.get('keywords', [])
                contexts = item.get('contexts', {})
                
                # --- XỬ LÝ ID (QUAN TRỌNG) ---
                # Tạo raw string định danh
                raw_id = f"{category}#{entity_name}"
                
                # Hash sang MD5 (vd: '5d4140...') để đảm bảo ID là ASCII và duy nhất
                # Việc này KHÔNG ảnh hưởng đến khả năng tìm kiếm ngữ nghĩa
                unique_id = hashlib.md5(raw_id.encode('utf-8')).hexdigest()
                
                # --- A. LƯU VÀO DYNAMODB (FULL DATA TIẾNG VIỆT) ---
                batch_db.put_item(Item={
                    'category': category,       # Partition Key
                    'entity_name': entity_name, # Sort Key (Vẫn giữ tiếng Việt có dấu)
                    'keywords': keywords,
                    'contexts': json.dumps(contexts, ensure_ascii=False)
                })
                
                # --- B. CHUẨN BỊ VECTOR CHO PINECONE ---
                # 1. Tạo chuỗi văn bản chứa TOÀN BỘ thông tin (Vẫn giữ tiếng Việt có dấu để Embedding chuẩn)
                context_str = flatten_contexts(contexts)
                keywords_str = ", ".join(keywords)
                
                text_to_embed = (
                    f"Chủ đề: {category}. "
                    f"Tên: {entity_name}. "
                    f"Từ khóa: {keywords_str}. "
                    f"Nội dung chi tiết: {context_str}"
                )
                
                # 2. Get Embedding (Cohere sẽ đọc text tiếng Việt ở đây)
                vector_values = get_embedding(text_to_embed)
                
                if vector_values:
                    # Thêm vào batch list
                    batch_vectors.append({
                        "id": unique_id, # ID là mã Hash ASCII (hợp lệ cho Pinecone)
                        "values": vector_values,
                        "metadata": {
                            "category": category,
                            "entity_name": entity_name, # Metadata vẫn lưu tiếng Việt để hiển thị lại cho user
                            "keywords": keywords_str 
                        }
                    })
                
                # 3. Batch Upload lên Pinecone
                if len(batch_vectors) >= BATCH_SIZE:
                    index.upsert(vectors=batch_vectors)
                    print(f"Upserted batch {len(batch_vectors)} items to Pinecone.")
                    batch_vectors = [] 
                
                success_count += 1
                
            except Exception as e:
                print(f"Lỗi item dòng {i}: {e}")

        # Upsert nốt số vector còn lại trong batch cuối cùng
        if len(batch_vectors) > 0:
            index.upsert(vectors=batch_vectors)
            print(f"Upserted final batch {len(batch_vectors)} items.")

    summary = f"HOÀN TẤT! Đã xử lý thành công: {success_count}/{len(lines)}"
    print(summary)
    
    return {
        'statusCode': 200,
        'body': json.dumps(summary)
    }