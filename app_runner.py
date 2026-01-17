import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH BIẾN MÔI TRƯỜNG ---
os.environ['BEDROCK_REGION'] = 'ap-southeast-1'
os.environ['DYNAMODB_TABLE_NAME'] = 'SorcererXStreme_Metaphysical_Table'
os.environ['DDB_MESSAGE_TABLE'] = 'sorcererxstreme-chatMessages'
os.environ['CACHE_TABLE'] = 'SorcererXStreme_Metaphysical_Cache'

# Import 2 handler khác nhau
try:
    from src.metaphysical.lambda_function import lambda_handler as metaphysical_handler
    from src.chatbot.lambda_function import lambda_handler as chatbot_handler
    print("✅ Đã kết nối thành công: Chatbot & Metaphysical Handlers.")
except ImportError as e:
    print(f"❌ Lỗi Import: {e}. Kiểm tra lại cấu trúc thư mục src/...")

app = Flask(__name__)
CORS(app)

@app.route('/test/metaphysical', methods=['POST'])
def test_metaphysical():
    event = request.json
    return jsonify(metaphysical_handler(event, {}))

@app.route('/test/chatbot', methods=['POST'])
def test_chatbot():
    event = request.json
    return jsonify(chatbot_handler(event, {}))

if __name__ == '__main__':
    print("🚀 Server Tester chạy tại: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)