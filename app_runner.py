import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Thiết lập biến môi trường giống trên Lambda Console
os.environ['BEDROCK_REGION'] = 'ap-southeast-1'
os.environ['DYNAMODB_TABLE_NAME'] = 'SorcererXStreme_Metaphysical_Table'
os.environ['BEDROCK_MODEL_ID'] = 'apac.amazon.nova-pro-v1:0'
os.environ['PINECONE_API_KEY'] = 'YOUR_PINECONE_KEY'
os.environ['PINECONE_HOST'] = 'YOUR_PINECONE_HOST'
os.environ['AWS_DEFAULT_REGION'] = os.environ.get('BEDROCK_REGION', 'ap-southeast-1')

# Import trực tiếp handler từ file của bạn
from src.metaphysical.lambda_function import lambda_handler as metaphysical_handler
from src.chatbot.lambda_function import lambda_handler as chatbot_handler

app = Flask(__name__)
CORS(app) # Cho phép Frontend gọi API

@app.route('/test/metaphysical', methods=['POST'])
def test_metaphysical():
    event = request.json
    # Giả lập context của Lambda
    context = {}
    result = metaphysical_handler(event, context)
    return jsonify(result)

@app.route('/test/chatbot', methods=['POST'])
def test_chatbot():
    event = request.json
    context = {}
    result = chatbot_handler(event, context)
    return jsonify(result)

if __name__ == '__main__':
    print("Server local đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)