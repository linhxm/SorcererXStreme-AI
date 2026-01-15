import boto3

try:
    # Thử lấy ID của tài khoản đang kết nối
    sts = boto3.client('sts')
    print("Kết nối thành công!")
    print(f"Account ID: {sts.get_caller_identity()['Account']}")
except Exception as e:
    print(f"Lỗi kết nối: {e}")