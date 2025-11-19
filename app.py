from flask import Flask, request, jsonify
import torch
import numpy as np
import io
from model import Simple3DCNN

app = Flask(__name__)

# 1. Cấu hình thiết bị (dùng CPU để chạy cho nhẹ trong Docker)
device = torch.device("cpu")

# 2. Khởi tạo mô hình
model = Simple3DCNN()

# 3. Tải trọng số (weights) từ file .pth
# Lưu ý: File simple3dcnn.pth phải nằm cùng thư mục
try:
    model.load_state_dict(torch.load("simple3dcnn.pth", map_location=device))
    model.to(device)
    model.eval() # Chuyển sang chế độ dự đoán
    print("Load model thành công!")
except Exception as e:
    print(f"Lỗi load model: {e}")

def preprocess_input(file_stream):
    """
    Hàm này đọc file .npy (numpy array) từ upload
    và chuyển đổi thành Tensor đúng định dạng cho mô hình.
    Yêu cầu đầu vào: Mảng numpy kích thước (32, 32, 32)
    """
    # Đọc dữ liệu từ file upload
    data = np.load(file_stream)
    
    # Kiểm tra kích thước (tùy chọn, nhưng nên có để tránh lỗi)
    if data.shape != (32, 32, 32):
        raise ValueError(f"Kích thước input sai. Mong đợi (32,32,32), nhận được {data.shape}")

    # Chuyển thành Tensor
    tensor = torch.from_numpy(data).float()
    
    # Thêm chiều batch và channel: (32,32,32) -> (1, 1, 32, 32, 32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Chưa gửi file"}), 400
    
    file = request.files['file']
    
    try:
        # 1. Tiền xử lý
        input_tensor = preprocess_input(file)
        
        # 2. Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            # output là xác suất từ 0 đến 1 (do dùng sigmoid)
            malignancy_risk = output.item()
            
        return jsonify({
            "filename": file.filename,
            "malignancy_risk": malignancy_risk,
            "risk_percentage": f"{malignancy_risk * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)