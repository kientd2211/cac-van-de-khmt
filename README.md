# LUNA25 Lung Nodule Malignancy Prediction API

Đây là Docker API của nhóm 23 để dự đoán nguy cơ ác tính của nốt phổi từ dữ liệu cắt lớp 3D (patch 32x32x32), sử dụng mô hình 3D-CNN.

## 1. Cấu trúc dự án
* `app.py`: API Server sử dụng Flask.
* `model.py`: Định nghĩa kiến trúc mô hình PyTorch.
* `simple3dcnn.pth`: Trọng số mô hình đã huấn luyện.
* `Dockerfile`: Cấu hình môi trường chạy.

## 2. Yêu cầu cài đặt
* Cần cài đặt **Docker Desktop** trên máy tính.

## 3. Cách cài đặt và Chạy (Deployment)

### Bước 1: Build Docker Image
Mở Terminal tại thư mục này và chạy lệnh:
```bash
docker build -t luna-api .
```

### Bước 2: Run Container
Chạy container và mở cổng 8080:
```bash
docker run -d -p 8080:5000 luna-api
```

> **Cách chạy từ file .tar:**
    > 1. `docker load -i luna-api.tar`
    > 2. `docker run -d -p 8080:5000 luna-api`