# LUNA25 - DesNet3D Lung Nodule Malignancy Prediction API

API dự đoán nguy cơ ác tính nốt phổi từ dữ liệu cắt lớp 3D sử dụng mô hình **DesNet3D**.

## 1. Cấu trúc dự án
* `app.py`: API Server (Flask).
* `desnet3D.py`: Kiến trúc mô hình DesNet3D.
* `desnet3d.pth`: Trọng số mô hình đã huấn luyện.
* `Dockerfile`: Cấu hình môi trường Docker.

## 2. Yêu cầu
* Docker Desktop được cài đặt.

## 3. Cách cài đặt và chạy

### Bước 1: Build Image
```bash
docker build -t luna-desnet .
```

### Bước 2: Run Container
Chạy container và mở cổng 8080:
```bash
docker run -d -p 8080:5000 luna-desnet
```

> **Cách chạy từ file .tar:**
    > 1. `docker load -i luna-desnet.tar`
    > 2. `docker run -d -p 8080:5000 luna-desnet`