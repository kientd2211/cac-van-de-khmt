# 1. Chọn ảnh nền (Base Image) Python nhẹ
FROM python:3.10-slim

# 2. Đặt thư mục làm việc bên trong container
WORKDIR /app

# 3. Sao chép tệp requirements vào trước để tối ưu cache
COPY requirements.txt .

# 4. Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép toàn bộ mã nguồn dự án vào container
COPY . .

# 6. Mở cổng 5000 (cổng mà Flask/Gunicorn đang chạy)
EXPOSE 5000

# 7. Lệnh để chạy ứng dụng khi container khởi động
# Dùng Gunicorn để chạy app:app (tệp app.py, biến app)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]