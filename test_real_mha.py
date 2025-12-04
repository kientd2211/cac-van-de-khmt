import os
import pandas as pd
import requests
import time

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn file CSV chứa nhãn thực tế (Ground Truth)
CSV_PATH = r"D:\Code\stuffs\other\LUNA25_Public_Training_Development_Data.csv"

# Đường dẫn thư mục chứa ảnh .mha
DATA_DIR = r"D:\Code\stuffs\data-test"

# URL API
API_URL = "http://localhost:8080/api/v1/predict/lesion"

# Số lượng file muốn test
BATCH_SIZE = 100

def main():
    print(f"--- BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG (BATCH SIZE: {BATCH_SIZE}) ---")

    # 1. Load dữ liệu CSV (Ground Truth)
    if not os.path.exists(CSV_PATH):
        print(f"❌ Lỗi: Không tìm thấy file CSV tại {CSV_PATH}")
        return
    
    try:
        df_ground_truth = pd.read_csv(CSV_PATH)
        print("✅ Đã load file CSV thành công.")
    except Exception as e:
        print(f"❌ Lỗi đọc CSV: {e}")
        return

    # 2. Lấy danh sách file .mha trong thư mục Data
    if not os.path.exists(DATA_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục data tại {DATA_DIR}")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mha')]
    
    if not all_files:
        print("❌ Không tìm thấy file .mha nào trong thư mục data.")
        return

    # files_to_test = all_files[:BATCH_SIZE]
    files_to_test = all_files

    # Danh sách chứa kết quả để xuất Excel
    results = []

    # 3. Vòng lặp xử lý từng file
    for idx, filename in enumerate(files_to_test, 1):
        print(f"[{idx}/{len(files_to_test)}] Processing: {filename} ... ", end="")
        
        # Lấy UID từ tên file (bỏ đuôi .mha)
        series_uid = os.path.splitext(filename)[0]
        
        # Tìm thông tin trong CSV
        matching_row = df_ground_truth[df_ground_truth['SeriesInstanceUID'] == series_uid]

        # Khởi tạo giá trị mặc định cho báo cáo
        status_code = "N/A"
        prob = "N/A"
        pred_label = "N/A"
        proc_time = "N/A"
        conclusion = "N/A"
        ground_truth_label = "N/A"
        
        if matching_row.empty:
            print("❌ Missing in CSV")
            status_code = "404 (CSV)"
            conclusion = "Không tìm thấy trong CSV"
        else:
            # Lấy dữ liệu dòng đầu tiên tìm thấy
            row_data = matching_row.iloc[0]
            ground_truth_label = int(row_data.get('label', -1)) # Lấy nhãn thực tế
            
            # Chuẩn bị Payload gửi API
            payload = {
                'seriesInstanceUID': series_uid,
                'lesionID': 1, 
                'coordX': row_data['CoordX'],
                'coordY': row_data['CoordY'],
                'coordZ': row_data['CoordZ'],
                'patientID': 'BATCH_TEST',
                'gender': 'Male' # Giả định hoặc lấy từ CSV nếu có
            }

            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                # Gửi Request
                with open(file_path, 'rb') as f:
                    files = {'file': (filename, f, 'application/octet-stream')}
                    start_req = time.time()
                    response = requests.post(API_URL, data=payload, files=files)
                    end_req = time.time()

                status_code = response.status_code
                
                if status_code == 200:
                    resp_json = response.json()
                    data = resp_json.get('data', {})
                    
                    prob = data.get('probability')
                    pred_label = data.get('predictionLabel')
                    proc_time = data.get('processingTimeMs')

                    # So sánh kết quả
                    if ground_truth_label != -1:
                        if pred_label == ground_truth_label:
                            conclusion = "ĐÚNG"
                            print("✅ OK")
                        else:
                            conclusion = "SAI"
                            print("❌ FAIL")
                    else:
                        conclusion = "Không có nhãn gốc"
                        print("⚠️ No Label")
                else:
                    print(f"❌ API Error {status_code}")
                    conclusion = f"Lỗi API: {response.text[:50]}"

            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                conclusion = f"Lỗi Code: {str(e)}"

        # Lưu kết quả vào danh sách
        results.append({
            "STT": idx,
            "Tên File Ảnh": filename,
            "Nhãn Thực Tế (Ground Truth)": ground_truth_label,
            "Status Code": status_code,
            "Xác Suất (Probability)": prob,
            "Dự Đoán (Prediction Label)": pred_label,
            "Thời Gian Xử Lý (ms)": proc_time,
            "Kết Luận": conclusion
        })

    # 4. Xuất ra Excel
    print("\n--- Đang xuất file Excel... ---")
    output_file = "ket_qua_danh_gia_model_2.xlsx"
    try:
        df_result = pd.DataFrame(results)
        df_result.to_excel(output_file, index=False)
        print(f"✅ Đã tạo file thành công: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ Lỗi khi ghi file Excel: {e}")

if __name__ == "__main__":
    main()