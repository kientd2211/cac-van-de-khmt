from flask import Flask, request, jsonify
import torch
import os
import time
import uuid
from desnet3D import densenet3d121
from preprocessing import process_mha_file

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/luna_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DEVICE = torch.device("cpu")

print("Đang tải model...")
try:
    model = densenet3d121(num_classes=1)
    checkpoint = torch.load("desnet3d.pth", map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    print("✅ Model DesNet3D đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    model = None

@app.route("/api/v1/predict/lesion", methods=["POST"])
def predict_lesion():
    start_time = time.time()
    temp_path = None

    if not model:
        return jsonify({"error": "SERVICE_UNAVAILABLE", "message": "Model chưa được tải"}), 404

    if 'file' not in request.files:
        return jsonify({"error": "INVALID_FILE_FORMAT", "message": "Thiếu file ảnh"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "INVALID_FILE_FORMAT", "message": "Tên file rỗng"}), 400

    required_fields = ['seriesInstanceUID', 'lesionID', 'coordX', 'coordY', 'coordZ']
    form_data = request.form
    for field in required_fields:
        if field not in form_data:
            return jsonify({"error": "MISSING_FIELD", "message": f"Thiếu trường {field}"}), 400

    try:
        series_uid = form_data.get('seriesInstanceUID')
        lesion_id = int(form_data.get('lesionID'))
        coord_x = float(form_data.get('coordX'))
        coord_y = float(form_data.get('coordY'))
        coord_z = float(form_data.get('coordZ'))
        
        ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        input_tensor = process_mha_file(temp_path, coord_x, coord_y, coord_z)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()
            

        prediction_label = 1 if probability > 0.5 else 0
        
        processing_time = int((time.time() - start_time) * 1000)

        response = {
            "status": "success",
            "data": {
                "seriesInstanceUID": series_uid,
                "lesionID": lesion_id,
                "probability": round(probability, 4),
                "predictionLabel": prediction_label,
                "processingTimeMs": processing_time
            }
        }
        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": "INVALID_DATA", "message": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": "PROCESSING_ERROR", "message": str(re)}), 422
    except Exception as e:
        return jsonify({"error": "INTERNAL_SERVER_ERROR", "message": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)