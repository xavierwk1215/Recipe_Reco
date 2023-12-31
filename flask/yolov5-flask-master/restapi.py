"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "G:\내 드라이브\ds_study\DL_team_4\DataSet\YOLOv5_Taebin3\yolov5"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640) # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('G:\내 드라이브\ds_study\DL_team_4\DataSet\YOLOv5_Taebin3\yolov5', args.model)
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
