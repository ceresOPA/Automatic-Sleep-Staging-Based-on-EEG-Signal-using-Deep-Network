import os
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
import time

from predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/test", methods=["GET","POST"])
def test():
    data = {"message":"hello"}
    return jsonify(data)

@app.route(f"/index", methods=["GET"])
def sleepstaging():
    return render_template("sleepstaging.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    msg = {}
    if('file' not in request.files):
        msg['state'] = 'fail'
        msg['msg'] = "文件获取失败"
        return jsonify(msg)
    file = request.files['file']
    if(file.filename == ''):
        msg['state'] = 'fail'
        msg['msg'] = "文件获取失败"
        return jsonify(msg)
    if(file and allowed_file(file.filename)):
        filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{filename}"

        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        file_path = f"{UPLOAD_FOLDER}/{filename}"
        print(file_path)
        file.save(file_path)
        
        preds = predict(seq_len=16, input_file=file_path)
        preds = preds.numpy().tolist()
        epochs = [i for i in range(len(preds))]

        msg['state'] = 'success'
        msg['filename'] = filename
        msg['preds'] = preds
        msg['epochs'] = epochs
        return jsonify(msg)
    msg['msg'] = 'fail'
    msg['msg'] = "请求错误"
    return jsonify(msg)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)