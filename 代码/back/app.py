import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings

warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
import uuid
from flask_cors import CORS  # 导入
import base64
import requests
import json
import numpy as np
import pandas as pd
import joblib
import pickle
import re
import jieba
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2

# 声明
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 显示中文，编码
CORS(app, supports_credentials=True)  # 开启跨域
########机器学习
Mlearn_MODEL = joblib.load("./mlearn/dx.pkl")  # 加载模型 预估器
with open("./mlearn/transfer.bin", "rb") as f: # 加载  转换器
    transfer = pickle.load(f)
stop = pd.read_csv("./mlearn/stopword.txt", sep='bingrong',                 # 加载 停用词
                   header=None, encoding="utf8", engine='python')
stop = stop[0].to_list()
dxcls = {0: "负面评价", 1: "正面评价"}  # 修改类别为新闻分类
########easydl
EASYDL_API_URL = "http://127.0.0.1:24401/"
########深度学习
Deep_MODEL = load_model('deeplearn/mask_VGG16_3.h5')  # 模型文件
DEEP_CLASS = ['未正确带好口罩','戴好口罩', '未戴口罩']  # label-->0，1，2，3
faceCascade = cv2.CascadeClassifier(os.path.join(os.getcwd(),"deeplearn","haarcascade_frontalface_default.xml"))
img_height = 64
img_width = 64
##############函数接口
##############机器学习情感分析
@ app.route('/sbtext', methods=["POST"])
def sbtextfun():
    #获取前端返回的POST
    msg = request.values.get("text")
    #清洗数据
    text = re.sub("x|[^\u4E00-\u9FA5]|[0-9]|\\s|\\t", "", msg)  # 去除非中文，保留中文
    text = jieba.lcut(text)
    msg = [" ".join([i for i in text if i not in stop])]
    #使用模型
    msg = transfer.transform(msg)  # 转换器
    result =Mlearn_MODEL.predict(msg)  # 预估器
    #返回结果
    data={"text": text, "textcode": str(msg), "sb":  dxcls.get(result[0])}
    return jsonify(data)
# ############easydl人脸检测
@ app.route('/sbeasydl', methods=["POST"])
def sbeasyfun():
    #读取前端传来的图片
    img = request.files.get("imgfile")
    imgbyte = img.read()
    PARAMS = {}
    #转换为base64编码
    base64_data = base64.b64encode(imgbyte)
    base64_str = base64_data.decode('UTF8')
    PARAMS["image"] = base64_str
    #获取前端返回的POST
    response = requests.post(url=EASYDL_API_URL, json=PARAMS)
    response_json = response.json()
    data = [{"name": i.get("name"), "loc": i.get("location"), "score": i.get(
        "score")} for i in response_json["results"]]
    #将图片转换为uint8类型，再转换为RGB格式
    img_array = np.frombuffer(imgbyte, np.uint8)
    img_array = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    #根据data绘制矩形框和文字标注
    for i in data:
        x1 = i.get("loc").get("left")
        y1 = i.get("loc").get("top")
        x2 = x1+i.get("loc").get("width")
        y2 = y1+i.get("loc").get("height")
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img_array, i.get("name"), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
    #保存绘制好的文件和json文件
    filepath = os.path.join(os.getcwd(), "easydl", "imgs", str(uuid.uuid1()))
    imgpath = filepath+"."+img.filename.split(".")[-1]
    with open(imgpath, "wb+") as f:
        f.write(imgbyte)
    jsonpath = filepath+".json"
    cv2.imwrite(imgpath, img_array)
    with open(jsonpath, "w+") as f:
        f.write(json.dumps(response_json, indent=4, ensure_ascii=False))
    #返回结果
    rimg64 = cv2.imencode('.jpeg', img_array)[1]
    rimg64 = str(base64.b64encode(rimg64))[2:-1]
    result = {
        "rimg": rimg64,
        "sb": data,
        "jsontext": response_json
    }
    return jsonify(result)
# ############深度学习口罩识别
@ app.route('/sbimg', methods=["POST"])
def sbimgfun():
    #保存前端传来的图片
    img = request.files.get("imgfile")
    imgbyte = img.read()
    img_path = os.path.join(os.getcwd(), "deeplearn", "imgs", str(uuid.uuid1())+"."+img.filename.split(".")[-1])
    with open(img_path, "wb+") as f:
        f.write(imgbyte)
    #读取图片转换为灰度图
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #在灰度图中检测人脸并将信息保存到文件中
    face3 = faceCascade.detectMultiScale(gray)
    if len(face3) != 0:
        box = face3[0]
        newimg = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        cv2.imwrite(img_path, newimg)
    else:
        return ""
    #读取图片人脸信息并将增加维度
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    #使用模型预测得分
    predictions = Deep_MODEL.predict(img_array)
    # score = np.argmax(predictions[0])
    #组装返回结果
    # result = [{"name": i[0], "score":i[1]}
    #           for i in list(zip(DEEP_CLASS, score.numpy().tolist()))]
    result = [{"name": i[0], "score":f"{i[1]*100:.3f}%"}
              for i in list(zip(DEEP_CLASS, predictions[0]))]
    print(result)
    # print(str(i[0] for i in list(zip(DEEP_CLASS, predictions[0]))))
    # json_str=json.dumps(result, indent=4,ensure_ascii=False)
    # print(json_str)
    data = {
        "sb": result,
        "jsontext": result
    }
    return jsonify(data)
##############初始界面
@app.route('/')
def hello():
    return 'ok'
if __name__ == '__main__':
    # 启动应用
    app.run(host="127.0.0.1", port=5100, debug=True)
