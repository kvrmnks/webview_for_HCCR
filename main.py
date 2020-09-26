import flask
from flask import request
import urllib
import io
import numpy as np
from PIL import Image
import model
import data

app = flask.Flask(__name__)


@app.route('/catch', methods=['POST'])
def catch():
    response = urllib.request.urlopen(request.form['id'])
    buf = io.BytesIO()
    buf.write(response.file.read())
    arr = np.asarray(Image.open(buf))
    # arr 为图片像素点数组
    ans = model.eval(arr)
    # data.get_ans(x) 为通过label x 找到对应的汉字

    # 下面的返回值为一个字符串，表示你希望在网页的识别结果栏显示什么东西
    return data.get_ans(ans[0]) + ' 左右: ' + data.get_ans(ans[1]) + ' 上下: ' + data.get_ans(ans[2])


@app.route('/')
def index():
    return flask.render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0')
