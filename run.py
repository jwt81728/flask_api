import numpy as np
import model
from flask import Flask,request,jsonify
from flask_cors import CORS

app =Flask(__name__)
CORS(app)

@app.route('/test')
def index():
    return 'hello!!'

@app.route('/predict',methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=insertValues['sepalLengthCm']
    x2=insertValues['sepalWidthCm']
    x3=insertValues['pepalLengthCm']
    x4=insertValues['pepalWidthCm']
    input = np.array([[x1,x2,x3,x4]])

    result = model.predict(input)

    print(input)

    return jsonify({'return': str(result)})
    

if __name__ == '__main__':
    app.run(host='localhost',port=3000,debug=True)