import pickle
import gzip

# 載入model
with gzip,open('./model/xgboost-iris.pgz','r') as f:
    xgboostModel = pickle.load(f)

def predict(input):
    pred=xgboostModel.predict(input)[0]
    print(pred)
    return pred