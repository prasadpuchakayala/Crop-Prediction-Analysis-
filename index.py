from flask import Flask, render_template,request

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crop_prediction')
def crop_prediction():
    return render_template('prediction.html')


@app.route('/prediction',methods=["POST","GET"])
def prediction():

    temp = request.form.get("temp")

    humd = request.form.get("humd")

    ph = request.form.get("ph")

    rain = request.form.get("rain")

    df = pd.read_csv("crop_dataset.csv")

    y_train = df["label"]
    del df["label"]

    x_train = df

    clf_knn = KNeighborsClassifier(n_neighbors=3)

    # Training the ML model
    clf_knn.fit(x_train, y_train)

    x_test = [[float(temp),float(humd), float(ph),float(rain)]]

    pre_res = clf_knn.predict(x_test)

    return  "Crop Prediction Results="+pre_res[0]






if __name__ == '__main__':
    app.run(host="localhost", port=1166, debug=True)