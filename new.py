from flask import Flask, render_template,request

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math


from sklearn.model_selection import train_test_split

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

#model
    df = pd.read_csv("Raini_2004-2006_10Min.csv")

    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = pd.to_datetime(df['Date Time']).dt.month

    df['Day'] = pd.to_datetime(df['Date Time']).dt.day

    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['Minute'] = pd.to_datetime(df['Time']).dt.minute
    df['Second'] = pd.to_datetime(df['Time']).dt.second

    df.drop(['Date Time', 'Time'], axis=1, inplace=True)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numericals = ['Temperature', 'Relative Humidity', 'Pressure', 'Wind speed', 'Wind direction', 'Rainfall',
                  'Snowfall', 'Snow depth', 'Short-wave irradiation', 'Day', 'Month', 'Hour', 'Minute', 'Second']
    df[numericals] = scaler.fit_transform(data[numericals])

    y = df['Short-wave irradiation']
    y

    del df["Short-wave irradiation"]

    x = df
    x

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)

    # Regressor model
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Prediction result
    LR_pred = regressor.predict(X_test)

    '''y_train = df["label"]
    del df["label"]

    x_train = df

    clf_knn = KNeighborsClassifier(n_neighbors=3)

    # Training the ML model
    clf_knn.fit(x_train, y_train)

    x_test = [[float(temp),float(humd), float(ph),float(rain)]]

    pre_res = clf_knn.predict(x_test)'''

    return  "Crop Prediction Results="+LR_pred[0]






if __name__ == '__main__':
    app.run(host="localhost", port=1166, debug=True)