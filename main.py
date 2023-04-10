from flask import Flask, request, jsonify
from GTRS import TWC_GTRS_MODEL
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def search():

    data = request.get_json()
    gender = data['gender']
    print(data,gender)
    my_df =  pd.json_normalize(data)
    print(my_df)

    model = TWC_GTRS_MODEL()
    df = pd.read_csv('dataset2.csv')
    predictions = model.predict(df)
    results = predictions
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
    