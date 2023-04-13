from flask import Flask, request, jsonify
from GTRS2 import TWC_GTRS_MODEL
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def search():

    data = request.get_json()
    my_df =  pd.json_normalize(data)
    print(my_df.head())
    model = TWC_GTRS_MODEL()
    predictions = model.predict(my_df)
    results = predictions
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
    