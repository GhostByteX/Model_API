from flask import Flask, request, jsonify
from GTRS import TWC_GTRS_MODEL
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Welcome to my Flask app!'


@app.route('/search', methods=['GET'])
def search():

# Load the GTRS_Final_Model file
    model = joblib.load('GTRS_Final_Model.joblib2')

    # Create a sample DataFrame for prediction
    df = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6], 'feat3': [7, 8, 9]})
    
    # Make predictions using the model
    predictions = model.predict(df)

    print(predictions)
    
    query = request.args.get('q')
    results = predictions # your search logic here, e.g. querying a database or calling an API

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
    