from flask import Flask, request, jsonify
from GTRS import TWC_GTRS_MODEL
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def search():

# Load the GTRS_Final_Model file
    model = TWC_GTRS_MODEL()

    # # Create a sample DataFrame for prediction
    df = pd.read_csv('dataset2.csv')
    
    # # Make predictions using the model
    predictions = model.predict(df)
    

    # print(predictions)
    
    # query = request.args.get('q')
    results = predictions # your search logic here, e.g. querying a database or calling an API
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
    