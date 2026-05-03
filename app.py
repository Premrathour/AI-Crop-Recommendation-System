from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model/crop_model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get input values
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create prediction array
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Predict crop
    prediction = model.predict(data)

    # Confidence score
    probabilities = model.predict_proba(data)
    confidence = max(probabilities[0]) * 100

    result = prediction[0]

    # Fertilizer suggestions
    fertilizer_dict = {
        "rice": "Use Nitrogen-rich fertilizer",
        "wheat": "Use NPK fertilizer",
        "maize": "Use Phosphorus fertilizer",
        "cotton": "Use Potassium fertilizer",
        "banana": "Use Organic Compost",
        "mango": "Use Farmyard Manure",
        "coffee": "Use Nitrogen + Potassium fertilizer"
    }

    fertilizer = fertilizer_dict.get(result.lower(), "General organic fertilizer")

    # Crop image
    image_file = result.lower() + ".jpg"

    return render_template(
        'index.html',
        prediction_text=f'Recommended Crop: {result}',
        confidence=f'Confidence: {confidence:.2f}%',
        crop_image=image_file,
        fertilizer=f'Fertilizer Suggestion: {fertilizer}'
    )


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)