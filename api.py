from flask import Flask, request, jsonify
import joblib  # ✅ use joblib, not pickle

# Load the model and vectorizer using joblib
model = joblib.load('nb_model.pkl')           # or whichever model you prefer
vectorizer = joblib.load('vectorizer.pkl')    # TF-IDF vectorizer

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Spam Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']

        # Transform and predict
        transformed_msg = vectorizer.transform([message])
        prediction = model.predict(transformed_msg)[0]

        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
