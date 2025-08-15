from flask import Flask, request, jsonify, render_template,session,redirect,url_for
import pickle
import numpy as np
from auth import auth_bp


with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
feature_means = saved['feature_means']
feature_order = saved['feature_order']

app = Flask(__name__)
app.secret_key = "supersecretkey"


app.register_blueprint(auth_bp)

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))  
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        temp = float(data['temp'])
        humidity = float(data['humidity'])
        oxygen = float(data['oxygen'])

        input_features = [feature_means[feat] for feat in feature_order]
        input_features[feature_order.index('temp')] = temp
        input_features[feature_order.index('RH')] = humidity
        input_features[feature_order.index('oxygen')] = oxygen

        input_array = np.array([input_features])
        prob = model.predict_proba(input_array)[0][1]
        pred = "🔥 Fire likely" if prob >= 0.5 else "✅ No fire risk"

        return jsonify({
            'prediction': pred,
            'confidence': round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
