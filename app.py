import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env (if using)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

# -----------------------
# Plant Disease Prediction
# -----------------------

# Load the trained plant disease model (update the path as needed)
plant_model = tf.keras.models.load_model("//Users//krishnareddy//Downloads//GOOGLE DEVS//farmer bot//PlantFruitapp//backend//models//plant_health_model.h5", compile=False)

# Define plant disease labels (38 classes; update these as per your training)
labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Define health suggestions for specific classes
health_suggestions = {
    'Apple___Apple_scab': "Prune infected leaves, apply fungicide, and ensure proper air circulation around trees.",
    'Apple___Black_rot': "Remove infected fruit and branches, use fungicide, and keep the orchard clean.",
    'Apple___Cedar_apple_rust': "Remove nearby cedar trees, use rust-resistant varieties, and apply fungicide.",
    'Apple___healthy': "No issues detected. Maintain regular watering and fertilization.",
    'Blueberry___healthy': "No issues detected. Ensure soil acidity is appropriate and provide adequate water.",
    'Cherry_(including_sour)___Powdery_mildew': "Use fungicide, avoid overhead watering, and prune affected areas.",
    'Cherry_(including_sour)___healthy': "No issues detected. Continue with regular care and monitoring.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use fungicide, rotate crops, and remove infected debris.",
    'Corn_(maize)___Common_rust_': "Apply fungicide and use rust-resistant varieties.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids, rotate crops, and apply fungicide if necessary.",
    'Corn_(maize)___healthy': "No issues detected. Maintain regular care for optimal growth.",
    'Grape___Black_rot': "Remove and destroy infected leaves and fruit, and apply fungicide.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines, ensure proper irrigation, and avoid over-fertilizing.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicide, prune affected areas, and ensure good air circulation.",
    'Grape___healthy': "No issues detected. Regular monitoring and care are recommended.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees, use resistant rootstocks, and control insect vectors.",
    'Peach___Bacterial_spot': "Apply copper-based bactericides, prune infected areas, and avoid overhead irrigation.",
    'Peach___healthy': "No issues detected. Continue with regular watering and care.",
    'Pepper,_bell___Bacterial_spot': "Use copper-based fungicide, rotate crops, and remove infected plants.",
    'Pepper,_bell___healthy': "No issues detected. Maintain consistent watering and nutrient supply.",
    'Potato___Early_blight': "Use fungicide, rotate crops, and remove plant debris after harvest.",
    'Potato___Late_blight': "Apply fungicide and practice crop rotation to minimize disease spread.",
    'Potato___healthy': "No issues detected. Ensure proper soil health and pest control.",
    'Raspberry___healthy': "No issues detected. Keep monitoring for pests and diseases.",
    'Soybean___healthy': "No issues detected. Regular crop rotation and pest monitoring are essential.",
    'Squash___Powdery_mildew': "Use fungicide, water at the base of the plant, and ensure good air circulation.",
    'Strawberry___Leaf_scorch': "Remove infected leaves, avoid overhead watering, and apply fungicide.",
    'Strawberry___healthy': "No issues detected. Continue regular monitoring and care.",
    'Tomato___Bacterial_spot': "Use copper-based sprays, avoid wetting leaves, and remove infected plants.",
    'Tomato___Early_blight': "Apply fungicide, remove affected leaves, and mulch plants to prevent soil splashing.",
    'Tomato___Late_blight': "Remove infected plants immediately, use fungicide, and avoid overhead irrigation.",
    'Tomato___Leaf_Mold': "Improve air circulation, use resistant varieties, and apply fungicide if necessary.",
    'Tomato___Septoria_leaf_spot': "Prune affected leaves, avoid overhead watering, and use fungicide.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use insecticidal soap, keep plants well-watered, and encourage natural predators.",
    'Tomato___Target_Spot': "Apply fungicide and remove diseased leaves to prevent spread.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control whiteflies with insecticidal soap.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants, disinfect tools, and avoid smoking near plants.",
    'Tomato___healthy': "No issues detected. Continue regular care and monitoring."
}

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    img = Image.open(image).convert("RGB")  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    img_array = preprocess_image(image)

    try:
        predictions = plant_model.predict(img_array)
        if predictions.shape[1] != len(labels):
            return jsonify({"error": "Mismatch between model output and labels"}), 500

        predicted_class = labels[np.argmax(predictions[0])]  # Highest probability class
        confidence = float(np.max(predictions[0]))  # Confidence score
        suggestion = health_suggestions.get(predicted_class, "No specific advice available.")

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence * 1000,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -----------------------
# Chatbot using Gemini API
# -----------------------

# Configure Gemini API using your API key from the environment
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
else:
    print("GEMINI_API_KEY loaded, length:", len(gemini_key))
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot requests."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    print("User message for chatbot:", user_message)  # Debug print
    try:
        # Generate response using Gemini API
        response = gemini_model.generate_content(user_message)
        print("Gemini response:", response)  # Debug print
        return jsonify({"response": response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This prints the full stack trace to the console
        return jsonify({"error": f"Chatbot failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
