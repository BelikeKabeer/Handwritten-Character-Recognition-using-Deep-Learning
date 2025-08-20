from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import base64
import io
from PIL import Image

app = Flask(__name__)

def load_model(path):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(36, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights(path)
    return model

def predict_character(model, image):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (character)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Crop character with padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)
        
        cropped = gray[y1:y2, x1:x2]
        
        # Resize to square
        size = max(cropped.shape)
        square = np.full((size, size), 255, dtype=np.uint8)
        start_y = (size - cropped.shape[0]) // 2
        start_x = (size - cropped.shape[1]) // 2
        square[start_y:start_y+cropped.shape[0], start_x:start_x+cropped.shape[1]] = cropped
        
        # Resize to 28x28
        resized = cv2.resize(square, (28, 28))
    else:
        resized = cv2.resize(gray, (28, 28))
    
    # Invert colors
    resized = 255 - resized
    
    # Normalize
    resized = resized / 255.0
    
    # Reshape for model
    resized = np.reshape(resized, (1, 28, 28, 1))
    
    # Predict
    prediction = model.predict(resized, verbose=0)
    
    best_predictions = []
    for i in range(3):
        max_i = np.argmax(prediction[0])
        acc = round(prediction[0][max_i] * 100, 1)
        if acc > 0:
            label = labels[max_i]
            best_predictions.append({"label": label, "accuracy": acc})
            prediction[0][max_i] = 0
        else:
            break
    
    return best_predictions

# Load model on startup
model = load_model("model/best_val_loss_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert RGBA to RGB if needed
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Convert to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Get predictions
        predictions = predict_character(model, image_np)
        
        return jsonify({"success": True, "predictions": predictions})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['image']
        
        # Read image
        image = Image.open(file.stream)
        image_np = np.array(image)
        
        # Convert to BGR for OpenCV
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # Get predictions
        predictions = predict_character(model, image_np)
        
        return jsonify({"success": True, "predictions": predictions})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
