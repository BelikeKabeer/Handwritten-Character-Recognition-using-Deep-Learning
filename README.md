# A-Z & 0-9 Handwritten Character Recognition

A web-based application for recognizing handwritten alphanumeric characters (A-Z and 0-9) using Convolutional Neural Networks (CNN). The application allows users to draw characters on a canvas or upload images for real-time prediction.

## Author
**Kabeer Munshipalkar**

## Features

- **Interactive Drawing Canvas**: Draw characters directly on a 400x400 pixel canvas
- **Image Upload**: Upload image files for character recognition
- **Real-time Predictions**: Get top 3 predictions with confidence scores
- **Mobile-Friendly**: Touch support for mobile devices
- **Clean UI**: Modern, responsive web interface

## Tech Stack

### Backend
- **Flask**: Web framework for Python
- **Keras/TensorFlow**: Deep learning framework for CNN model
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing
- **PIL (Pillow)**: Image handling

### Frontend
- **HTML5 Canvas**: For drawing interface
- **CSS3**: Styling and responsive design
- **Vanilla JavaScript**: Client-side functionality

### Model Architecture
- **CNN with Batch Normalization**
- **Input**: 28x28 grayscale images
- **Output**: 36 classes (A-Z: 26 + 0-9: 10)
- **Layers**:
  - Conv2D (32 filters, 5x5 kernel) + BatchNormalization
  - Conv2D (32 filters, 5x5 kernel) + BatchNormalization
  - MaxPooling2D (2x2) + Dropout (0.25)
  - Flatten + Dense (256 units)
  - Dense (36 units, softmax activation)

## Project Structure

```
project/
│
├── app.py                    # Flask application main file
├── cnn_architecture.py       # CNN model definition and training
├── data_preprocessing.py     # Data preprocessing utilities
├── templates/
│   └── index.html           # Main HTML template
├── static/
│   ├── script.js           # JavaScript functionality
│   └── style.css           # CSS styling
├── models/
│   └── best_val_loss_model.h5  # Trained model weights
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd handwritten-character-recognition
   ```

2. **Install required packages**
   ```bash
   pip install flask opencv-python keras tensorflow numpy pillow matplotlib pandas
   ```

3. **Create necessary directories**
   ```bash
   mkdir models templates static
   ```

4. **Place files in correct locations**:
   - Move `index.html` to `templates/` folder
   - Move `script.js` and `style.css` to `static/` folder
   - Ensure trained model `best_val_loss_model.h5` is in `models/` folder

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Using the Interface

#### Drawing Method:
1. Use your mouse or touch to draw a character on the canvas
2. Click "Predict" to get recognition results
3. Click "Clear" to erase and try again

#### Upload Method:
1. Click "Choose File" to select an image
2. Click "Upload & Predict" to process the image
3. View the top 3 predictions with confidence scores

## Model Training

### Data Sources
- **Digits (0-9)**: MNIST dataset
- **Letters (A-Z)**: A-Z Handwritten Data CSV (placeholder in current implementation)

### Training Process

1. **Preprocess data**:
   ```bash
   python data_preprocessing.py
   ```

2. **Train the model**:
   ```bash
   python cnn_architecture.py
   ```

The training script will:
- Create the CNN architecture
- Train on combined letter and digit data
- Save the best model weights based on validation loss
- Generate training history plots

### Model Performance
- **Input Shape**: (28, 28, 1) - grayscale images
- **Classes**: 36 (A-Z + 0-9)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## API Endpoints

### `/` (GET)
- Serves the main application interface

### `/predict` (POST)
- **Input**: JSON with base64 encoded image
- **Output**: JSON with top 3 predictions and confidence scores

### `/upload` (POST)
- **Input**: Form data with image file
- **Output**: JSON with top 3 predictions and confidence scores

## Image Processing Pipeline

1. **Input**: Canvas drawing or uploaded image
2. **Preprocessing**:
   - Convert to grayscale
   - Apply binary threshold
   - Find contours and extract character
   - Add padding and resize to square
   - Resize to 28x28 pixels
   - Invert colors (white background → black)
   - Normalize pixel values (0-1)
3. **Prediction**: Feed to CNN model
4. **Output**: Top 3 predictions with confidence scores

## Configuration

### Canvas Settings
- **Size**: 400x400 pixels
- **Line Width**: 15 pixels
- **Background**: White
- **Stroke Color**: Black

### Model Settings
- **Input Size**: 28x28 pixels
- **Batch Size**: 200 (training)
- **Epochs**: Configurable (default: 2 for demo)

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `best_val_loss_model.h5` exists in `models/` directory
   - Run training script to generate model weights

2. **Canvas not responding**
   - Check browser compatibility (modern browsers required)
   - Ensure JavaScript is enabled

3. **Poor prediction accuracy**
   - Draw characters clearly with adequate size
   - Ensure good contrast between character and background
   - Try uploading higher quality images

4. **Server errors**
   - Check all dependencies are installed
   - Verify file paths and directory structure
   - Check console for detailed error messages

## Future Improvements

- [ ] Add support for more character sets (lowercase letters, special characters)
- [ ] Implement data augmentation for better training
- [ ] Add confidence threshold filtering
- [ ] Support for multi-character recognition
- [ ] Mobile app version
- [ ] Real-time video stream recognition
- [ ] Custom model training interface

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset for digit training data
- Keras/TensorFlow community for deep learning frameworks
- OpenCV community for computer vision tools

---

**Note**: This project includes placeholder data for the A-Z character dataset. For production use, replace with actual handwritten letter dataset for better accuracy.
