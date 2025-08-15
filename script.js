const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');

let isDrawing = false;

// Set up canvas
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = '#000';

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile
canvas.addEventListener('touchstart', handleTouch);
canvas.addEventListener('touchmove', handleTouch);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
    }
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

// Clear canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearPredictions();
    imageUpload.value = '';
});

// Predict character
predictBtn.addEventListener('click', async () => {
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictions(result.predictions);
        } else {
            console.error('Prediction failed:', result.error);
            alert('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error occurred. Please try again.');
    }
});

function displayPredictions(predictions) {
    const predictionItems = document.querySelectorAll('.prediction-item');
    
    predictionItems.forEach((item, index) => {
        const label = item.querySelector('.label');
        const accuracy = item.querySelector('.accuracy');
        
        if (predictions[index]) {
            label.textContent = predictions[index].label;
            accuracy.textContent = predictions[index].accuracy + '%';
        } else {
            label.textContent = '-';
            accuracy.textContent = '-';
        }
    });
}

function clearPredictions() {
    const predictionItems = document.querySelectorAll('.prediction-item');
    predictionItems.forEach(item => {
        item.querySelector('.label').textContent = '-';
        item.querySelector('.accuracy').textContent = '-';
    });
}

// Upload functionality
const imageUpload = document.getElementById('imageUpload');
const uploadBtn = document.getElementById('uploadBtn');

uploadBtn.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) {
        alert('Please select an image first.');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictions(result.predictions);
        } else {
            console.error('Upload failed:', result.error);
            alert('Upload failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error occurred. Please try again.');
    }
});