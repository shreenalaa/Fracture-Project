#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import torch
from torchvision import transforms as T
from PIL import Image


# In[ ]:


# Initialize Flask application
app = Flask(__name__)

# Load your trained model
model = torch.load('fracture_best_model.pth')
model.eval()

# Define transformations for input images
im_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

# Define a route for predicting bone fracture
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get image data from the request
        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image = tfs(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()

        # Map the predicted class index to class name
        class_names = {0: 'fractured', 1: 'not fractured'}
        predicted_class_name = class_names[predicted_class]

        # Return the prediction as JSON response
        return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)

