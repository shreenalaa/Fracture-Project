#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as lite


# In[ ]:


# Initialize Flask application
app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = lite.Interpreter(model_path='xray_class_weights.best.hdf5')
interpreter.allocate_tensors()

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Add any preprocessing steps here, such as normalization
    return input_data

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the request
        input_data = request.json.get('data')

        # Preprocess the input data
        input_data = preprocess_input(input_data)

        # Perform inference using the TensorFlow Lite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Post-process the output data if needed
        # For example, convert softmax output to predicted class

        # Return the prediction as JSON response
        return jsonify({'prediction': output_data.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

