import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import model_from_json


app = Flask(__name__,template_folder='templates')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features).reshape(1,1,4)]
    prediction = loaded_model.predict(final_features)

    output = prediction[0]#round(prediction[0], 2)

    return render_template('index.html', prediction_text='Closing price should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)