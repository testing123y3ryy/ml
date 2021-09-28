from flask import Flask , render_template , request, jsonify , redirect , url_for
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from numpy.lib.index_tricks import AxisConcatenator
app = Flask(__name__)

images = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = images


mode = pickle.load(open("mode.pkl", "rb"))
ohe_gender = pickle.load(open("ohe_gender.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    df = pd.read_csv("weight-height.csv")
    print(df.head())
    float_features = []
    if request.method == 'POST':
        height=float(request.form["height"])
        gender=request.form["gender"]
        float_features = [height , gender]
        
    print(float_features)
    test_input = np.array(float_features, dtype=object).reshape(1,2)
    

    test_input_gen = ohe_gender.transform(test_input[:,1].reshape(1,1))
    print("test_input_gen",test_input_gen , test_input_gen.shape)

    test_input_weight = [float(float_features[0])]
    print("test_input_weight",test_input_weight)
    test_input_weight = np.array(test_input_weight, dtype=object).reshape(1,1)
    print("test_input_weight",test_input_weight.shape)

    test_input_transformed = np.concatenate((test_input_weight, test_input_gen), axis=1)
    print("test_input_weight",test_input_transformed.shape)
    print(test_input_transformed)
    
    predictions = mode.predict(test_input_transformed)
    picture = sns.scatterplot(data = df , x="Height" , y="Weight" , hue="Gender")
    fig = picture.get_figure()
    fig.savefig('static/images/3.png')
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],"3.png")
    print(full_filename)

    return render_template("index.html", pic=full_filename ,  prediction_text = "The weight is {}".format(predictions))
    
if __name__ == "__main__":
    app.run(debug=True)