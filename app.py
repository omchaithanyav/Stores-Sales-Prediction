from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

        Item_Weight=float(request.form['Item_Weight'])
        Item_Fat_Content=str(request.form['Item_Fat_Content'])
        Item_Visibility=float(request.form['Item_Visibility'])
        Item_Type=str(request.form['Item_Type'])
        Item_MRP=float(request.form['Item_MRP'])
        Outlet_Size=str(request.form['Outlet_Size'])
        Outlet_Location_Type=str(request.form['Outlet_Location_Type'])
        Outlet_Type=str(request.form['Outlet_Type'])


        if Item_Fat_Content == "Low Fat":
                Item_Fat_Content = 0
        elif Item_Fat_Content == "Regular":
                Item_Fat_Content = 1


        if Item_Type == "edible":
                Item_Type = 0
        elif Item_Type == "non-edible":
                Item_Type = 1


        if Outlet_Size == "Medium":
                Outlet_Size = 1
        elif Outlet_Size == "Small":
                Outlet_Size = 2
        elif Outlet_Size == "High":
                Outlet_Size = 0


        if Outlet_Location_Type == "Tier 1":
                Outlet_Location_Type = 0
        elif Outlet_Location_Type == "Tier 2":
                Outlet_Location_Type = 1
        elif Outlet_Location_Type == "Tier 3":
                Outlet_Location_Type = 2


        if Outlet_Type == "Supermarket Type1":
                Outlet_Type = 1
        elif Outlet_Type == "Supermarket Type2":
                Outlet_Type = 2
        elif Outlet_Type == "Supermarket Type3":
                Outlet_Type = 3
        elif Outlet_Type == "Grocery Store":
                Outlet_Type = 0

        inputs = np.array([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type]).reshape(1, -1)

        # prediction
        model = pickle.load(open('models/gb.pkl', 'rb'))
        prediction = model.predict(inputs)
        prediction = float(prediction)
        return render_template('result.html', results='The outlet sales for the given item should be: {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)