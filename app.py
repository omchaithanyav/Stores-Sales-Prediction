from flask import Flask, render_template, request, jsonify
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


        if Item_Fat_Content == "Low Fat" or Item_Fat_Content == "low fat" or Item_Fat_Content == "low":
                Item_Fat_Content = 0
        elif Item_Fat_Content == "Regular" or Item_Fat_Content == "regular":
                Item_Fat_Content = 1
        else:
                Item_Fat_Content = 0


        if Item_Type == "edible":
                Item_Type = 0
        elif Item_Type == "non-edible" or Item_Type == "non edible":
                Item_Type = 1
        else:
                Item_Type = 0


        if Outlet_Size == "Medium" or Outlet_Size == "medium":
                Outlet_Size = 1
        elif Outlet_Size == "Small" or Outlet_Size == "small":
                Outlet_Size = 2
        elif Outlet_Size == "High" or Outlet_Size == "high":
                Outlet_Size = 0
        else:
                Outlet_Size = 2


        if Outlet_Location_Type == "Tier 1" or Outlet_Location_Type == "tier1" or Outlet_Location_Type == "tier 1":
                Outlet_Location_Type = 0
        elif Outlet_Location_Type == "Tier 2" or Outlet_Location_Type == "tier 2" or Outlet_Location_Type == "tier2":
                Outlet_Location_Type = 1
        elif Outlet_Location_Type == "Tier 3" or Outlet_Location_Type == "tier 3" or Outlet_Location_Type == "tier3":
                Outlet_Location_Type = 2
        else:
                Outlet_Location_Type = 0


        if Outlet_Type == "Supermarket Type1" or Outlet_Type == "supermarket type1":
                Outlet_Type = 1
        elif Outlet_Type == "Supermarket Type2" or Outlet_Type == "supermarket type2":
                Outlet_Type = 2
        elif Outlet_Type == "Supermarket Type3" or Outlet_Type == "supermarket type3":
                Outlet_Type = 3
        elif Outlet_Type == "Grocery Store" or Outlet_Type == "grocery store":
                Outlet_Type = 0
        else:
                Outlet_Type = 1

        inputs = np.array([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type]).reshape(1, -1)

        # prediction
        model = pickle.load(open('models/gb.pkl', 'rb'))
        prediction = model.predict(inputs)
        prediction = float(prediction)
        return render_template('result.html', results='The outlet sales for the given item should be: {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True, port=5544)