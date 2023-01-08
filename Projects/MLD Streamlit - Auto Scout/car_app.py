import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

html_temp = """
	<div style ="background-color:#3d2fd6; padding:13px">
	<h1 style ="color:#f0f0f5; text-align:center; ">Streamlit Iris Flower Classifier </h1>
	</div>
	"""

st.markdown(html_temp, unsafe_allow_html = True)

image = Image.open("car.jpg")
st.image(image, use_column_width=True)



st.header("_Car Price Prediction_")
df = pd.read_csv('df_with_feature_imp.csv')
st.write(df.head())

model = pickle.load(open("final_model", "rb"))


st.sidebar.title("_Please Select the Car's Model and Enter Details to Predict the Price_")
car_model = st.sidebar.selectbox('Select the Car Brand-Model', (df.make_model.unique()))
hp_kw = st.sidebar.number_input("Car Machine Power (hp_kW)", 40, 239, 90, 1)
km = st.sidebar.number_input("Made Road by Car (KiloMeter) ", 0, 317000, 33000, 1)
age = st.sidebar.slider("The Car's Age", 0, 3, 1, 1)
gear_type = st.sidebar.selectbox("Select the Car's Gear Type", (df.Gearing_Type.unique()))
gears = st.sidebar.slider("The Car Gears", 5, 8, 6, 1)
type = st.sidebar.selectbox("Select the Car's Used Type", (df.Type.unique()))
package = st.sidebar.selectbox("Select the Car's Package", (df.Safety_Security_Package.unique()))




pred_dict = {
    "make_model":car_model,
    "hp_kW":hp_kw,
    "km":km,
    "age":age,
    "Gearing_Type":gear_type,
    "Gears":gears,
    "Type":type,
    'Safety_Security_Package':package
}

df_pred = pd.DataFrame.from_dict([pred_dict])

def prediction(model, df_pred):

	prediction = model.predict(df_pred)
	return prediction


if st.button("Predict the Car Price"):
    result = prediction(model, df_pred)[0]

try:
    st.success(f"Predicted Price of The Car: **{result}**")
    if df_pred.make_model.str.startswith("Audi"):
    	st.image(Image.open("audi.png"))
    elif df_pred.make_model.str.startswith("Opel"):
        st.image(Image.open("opel.png"))
    elif df_pred.make_model.str.startswith("Rena"):
        st.image(Image.open("renault.png"))
except NameError:
    st.write("Please **Predict** button to display the result!")