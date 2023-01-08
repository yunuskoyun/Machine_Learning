import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


html_temp = """
	<div style ="background-color:#3E3F31; padding:5px">
	<h1 style ="color:#B8EAFF; text-align:center; ">Streamlit Car Price Prediction Project</h1>
	</div>
	"""

st.markdown(html_temp, unsafe_allow_html = True)


image = Image.open("car.jpg")
st.image(image, use_column_width=True)

st.empty()


st.header("Welcome!")
st.markdown("Please provide your car information on the left sidebar and than click the _Predict the Car Price_ button.")
st.markdown("After click the button, you will the see your car price as predictably.")

df = pd.read_csv('df_with_feature_imp.csv')


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
    st.success(f"Predicted Price of The Car: **€ {result}**")
    if df_pred.make_model.str.startswith("Audi").any():
    	st.image(Image.open("audi.png"), output_format="PNG")
    elif df_pred.make_model.str.startswith("Opel").any():
        st.image(Image.open("opel.png"))
    elif df_pred.make_model.str.startswith("Rena").any():
        st.image(Image.open("renault.png"))
except NameError:
    st.write("Please **Predict** button to display the result!")