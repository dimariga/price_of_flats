link for requirements.txt
https://github.com/dimariga/price_of_flats/edit/main/requirements.txt

import joblib
import streamlit as st
import pandas as pd
import json
st.header('Цена квартиры')

PATH_DATA = "flats2.csv"
PATH_UNIQUE_VALUES = "unique_values.json"
PATH_MODEL = "lr_pipeline.sav"

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data = data.sample(5000)
    return data

@st.cache_data
def load_model(PATH_MODEL):
    model = joblib.load(PATH_MODEL)
    return model

df = load_data(PATH_DATA)
st.write(df[:4])

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)
city = st.sidebar.selectbox("Город", (dict_unique["city"]))
house_wall_type = st.sidebar.selectbox("Тип стен", (dict_unique["house_wall_type"]))
renovation = st.sidebar.selectbox("Ремонт", (dict_unique["renovation"]))


area = st.sidebar.slider(
    "Площадь",
    min_value=min(dict_unique["area"]),
    max_value=max(dict_unique["area"])
)

rooms = st.sidebar.slider(
    "Количество комнат",
    min_value=min(dict_unique["rooms"]),
    max_value=max(dict_unique["rooms"])
)

floor = st.sidebar.slider(
    "Этаж",
    min_value=min(dict_unique["floor"]),
    max_value=max(dict_unique["floor"])
)

house_floors = st.sidebar.slider(
    "Этажность дома",
    min_value=min(dict_unique["house_floors"]),
    max_value=max(dict_unique["house_floors"])
)
kitchen_area = st.sidebar.slider(
    "Площадь кухни",
    min_value=min(dict_unique["kitchen_area"]),
    max_value=max(dict_unique["kitchen_area"])
)

balconies = st.sidebar.slider(
    "Количество балконов",
    min_value=min(dict_unique["balconies"]),
    max_value=max(dict_unique["balconies"])
)

lifts = st.sidebar.slider(
    "Количество лифтов",
    min_value=min(dict_unique["lifts"]),
    max_value=max(dict_unique["lifts"])
)

dict_data = {
    "city": city,
    "house_wall_type": house_wall_type,
    "renovation": renovation,
    "area": area,
    "rooms": rooms,
    "floor": floor,
    "house_floors": house_floors,
    "kitchen_area": kitchen_area,
    "balconies": balconies,
    "lifts": lifts,
}
data_predict = pd.DataFrame([dict_data])
#data_predict.to_csv("123")
model = joblib.load(PATH_MODEL)
button = st.button("Предварительная цена")
if button:
    result = model.predict(data_predict)[0]
    st.write(result)
