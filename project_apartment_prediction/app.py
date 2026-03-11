import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from math import radians, cos, sin, asin, sqrt

# --- Load model ---
cwd = os.getcwd()
model_filename = "apartment_price_model.pkl"
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

# --- Load data ---
df_bfs = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs["tax_income"] = df_bfs["tax_income"].str.replace("'", "").astype(float)

df_apt = pd.read_csv("apartments_data_enriched_with_new_features.csv", sep=",", encoding="utf-8")
df_apt = df_apt.dropna().drop_duplicates()

# --- Precompute lookups ---
town_to_bfs      = df_apt.groupby("town")["bfs_number"].first().to_dict()
town_coords      = df_apt.groupby("town")[["lat", "lon"]].mean().to_dict("index")
town_to_postal   = df_apt.groupby("town")["postalcode"].agg(lambda x: x.value_counts().index[0]).to_dict()
avg_price_lookup = df_apt.groupby(["postalcode", "rooms"])["price"].mean().to_dict()
fallback_price   = df_apt["price"].mean()
towns            = sorted(town_to_bfs.keys())

# --- Helpers ---
def haversine_km(lat, lon, zh_lat=47.3782, zh_lon=8.5403):
    R = 6371
    lat, lon, zh_lat, zh_lon = map(radians, [lat, lon, zh_lat, zh_lon])
    dlat, dlon = zh_lat - lat, zh_lon - lon
    a = sin(dlat/2)**2 + cos(lat) * cos(zh_lat) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def area_cat(area):
    if area < 60:   return 0
    elif area <= 90: return 1
    else:            return 2

# --- Prediction ---
def predict_apartment(rooms, area, town, luxurious, furnished, temporary):
    bfs_number  = town_to_bfs[town]
    bfs_row     = df_bfs[df_bfs["bfs_number"] == bfs_number].iloc[0]
    coords      = town_coords[town]
    dist_to_zhb = haversine_km(coords["lat"], coords["lon"])
    postal      = town_to_postal[town]
    avg_price   = avg_price_lookup.get((postal, rooms), fallback_price)

    features = pd.DataFrame([{
        "rooms":                       rooms,
        "area":                        area,
        "pop":                         bfs_row["pop"],
        "pop_dens":                    bfs_row["pop_dens"],
        "frg_pct":                     bfs_row["frg_pct"],
        "emp":                         bfs_row["emp"],
        "tax_income":                  bfs_row["tax_income"],
        "room_per_m2":                 rooms / area,
        "luxurious":                   int(luxurious),
        "temporary":                   int(temporary),
        "furnished":                   int(furnished),
        "area_cat_ecoded":             area_cat(area),
        "zurich_city":                 int(bfs_number == 261),
        "avg_price_postal_rooms_area": avg_price,
        "dist_to_zhb":                 dist_to_zhb
    }])

    prediction = model.predict(features)[0]
    return f"CHF {round(prediction):,} / month"

# --- Gradio UI ---
demo = gr.Interface(
    fn=predict_apartment,
    inputs=[
        gr.Number(label="Rooms",     value=3.5),
        gr.Number(label="Area (m²)", value=80),
        gr.Dropdown(choices=towns, label="Town"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Furnished"),
        gr.Checkbox(label="Temporary rental"),
    ],
    outputs="text",
    examples=[
        [4.5, 120, "Zürich",     False, False, False],
        [3.5,  75, "Winterthur", False, False, False],
        [2.5,  55, "Dietlikon",  False, True,  False],
    ],
    title="Apartment Price Prediction – Canton of Zurich",
    description="Enter the apartment details to get a predicted monthly rent in CHF."
)

demo.launch()
