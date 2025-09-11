import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer

# ---------------------------
# Patch InputLayer to ignore batch_shape during deserialization
# ---------------------------
_orig_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs.pop('batch_shape')  # remove batch_shape
    _orig_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

# ---------------------------
# Load the trained model
# ---------------------------
model = load_model(
    "retry/indian_food_classifier_mobilenetv5.keras",
    compile=False,
    custom_objects={'dtype': tf.float32}
)
# ---------------------------
# Food classes
# ---------------------------
food_classes = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa",
    "momos", "paani_puri", "pakode", "pav_bhaji", "pizza", "samosa"
]

# ---------------------------
# Nutritional information (example per serving)
# ---------------------------
nutritional_info = {
    "burger": {"Calories": 295, "Protein": 17, "Carbs": 33, "Fat": 12},
    "butter_naan": {"Calories": 310, "Protein": 8, "Carbs": 40, "Fat": 14},
    "chai": {"Calories": 120, "Protein": 3, "Carbs": 18, "Fat": 4},
    "chapati": {"Calories": 120, "Protein": 3, "Carbs": 20, "Fat": 3},
    "chole_bhature": {"Calories": 420, "Protein": 12, "Carbs": 50, "Fat": 18},
    "dal_makhani": {"Calories": 350, "Protein": 15, "Carbs": 25, "Fat": 20},
    "dhokla": {"Calories": 140, "Protein": 5, "Carbs": 20, "Fat": 5},
    "fried_rice": {"Calories": 250, "Protein": 6, "Carbs": 45, "Fat": 6},
    "idli": {"Calories": 58, "Protein": 2, "Carbs": 12, "Fat": 0.5},
    "jalebi": {"Calories": 150, "Protein": 2, "Carbs": 35, "Fat": 2},
    "kaathi_rolls": {"Calories": 300, "Protein": 12, "Carbs": 35, "Fat": 10},
    "kadai_paneer": {"Calories": 320, "Protein": 14, "Carbs": 12, "Fat": 24},
    "kulfi": {"Calories": 200, "Protein": 5, "Carbs": 20, "Fat": 12},
    "masala_dosa": {"Calories": 190, "Protein": 5, "Carbs": 35, "Fat": 5},
    "momos": {"Calories": 180, "Protein": 7, "Carbs": 30, "Fat": 4},
    "paani_puri": {"Calories": 150, "Protein": 4, "Carbs": 25, "Fat": 4},
    "pakode": {"Calories": 220, "Protein": 5, "Carbs": 20, "Fat": 14},
    "pav_bhaji": {"Calories": 350, "Protein": 10, "Carbs": 45, "Fat": 15},
    "pizza": {"Calories": 285, "Protein": 12, "Carbs": 36, "Fat": 10},
    "samosa": {"Calories": 150, "Protein": 3, "Carbs": 18, "Fat": 8}
}

# ---------------------------
# Sugar content per serving (grams)
# ---------------------------
sugar_content = {
    "burger": 5, "butter_naan": 2, "chai": 10, "chapati": 1,
    "chole_bhature": 5, "dal_makhani": 3, "dhokla": 2, "fried_rice": 2,
    "idli": 1, "jalebi": 20, "kaathi_rolls": 5, "kadai_paneer": 4,
    "kulfi": 15, "masala_dosa": 3, "momos": 2, "paani_puri": 3,
    "pakode": 1, "pav_bhaji": 5, "pizza": 4, "samosa": 2
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NutriFit", page_icon="🍲", layout="wide")
st.title("🥗 NutriFit - Indian Food Detection & Nutrition App")
st.markdown(
    "Upload an image of Indian food and get **predictions**, **nutritional info**, "
    "**health suggestions**, and **sugar check**!"
)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of Indian food.
2. Wait for the model to predict the food.
3. See the predicted food, nutrition table, pie chart, health suggestion, condition advice, and sugar info.
""")

# Health condition selection
st.sidebar.subheader("Select Your Health Condition")
health_condition = st.sidebar.selectbox(
    "Choose a condition:",
    ["None", "Diabetes", "High Cholesterol", "Weight Loss Goal", "Low Sodium Diet"]
)

# Sugar check toggle
check_sugar = st.sidebar.checkbox("Check Sugar Content / Diabetes Suitability")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    max_width = 400  # optimized image width

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict food
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    predicted_food = food_classes[class_idx]
    confidence = prediction[0][class_idx] * 100

    # Display in columns
    st.subheader("Prediction Result")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(img, caption="Uploaded Image", width=max_width)
    with col2:
        st.success(f"**Food:** {predicted_food}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    # Nutritional info
    if predicted_food in nutritional_info:
        st.subheader("Nutritional Information (per serving)")
        nutrient_data = nutritional_info[predicted_food]
        st.table(pd.DataFrame(nutrient_data, index=[0]))

        # Pie chart
        macros = {k:v for k,v in nutrient_data.items() if k in ["Protein","Carbs","Fat"]}
        fig, ax = plt.subplots()
        ax.pie(macros.values(), labels=macros.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.subheader("Macronutrient Distribution")
        st.pyplot(fig)

        # General health suggestion
        cal = nutrient_data["Calories"]
        fat = nutrient_data["Fat"]
        if cal > 350 or fat > 15:
            suggestion = "⚠️ High-calorie / high-fat food. Consume in moderation."
        elif fat < 5 and cal < 200:
            suggestion = "✅ Low-fat & low-calorie. Healthy choice!"
        else:
            suggestion = "⚡ Balanced food. Enjoy in moderation."
        st.subheader("Health Suggestion")
        st.info(suggestion)

        # Condition-based advice
        condition_advice = ""
        if health_condition == "Diabetes" and macros["Carbs"] > 30:
            condition_advice = "⚠️ High carb content may not be suitable for diabetes."
        elif health_condition == "High Cholesterol" and fat > 15:
            condition_advice = "⚠️ High-fat food. Not ideal for high cholesterol."
        elif health_condition == "Weight Loss Goal" and cal > 300:
            condition_advice = "⚠️ High-calorie food. Consider lower-calorie alternatives."
        elif health_condition == "Low Sodium Diet":
            condition_advice = "⚡ Check preparation; may contain moderate salt."
        else:
            condition_advice = "✅ Suitable for your selected condition."
        if health_condition != "None":
            st.subheader(f"Condition-based Advice for {health_condition}")
            st.info(condition_advice)

        # Sugar check
