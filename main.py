import streamlit as st
from fastai.vision.all import *
import pathlib
import tempfile

# Set the title of the app
st.title("Mongolian Food Classifier")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = pathlib.Path('mongolian_food_classifier.pkl')  # Update this path
    learn = load_learner(model_path)
    return learn

learn = load_model()

# Function to classify the image
def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)
    return pred, probs[pred_idx]

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload an image of Mongolian food...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Open the image and classify it
    img = PILImage.create(tmp_file_path)
    pred, confidence = classify_image(img)

    # Display the result
    st.write(f"Prediction: {pred}")
    st.write(f"Confidence: {confidence:.4f}")

    # Clean up the temporary file
    os.unlink(tmp_file_path)