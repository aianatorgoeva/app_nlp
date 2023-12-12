import streamlit as st
from app import nlp_1, nlp_2, nlp_3
import gdown
import zipfile
import os

# Download models if not already present
if not os.path.exists("models/subset-roberta/config.json"):
    st.warning("Downloading models. Please wait...")
    
    model_ids = {
        "subset-roberta": "1E66RxVCzK6VrJKXUW2cVhQ0U6vvc5uVT",
        "subset-1-ml": "1p-9dYOhVfTaiJE1jQf-A_aps4yIHA_NI",
        "fine-tuned-toxic-ml": "1jJ11Z919o2ehPT-UHJgGyA6T_jOUW3Y3",
    }

    for model_name, file_id in model_ids.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output = f"models/{model_name}.zip"
        gdown.download(url, output, quiet=False)

        # Extract the downloaded ZIP file
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(f"models/{model_name}")
st.set_page_config(page_title="Toxic Comment Classification App")

# Create a sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Script 1", "Script 2", "Script 3"])

# Load the selected page
if page == "Home":
    st.title("Toxic Comment Classification App")
    st.write("Welcome to the multi-page Streamlit app.")
elif page == "DistilBERT fullset":
    nlp_1.run()
elif page == "DistilBERT subset":
    nlp_2.run()
elif page == "RoBERTa subset":
    nlp_3.run()
