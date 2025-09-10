# How to Run the Project

Follow these steps to set it up:
## 1. Prepare the Data

Place all the `.csv` files in a root-level folder named `Data/`. The notebooks expect this exact folder name since i couldn't submit it due to sheer size of it.
## 2. Exploratory Data Analysis

Run the notebook `EDA/EDA.ipynb` to process raw data, generate cleaned datasets, and produce plots. Save the outputs for the next stage.
## 3. Feature Engineering

Run `Model/Prep.ipynb` to perform feature engineering and prepare the data for training.
## 4. Train the Models

Experiment with the recommender approaches:

* Content-Based Filtering
* Collaborative Filtering (no `surprise` library, kept for reference)
* Hybrid Approach (recommended for deployment)

The order does not matter, but you must run the Hybrid notebook to deploy.
## 5. Run the App

Two versions exist:

* **Slapp (Streamlit app):** First prototype. Couldn’t be deployed on Hugging Face due to file size and Streamlit restrictions caused by using docker.
* **Cinematch (Gradio app):** Final version. Located in the `cinematch/` folder. Uses the saved models and CSV files for deployment on Hugging Face.