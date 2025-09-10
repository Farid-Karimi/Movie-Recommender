# CineMatch: A Hybrid Movie Recommendation System

CineMatch is a movie recommendation system designed to provide personalized movie suggestions. It leverages [data](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data) from **The Movie Database (TMDb)** and **MovieLens** to deliver relevant recommendations.

The system is built on a hybrid model that combines the strengths of both **content-based** and **collaborative filtering** methods. It analyzes movie attributes like genre, cast, and keywords to find similar content, and then re-ranks these suggestions based on predicted user ratings, ensuring the final recommendations are tailored to individual tastes. A popularity-based model also serves as a baseline for general recommendations.

## Live Demo

You can try out the deployed application on Hugging Face Spaces:

[CineMatch on Hugging Face](https://huggingface.co/spaces/Pro-metheus/CineMatch)

## How It Works

The recommendation process begins by using a **TF-IDF** vectorizer to analyze textual features of movies, calculating content similarity with **cosine similarity**. To personalize these results, a **Singular Value Decomposition (SVD)** model, trained on user ratings, predicts how a user would rate the similar movies. The final list of recommendations is then sorted based on these predicted scores.

## Running the Project Locally

To set up and run this project on your own machine, follow these steps:

1. Place all the necessary `.csv` data files into a folder named `Data/` at the root of the project.

2. Run the exploratory data analysis and feature engineering notebooks (`EDA/EDA.ipynb` and `Model/Prep.ipynb`) to process the raw data.

3. After preparing the data, you can train the models. While there are notebooks for content-based and collaborative filtering, the hybrid approach is recommended for the best results.

4. The final, deployable application is a Gradio app located in the `cinematch/` folder.
