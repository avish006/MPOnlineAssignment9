# MPOnline Assignment: Machine Learning Projects Repository

Welcome to the MPOnline Assignment repository! This repository contains a comprehensive collection of 8 distinct machine learning, deep learning, and reinforcement learning projects. Each project is organized into its own directory containing the necessary Python scripts to download data, build models, and evaluate them.

## Table of Contents
1. [Adult Census Income Classification](#1-adult-census-income-classification)
2. [CIFAR-10 Image Classification using CNN](#2-cifar-10-image-classification-using-cnn)
3. [Face Recognition using CNN in Wild Life](#3-face-recognition-using-cnn-in-wild-life-lfw-dataset)
4. [Cancer Detection using MRI Images](#4-cancer-detection-using-mri-images)
5. [Cart-Pole RL Agent Training](#5-cart-pole-rl-agent-training)
6. [Lunar Lander RL Agent Training](#6-lunar-lander-rl-agent-training)
7. [Movie Recommendation System](#7-movie-recommendation-system)
8. [End-to-End Render Deployment Project](#8-end-to-end-render-deployment-project)

---

### 1. Adult Census Income Classification
**Folder:** `1_Adult_Census_Income_Classification`

This project tackles a classic tabular data problem: predicting whether an individual makes more than $50,000 a year based on census data (age, education, marital status, occupation, etc.).
- **Dataset:** UCI Adult Income Dataset.
- **Model:** Random Forest Classifier (`scikit-learn`).
- **Highlights:** Demonstrates data preprocessing (handling missing values, label encoding categorical variables, standardizing numerical features). Outputs an accuracy score, classification report, confusion matrix, and feature importances.

### 2. CIFAR-10 Image Classification using CNN
**Folder:** `2_Cifar10_Image_Classification_using_CNN`

An introduction to computer vision using Deep Learning. The goal is to classify 32x32 color images into one of 10 classes (airplanes, cars, birds, cats, etc.).
- **Dataset:** CIFAR-10 (loaded directly via Keras).
- **Model:** Convolutional Neural Network (CNN) built with `TensorFlow/Keras`.
- **Highlights:** Utilizes multiple Conv2D and MaxPooling2D layers to extract spatial features, followed by dense layers for classification. Training history (accuracy and loss curves) is saved as a plot.

### 3. Face Recognition using CNN in Wild Life (LFW Dataset)
**Folder:** `3_Face_recognition_using_CNN_in_wild_life_LFW`

This project focuses on facial recognition using a subset of the famous Labeled Faces in the Wild (LFW) dataset. 
- **Dataset:** LFW Dataset (fetched via `scikit-learn`).
- **Model:** Custom CNN architecture in `TensorFlow/Keras`.
- **Highlights:** Handles image resizing and normalization, trains a model to distinguish between different prominent public figures, and plots a visual gallery of actual vs. predicted faces.

### 4. Cancer Detection using MRI Images
**Folder:** `4_Cancer_Detection_using_MRI_images`

A vital application of AI in healthcare. This script builds a model to classify ultrasound images to detect signs of breast cancer.
- **Dataset:** BreastMNIST from the `MedMNIST` collection.
- **Model:** CNN tailored for grayscale medical imaging (`TensorFlow/Keras`).
- **Highlights:** Demonstrates end-to-end medical image processing. Loads `.npz` binary data, scales pixel intensities, trains a CNN, and outputs a visual validation grid comparing the model's predictions to true ground-truth labels.

### 5. Cart-Pole RL Agent Training
**Folder:** `5_Cart_Pole_RL_agent_Training`

A foundational Reinforcement Learning problem where an agent must learn to balance a pole on a moving cart without letting it fall over.
- **Environment:** `CartPole-v1` from `Gymnasium`.
- **Model:** Proximal Policy Optimization (PPO) using `stable-baselines3`.
- **Highlights:** Shows how to initialize an environment, train a PPO agent for thousands of timesteps, evaluate the mean reward, and save the model to disk.

### 6. Lunar Lander RL Agent Training
**Folder:** `6_Lunar_Lander_RL_Agent_Training`

A more advanced Reinforcement Learning challenge. The agent must learn to fire thrusters correctly to safely land a spacecraft on a landing pad on the lunar surface.
- **Environment:** `LunarLander-v3` from `Gymnasium`.
- **Model:** Proximal Policy Optimization (PPO) using `stable-baselines3`.
- **Highlights:** Handles a continuous state space and discrete action space. The script trains an agent over 50,000 timesteps to master fuel management and physics-based landing.

### 7. Movie Recommendation System
**Folder:** `7_Movie_Recommendation_System`

This project builds a personalized recommendation engine to predict what rating a user would give to a movie they haven't seen yet.
- **Dataset:** MovieLens-100k.
- **Model:** Singular Value Decomposition (SVD) using `scikit-surprise`.
- **Highlights:** Implements collaborative filtering. Trains an SVD model, evaluates it using Root Mean Squared Error (RMSE), and features a custom function to extract the Top-5 movie recommendations for any given user.

### 8. End-to-End Render Deployment Project
**Folder:** `8_End_to_End_Render_Deployment_Project`

The final step of the ML lifecycle: Deployment. This project provides a template for deploying a Python-based model to the web using Render.com.
- **Tools:** `FastAPI`, `Uvicorn`.
- **Highlights:** Includes a lightweight `app.py` script that exposes a `/predict` REST endpoint. It also includes the necessary `requirements.txt` and `render.yaml` configuration files for seamless Infrastructure-as-Code deployment on Render.

## Setup & Installation

To run these projects locally, ensure you have Python 3.9+ installed. You can install all required dependencies by running:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost tensorflow gymnasium[box2d] stable-baselines3 shimmy fastapi uvicorn medmnist scikit-surprise
```

Run any project by navigating to its directory and executing the Python script, for example:
```bash
cd 1_Adult_Census_Income_Classification
python adult_income.py
```
