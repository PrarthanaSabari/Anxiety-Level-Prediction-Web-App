# Anxiety Level Prediction Web App

This project predicts an individual's anxiety risk level (Low / Moderate / High) based on demographic, lifestyle, and medical history data. It uses Machine Learning models (Random Forest and LightGBM) to analyze user inputs and provides personalized mental health suggestions.

## Project Overview

The project uses:
* **Random Forest & LightGBM:** Two powerful classification models trained to predict anxiety levels.
* **SMOTE Balancing:** Uses Synthetic Minority Oversampling Technique to handle class imbalances in the dataset.
* **Flask Web Interface:** An interactive web app where users can input data and receive instant predictions and actionable advice.

## Folder Structure

```text
ANXIETY_LEVEL_APP/
│
├── app.py
├── enhanced_anxiety_dataset.csv
├── templates/
│   ├── landing.html
│   └── index.html
│
└── README.md
```
## Dependencies

Before running the project, install the required Python packages:

``` bash
pip install flask pandas numpy scikit-learn imbalanced-learn lightgbm joblib
```

## Steps to Run the Project

1. Download the project

    Download the project folder to your local computer.

    Note: Ensure app.py, the dataset, and the templates folder are all in the same directory.

2. Open the Terminal

    Open Command Prompt (or Terminal) inside that folder.

3. Run the Flask application

    Type the following command and hit enter:

   ```bash
   python app.py
   ```

4. Wait for training to complete

    You will see the model accuracy scores printed in the terminal. It will automatically select the best performing model (Random Forest or LightGBM).

```text
Model Training Complete
RandomForest Accuracy: 0.9072
Using Random Forest for predictions.
```

5. Launch the Web App

    Once the training is finished, open the URL shown in the terminal: http://127.0.0.1:5000

6. Get Predictions

    The web app will open. Enter your details (Gender, Smoking status, Medical History, etc.), and the app will predict your anxiety risk level and show personalized coping strategies.

## Model Details
•Dataset: enhanced_anxiety_dataset.csv

•Target Variable: Anxiety Level (Mapped to: 0=Low, 1=Moderate, 2=High)

•Preprocessing:

    Missing values filled with median.

    Categorical variables encoded using Label Encoding.

    Data scaling using Standard Scaler.

•Performance: The models typically achieve ~90-91% accuracy.

## Features
• Smart Model Selection: The app trains both Random Forest and LightGBM on startup and selects the one with higher accuracy.

• Robust Error Handling: Can handle unseen categories (e.g., new job titles) without crashing.

•Personalized Suggestions:

    • High Risk: Suggests therapy and urgent stress relief.

    • Moderate Risk: Suggests breathing exercises and mindfulness.

    • Lifestyle: Suggests sleep hygiene or physical activity improvements based on specific inputs

## Project Contributors

| Name | Role |
| :--- | :--- |
| **Prarthana S** | Lead Developer | 

**Tools Used :** Python, Flask, LightGBM, Scikit-learn, Pandas
