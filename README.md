# Hate-Speech-Classification

Welcome to the **Hate Speech Classification** project!

This project is all about leveraging the power of **machine learning** to detect and classify hate speech in text. Our goal is to make the internet a safer and more inclusive space by identifying offensive and harmful language before the user share it or post it.

## 🌟 Features

- **Dual-Model Classification:** 
  - **LinearSVC:** A powerful support vector classifier for accurate text classification.
  - **Bernoulli Naive Bayes:** A lightweight and efficient probabilistic model for quick and reliable predictions.
  
- **Real-Time Predictions:** Input your text and get instant feedback on its classification.

- **Multiclass Categorization:** 
  - Hate speech is categorized into classes such as `Disability`, `Gender`, `Origin`, `Religion`, and `Sexual_Orientation`.
  - Sentimenet speech is categorized into classes such as `Hate`,  `Offensive`, `Normal`

- **User-Friendly Interface:** Built with Flask for a seamless, interactive user experience.

## 🔧 How It Works

1. **Preprocessing:** Text data is cleaned, tokenized, and vectorized using a custom TF-IDF representation.
2. **Feature Selection:** The most significant features are selected using statistical methods like Chi-Square.
3. **Model Training:** Both models are trained on labeled datasets for robust performance.
4. **Prediction:** User-submitted text is evaluated in real-time and classified into categories.

## 🚀 Getting Started

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Hate-Speech-Classification.git
   cd Hate-Speech-Classification

2. Start the Flask application:

```bash
python app.py
```

3. Open your browser and navigate to:
```bash
http://127.0.0.1:5000
```

## 📂 Project Structure

- **`app.py`**: The main Flask application.
- **`linearSVC/`**: Contains the pre-trained LinearSVC model and its dictionary.
- **`naiveBayes/`**: Contains the Naive Bayes model and supporting resources.
- **`templates/`**: HTML templates for the web interface.
- **`static/`**: CSS and other static assets.

---

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework for building the interactive interface.
- **Scikit-learn**: For machine learning model training and evaluation.
- **HTML/CSS**: For frontend development.

---

## 🎯 Why This Matters

Hate speech is a growing problem in online spaces, and tackling it is essential for fostering healthy digital communities. By using AI and machine learning, we aim to detect and prevent hate speech effectively, contributing to a safer internet for everyone.
