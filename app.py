from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

# Load the LinearSVC model and dictionary
with open("./linearSVC/LinearSVC.pkl", "rb") as f:
    model = pickle.load(f)

with open("./linearSVC/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# Initialize LabelEncoder with class labels for LinearSVC
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Disability', 'Gender', 'Origin', 'Religion', 'Sexual_Orientation'])

# Load the Naive Bayes model and dictionary
with open("./naiveBayes/bernoulli_naive_bayes.pkl", "rb") as f:
    model_data = pickle.load(f)

class_feature_probs = model_data["class_feature_probs"]
class_priors = model_data["class_priors"]
selected_features = model_data["selected_features"]

with open("./naiveBayes/dictionary.pkl", "rb") as f:
    naive_dictionary = pickle.load(f)

# Initialize LabelEncoder with class labels for Naive Bayes
naive_label_encoder = LabelEncoder()
naive_label_encoder.classes_ = np.array(['Hate', 'Offensive', 'Normal'])

# Define helper functions
def text_to_sparse_vector(text, dictionary):
    """
    Converts input text into a sparse vector based on the provided dictionary.
    """
    sparse_vector = []
    words = text.split()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            sparse_vector.append((index, 1))
    return sparse_vector

def sparse_to_dense(sparse_vectors, num_features):
    """
    Converts sparse vectors to a dense matrix representation.
    """
    dense_matrix = np.zeros((len(sparse_vectors), num_features))
    for i, sparse_vector in enumerate(sparse_vectors):
        for index, value in sparse_vector:
            dense_matrix[i, index] = value
    return dense_matrix

def predict_with_threshold(model, X, threshold, label_encoder):
    """
    Predicts using the LinearSVC model with a threshold for classification confidence.
    Returns 'Accept' if all probabilities are below the threshold.
    """
    # Get decision scores from the model
    decision_scores = model.decision_function(X)
    
    # Convert scores to pseudo-probabilities using softmax
    probabilities = softmax(decision_scores, axis=1)
    
    predictions = []
    for i, probs in enumerate(probabilities):
        if max(probs) < threshold:
            # If all probabilities are below the threshold, classify as 'Accept'
            predictions.append('No Target')
        else:
            # Otherwise, classify based on the highest probability
            predicted_index = np.argmax(probs)
            predictions.append(label_encoder.inverse_transform([predicted_index])[0])
    
    return predictions

def predict_bernoulli_naive_bayes(X_test, class_feature_probs, class_priors):
    """
    Predict using the Bernoulli Naive Bayes model.
    """
    X_test_binary = (X_test > 0).astype(int)
    predictions = []

    for row in X_test_binary:
        max_prob = -np.inf
        best_class = None

        for label, prior in class_priors.items():
            log_prob = np.log(prior)
            for feature_index, feature_value in enumerate(row):
                if feature_value == 1:
                    prob = class_feature_probs[label].get(feature_index, 1e-6)
                else:
                    prob = 1 - class_feature_probs[label].get(feature_index, 1e-6)

                log_prob += np.log(prob)

            if log_prob > max_prob:
                max_prob = log_prob
                best_class = label

        predictions.append(best_class)
    return predictions

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user input
        user_input = request.form["user_input"]
        
        print("User input:", user_input)

        # Preprocess input for LinearSVC
        sparse_vector = text_to_sparse_vector(user_input, dictionary)
        dense_vector = sparse_to_dense([sparse_vector], len(dictionary))
        
        # Set threshold for classification
        threshold = 0.4

        # Use prediction with threshold for LinearSVC
        predictions_with_threshold = predict_with_threshold(model, dense_vector, threshold, label_encoder)
        predicted_class = predictions_with_threshold[0]

        print("Target prediction (LinearSVC):", predicted_class)

        # Preprocess input for Naive Bayes
        naive_sparse_vector = text_to_sparse_vector(user_input, naive_dictionary)
        naive_dense_vector = sparse_to_dense([naive_sparse_vector], len(naive_dictionary))

        # Apply feature selection
        dense_vector_selected = naive_dense_vector[:, selected_features]

        # Predict using Naive Bayes with threshold
        naive_y_preds = predict_bernoulli_naive_bayes(dense_vector_selected, class_feature_probs, class_priors)
        naive_predicted_class = naive_label_encoder.inverse_transform([naive_y_preds[0]])[0]

        print("Sentiment prediction (Naive Bayes):", naive_predicted_class)

        # Render results
        return render_template(
            "index.html",
            post_content=user_input,
            classification_label=predicted_class,
            naive_classification_label=naive_predicted_class
        )
    
    # Default render for GET requests
    return render_template("index.html", post_content=None, classification_label=None, naive_classification_label=None)

if __name__ == '__main__':
    app.run(debug=True)
