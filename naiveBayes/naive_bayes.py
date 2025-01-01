import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder

# Load labeled data
# labeled_data = pd.read_csv("data/labeled_data.csv")
labeled_data = pd.read_csv("data/labeled_data.csv")
df = pd.DataFrame(labeled_data)

# Load preprocessed data
with open("./naiveBayes/unit_tfidf_vectors.pkl", "rb") as f:
    unit_tfidf_vectors = pickle.load(f)

with open("./naiveBayes/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# Convert sparse vectors to dense matrix
def sparse_to_dense(sparse_vectors, num_features):
    dense_matrix = np.zeros((len(sparse_vectors), num_features))
    for i, sparse_vector in enumerate(sparse_vectors):
        for index, value in sparse_vector:
            dense_matrix[i, index] = value
    return dense_matrix

# Ensure the total number of features (max index + 1)
num_features = len(dictionary)
X_dense = sparse_to_dense(unit_tfidf_vectors, num_features)



# Apply LabelEncoder on the filtered data
y = df['class'].values\


# Feature selection using Chi-Square
k_best_features = 2000  # Select top k features
chi2_selector = SelectKBest(chi2, k=k_best_features)
X_dense = chi2_selector.fit_transform(X_dense, y)
selected_features = chi2_selector.get_support(indices=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
def train_naive_bayes(X_train, y_train):
    class_term_probs = defaultdict(lambda: defaultdict(float))  # P(term|class)
    class_counts = defaultdict(int)  # Count of samples per class
    total_docs = len(y_train)  # Total number of samples

    term_counts_per_class = defaultdict(lambda: defaultdict(float))  # Count of terms per class
    for i, row in enumerate(X_train):
        label = y_train[i]
        class_counts[label] += 1
        for term_index, term_value in enumerate(row):
            if term_value > 0:
                term_counts_per_class[label][term_index] += term_value

    class_priors = {label: count / total_docs for label, count in class_counts.items()}

    for label, term_counts in term_counts_per_class.items():
        total_terms = sum(term_counts.values())  # Total term weight for this class
        for term_index, count in term_counts.items():
            class_term_probs[label][term_index] = count / total_terms  # Normalize by total terms in class

    return class_term_probs, class_priors

def predict_naive_bayes(X_test, class_term_probs, class_priors):
    predictions = []
    for row in X_test:
        max_prob = -np.inf
        best_class = None

        for label, prior in class_priors.items():
            log_prob = np.log(prior)  # Start with log(P(class))
            for term_index, term_value in enumerate(row):
                if term_value > 0:
                    term_prob = class_term_probs[label].get(term_index, 1e-6)  # Smoothing for unseen terms
                    log_prob += term_value * np.log(term_prob)

            if log_prob > max_prob:
                max_prob = log_prob
                best_class = label

        predictions.append(best_class)
    return predictions

def default_dict_factory():
    return defaultdict(float)

def train_bernoulli_naive_bayes(X_train, y_train):
    """
    Train a Bernoulli Naive Bayes model.
    """
    # class_feature_probs = defaultdict(lambda: defaultdict(float))  # P(feature|class)
    class_feature_probs = defaultdict(default_dict_factory)  # P(feature|class)
    class_counts = defaultdict(int)  # Count of samples per class
    total_docs = len(y_train)  # Total number of samples

    # Binarize the data (convert all non-zero values to 1)
    X_train_binary = (X_train > 0).astype(int)

    feature_counts_per_class = defaultdict(lambda: defaultdict(float))  # Count of features per class
    for i, row in enumerate(X_train_binary):
        label = y_train[i]
        class_counts[label] += 1
        for feature_index, feature_value in enumerate(row):
            if feature_value > 0:
                feature_counts_per_class[label][feature_index] += 1

    # Compute class priors
    class_priors = {label: count / total_docs for label, count in class_counts.items()}

    # Compute conditional probabilities P(feature|class) using Laplace smoothing
    for label, feature_counts in feature_counts_per_class.items():
        total_samples_in_class = class_counts[label]
        for feature_index in range(X_train_binary.shape[1]):
            count = feature_counts[feature_index]
            # Apply Laplace smoothing
            class_feature_probs[label][feature_index] = (count + 1) / (total_samples_in_class + 2)

    return class_feature_probs, class_priors

def predict_bernoulli_naive_bayes(X_test, class_feature_probs, class_priors):
    """
    Predict using the Bernoulli Naive Bayes model.
    """
    X_test_binary = (X_test > 0).astype(int)  # Binarize the test data
    predictions = []

    for row in X_test_binary:
        max_prob = -np.inf
        best_class = None

        for label, prior in class_priors.items():
            log_prob = np.log(prior)  # Start with log(P(class))
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

# Train the Naive Bayes model
# class_term_probs, class_priors = train_naive_bayes(X_train, y_train)
class_feature_probs, class_priors = train_bernoulli_naive_bayes(X_train, y_train)
class_feature_probs = {k: dict(v) for k, v in class_feature_probs.items()}
# 保存 Bernoulli Naive Bayes 模型
model_data = {
    "class_feature_probs": class_feature_probs,
    "class_priors": class_priors,
    "selected_features": selected_features  # 保存特徵選擇結果
}
with open("bernoulli_naive_bayes.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved to bernoulli_naive_bayes.pkl")

# Predict and evaluate the Bernoulli model
y_preds = predict_bernoulli_naive_bayes(X_test, class_feature_probs, class_priors)

report = classification_report(y_test, y_preds)
print(report)

