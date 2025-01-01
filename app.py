from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 載入模型和相關資源
with open("./linearSVC/LinearSVC.pkl", "rb") as f:
    model = pickle.load(f)

with open("./linearSVC/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# 初始化 LabelEncoder 並設定類別
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Disability', 'Gender', 'Origin', 'Religion', 'Sexual_Orientation'])  # 替換為實際的類別名稱

# 確定特徵總數（即字典中的最大索引 + 1）
num_features = len(dictionary)

with open("./naiveBayes/bernoulli_naive_bayes.pkl", "rb") as f:
    model_data = pickle.load(f)

class_feature_probs = model_data["class_feature_probs"]
class_priors = model_data["class_priors"]
selected_features = model_data["selected_features"]

with open("./naiveBayes/dictionary.pkl", "rb") as f:
    naive_dictionary = pickle.load(f)

naive_label_encoder = LabelEncoder()
naive_label_encoder.classes_ = np.array(['Hate', 'Offensive', 'Normal'])

naive_num_feature = len(naive_dictionary)



# 定義將文本轉換為稀疏向量的函式
def text_to_sparse_vector(text, dictionary):
    """
    將文本轉換為稀疏向量表示。
    """
    sparse_vector = []
    words = text.split()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            sparse_vector.append((index, 1))  # 假設使用簡單詞頻
    return sparse_vector

# 將稀疏向量轉換為密集矩陣
def sparse_to_dense(sparse_vectors, num_features):
    dense_matrix = np.zeros((len(sparse_vectors), num_features))
    for i, sparse_vector in enumerate(sparse_vectors):
        for index, value in sparse_vector:
            dense_matrix[i, index] = value
    return dense_matrix

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

# 建立 Flask 應用程式
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        print("user input:", user_input)
        # 處理用戶輸入
        sparse_vector = text_to_sparse_vector(user_input, dictionary)
        dense_vector = sparse_to_dense([sparse_vector], num_features)
        
        try:
            prediction = model.predict(dense_vector)
            predicted_class = label_encoder.inverse_transform(prediction)[0]
        except ValueError:
            predicted_class = "Unknown"

        print("target prediction: ", predicted_class)
        
        # print("prediction result:", predicted_class)
        naive_sparse_vector = text_to_sparse_vector(user_input, naive_dictionary)
        naive_dense_vector = sparse_to_dense([naive_sparse_vector], naive_num_feature)

        # 应用特征选择
        dense_vector_selected = naive_dense_vector[:, selected_features]

        # 使用贝叶斯模型进行预测
        naive_y_preds = predict_bernoulli_naive_bayes(dense_vector_selected, class_feature_probs, class_priors)
        naive_predicted_class = naive_label_encoder.inverse_transform([naive_y_preds[0]])[0]

        print("sentiment prediction: ", naive_predicted_class)
        # 返回預測結果
        return render_template(
            "index.html",
            post_content = user_input,
            classification_label = predicted_class,
            naive_classification_label = naive_predicted_class
        )
        
    
    return render_template("index.html", post_content=None, classification_label=None)

if __name__ == '__main__':
    app.run(debug = True)
