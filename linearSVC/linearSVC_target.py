import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load labeled data
labeled_data = pd.read_csv("data/en_dataset.csv")
df = pd.DataFrame(labeled_data)

# 去掉 "other" 的數據
df = df[(df['target'] != "other")]

# 從文件讀取
with open("./linearSVC/unit_tfidf_vectors.pkl", "rb") as f:
    unit_tfidf_vectors = pickle.load(f)

with open("./linearSVC/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)
    print(len(dictionary))

# 確定特徵總數（即字典中的最大索引 + 1）
num_features = len(dictionary)

# 將稀疏向量轉換為密集矩陣
def sparse_to_dense(sparse_vectors, num_features):
    dense_matrix = np.zeros((len(sparse_vectors), num_features))
    for i, sparse_vector in enumerate(sparse_vectors):
        for index, value in sparse_vector:
            dense_matrix[i, index] = value
    return dense_matrix

# 將 unit_tfidf_vectors 轉換為密集矩陣
X_dense = sparse_to_dense(unit_tfidf_vectors, num_features)

# 過濾掉 "other" 的數據後，相應地過濾 X_dense
filtered_indices = df.index
X_dense_filtered = X_dense[filtered_indices]

# 初始化 LabelEncoder
label_encoder = LabelEncoder()

# 將文字標籤轉換為數值標籤
y = label_encoder.fit_transform(df['target'])

# 查看映射關係
print("Class Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_dense_filtered, y, test_size=0.2, random_state=42)

# 初始化並訓練 LinearSVC
model = LinearSVC(class_weight='balanced', C=0.1, multi_class='ovr') # One-vs-Rest
model.fit(X_train, y_train)

with open("LinearSVC.pkl", "wb") as f:
    pickle.dump(model, f)

# 預測並評估模型
y_preds = model.predict(X_test)
report = classification_report(y_test, y_preds, target_names=label_encoder.classes_)
print(report)

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_preds)

# 繪製混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')

# 顯示圖形
plt.title("Confusion Matrix")
plt.show()
