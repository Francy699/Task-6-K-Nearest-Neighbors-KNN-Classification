# 🌸 KNN Classification on Iris Dataset

This project demonstrates how to use the **K-Nearest Neighbors (KNN)** algorithm from `scikit-learn` to classify iris flower species using the classic Iris dataset.

---

## 📦 Dataset Description

- **Source**: `sklearn.datasets.load_iris()`
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**:
  - 0 = Setosa
  - 1 = Versicolor
  - 2 = Virginica

---
🔍 What is K-Nearest Neighbors (KNN)?
K-Nearest Neighbors (KNN) is a simple, intuitive supervised machine learning algorithm used for both classification and regression. It’s based on the idea that similar data points exist close to each other in space.

📌 Key Concepts:
Lazy Learning: KNN does not build a model during training. It simply stores the data.

Instance-Based: It makes predictions based on the actual training data points.

Distance-Based: Commonly uses Euclidean distance to measure how close points are.

🔢 How KNN Works (for Classification):
Choose a value for k (the number of neighbors).

Calculate the distance between the input point and all points in the training set.

Select the k nearest data points.

Count the most frequent class among the k neighbors.

Assign that class to the input point.

🧠 Summary of Your KNN Model (from the Notebook)
Dataset: Iris flower dataset (3 classes, 4 features).

Preprocessing:

Standardized features using StandardScaler.

Split data into training and test sets (80/20).

Training:

Applied KNN for values of k from 1 to 10.

Used KNeighborsClassifier from sklearn.

Evaluation:

For each k, printed out the classification accuracy.

Accuracy ranged across values of k, helping to find the optimal one.

✅ Observations:
Higher accuracy is typically achieved with a moderate value of k (not too small to overfit, not too large to underfit).

Feature scaling (standardization) is critical for KNN to work properly since it relies on distance calculations.

## 🤖 KNN Algorithm

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm that:
1. Stores all training data points.
2. When a prediction is needed, calculates the distance between the new point and all training points.
3. Selects the top **k** nearest neighbors and returns the most common class among them.
   
![image](https://github.com/user-attachments/assets/c6028e3b-cd20-4614-a4a1-011836dc2b50)

Key points:
- Lazy learning (no training phase).
- Uses Euclidean distance (by default).
- Affected by feature scaling.

---

## 🧪 Steps Performed

1. Load and inspect the dataset.
2. Normalize features using `StandardScaler`.
3. Split data into training and test sets.
4. Train KNN classifiers with `k` from 1 to 10.
5. Measure and print accuracy for each value of `k`.

---

## 🧾 Code

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris=load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.Series(iris.target, name="species")

print(X.head())

![Screenshot 2025-05-02 145628](https://github.com/user-attachments/assets/6b617a54-df51-40ed-b818-9c13775d6a93)

print(y.value_counts())

#Normalize features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#implement KNN classifier with different values of K
for k in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(f"K={k} --> Accuracy : {acc: .2f}")

![Screenshot 2025-05-02 145638](https://github.com/user-attachments/assets/23e1cd92-078e-48a7-99e6-512454c1c099)


best_k=3
knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

#Evaluate model using accuracy and confusiin matrix
cnm=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
print("confusion matrix:",cnm)
sns.heatmap(cnm,annot=True,cmap="Blues",fmt="d")
plt.title(f"Confusion Matrix for K={best_k}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

![Screenshot 2025-05-02 145651](https://github.com/user-attachments/assets/2d23a9f4-6abd-4f33-b23d-d59ff272fdc3)


from matplotlib.colors import ListedColormap
X_2d=X_scaled[:, :2] #only first 2 features
X_train2,X_test2,y_train2,y_test2=train_test_split(X_2d,y, test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train2,y_train2)

#Create Meshgrid for visualization
#Predict on Meshgrid 
h=.02
x_min, x_max=X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min ,y_max=X_2d[:, 0].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))

Z=knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)

#plot Decision Boundaries
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']),aplha=0.5)
plt.scatter(X_2d[0:, 0], X_2d[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000','#00FF00','#0000FF']))
plt.title("KNN Decision Boundaries (using 2 features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
![Screenshot 2025-05-02 145734](https://github.com/user-attachments/assets/76e028ac-01e6-40b2-b27e-e8a14b12e155)

---

## ✅ Requirements

Install the required libraries via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📚 License

This project is open-source and free to use for educational purposes.
