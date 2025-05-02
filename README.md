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

## 🤖 KNN Algorithm

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm that:
1. Stores all training data points.
2. When a prediction is needed, calculates the distance between the new point and all training points.
3. Selects the top **k** nearest neighbors and returns the most common class among them.

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

---

## ✅ Requirements

Install the required libraries via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📚 License

This project is open-source and free to use for educational purposes.
