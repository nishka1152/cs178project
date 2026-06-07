#%%
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = fashion.data, fashion.target.astype(int)

X = X / 255.0

from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

print(X_train.shape)   
print(X_val.shape)     
print(X_test.shape) 

# %%

k_values = [1, 3, 5, 10, 15]

X_train_small = X_train[:5000]
y_train_small = y_train[:5000]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train_small, y_train_small)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"k={k}: val accuracy = {val_acc:.4f}")

# %%
#trying with k values between 5-10 to see which is best since 5, 10 were best before
k_values = [5, 6, 7, 8, 9, 10]
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train_small, y_train_small)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"k={k}: val accuracy = {val_acc:.4f}")

# %%

#result: k = 6
best_model = KNeighborsClassifier(n_neighbors=6, n_jobs=-1)
best_model.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.4f}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(classification_report(y_test, y_pred, target_names=class_names))
# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im)

# Label axes
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('KNN (k=6) Confusion Matrix on Fashion-MNIST')

# Annotate each cell with its count
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.show()

# %%
