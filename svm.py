import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.transform import resize

# Step 1: Prepare the dataset
cats_dir = r"C:\Users\sowmi\PycharmProjects\SVM\.venv\PetImages\Cat" #Add the path of your dataset here
dogs_dir = r"C:\Users\sowmi\PycharmProjects\SVM\.venv\PetImages\Dog"

categories = {'cats': 0, 'dogs': 1}

X = []
y = []

for category, label in categories.items():
    folder_path = os.path.join(cats_dir if category == 'cats' else dogs_dir)
    for img_file in os.listdir(folder_path):
        img = imread(os.path.join(folder_path, img_file), as_gray=True)
        img = resize(img, (100, 100))  # Resize images to a consistent size
        X.append(img.flatten())  # Convert images to 1D arrays
        y.append(label)

# Step 2: Preprocess the data
X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Extraction
# (You may want to use a more sophisticated feature extraction method, such as pre-trained CNNs)

# Step 4: Train the SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='linear')  # You can experiment with different kernels
svm.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
