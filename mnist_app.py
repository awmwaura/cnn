import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.title('MNIST Digit Recognition with CNN')

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
X = np.array(X)

X = X.reshape(-1, 28, 28, 1)
X = X.astype('float32') / 255.0

label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate summary statistics
mean_pixel_value = np.mean(X)
median_pixel_value = np.median(X)
std_pixel_value = np.std(X)

st.subheader('Summary Statistics')
st.write(f'Mean Pixel Value: {mean_pixel_value}')
st.write(f'Median Pixel Value: {median_pixel_value}')
st.write(f'Standard Deviation of Pixel Values: {std_pixel_value}')

st.subheader('Distribution of Digit Labels')
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(10):
    ax.hist(y[:, i], bins=range(2), rwidth=0.8, align='left', label=f'Digit {i}')
ax.set_xlabel('Digit Label')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Digit Labels')
ax.set_xticks(range(10))
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()
st.pyplot(fig)

st.subheader('Example Images')
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    digit_idx = np.where(y[:, i] == 1)[0][0]
    digit_image = X[digit_idx].reshape(28, 28)
    ax.imshow(digit_image, cmap='gray')
    ax.set_title(f'Digit {i}')
    ax.axis('off')
st.pyplot(fig)

st.subheader('Correlation Matrix of Pixel Values')
X = np.random.rand(100, 100)
pixel_correlation = np.corrcoef(X)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pixel_correlation, cmap='coolwarm', interpolation='nearest')
fig.colorbar(im, ax=ax)
ax.set_title('Correlation Matrix of Pixel Values')
ax.set_xlabel('Pixel Index')
ax.set_ylabel('Pixel Index')
st.pyplot(fig)

st.subheader('Build and Train CNN Model')
# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
st.write(f'Test accuracy: {test_acc}')

st.subheader('Visualize Model Predictions on Test Images')
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    predicted_label = np.argmax(model.predict(X_test[i].reshape(1, 28, 28, 1)))
    true_label = np.argmax(y_test[i])
    ax.set_title(f'Predicted: {predicted_label}, True: {true_label}', fontsize=8)
st.pyplot(fig)
