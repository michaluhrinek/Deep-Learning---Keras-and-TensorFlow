#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Image Classification with Keras and TensorFlow 

# import libraries 

# In[194]:


get_ipython().system('pip install tensorflow')


# In[36]:


import os
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[147]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set the main directory path
main_dir = r'C:\Users\User\Downloads\data'

# Load images and labels
images = []
labels = []
label_to_int = {'passport': 0, 'id': 1}


# In[148]:


# Load passport images
passport_dir = os.path.join(main_dir, "passport")
for filename in os.listdir(passport_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(passport_dir, filename)
        image = cv2.imread(image_path)
        # Resize the image to a fixed size
        image = cv2.resize(image, (28, 28))
        images.append(image)
        labels.append(0)  # Assign integer label 0 for passport


# In[149]:


# Load id images        
receipt_dir = os.path.join(main_dir, "id")
for filename in os.listdir(receipt_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(receipt_dir, filename)
        image = cv2.imread(image_path)
        # Resize the image to a fixed size
        image = cv2.resize(image, (28, 28))
        images.append(image)
        labels.append(1)  # Assign integer label 1 for receipt


# In[150]:


# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)


# In[151]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[152]:


# Preprocess the data (normalize pixel values)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# In[153]:


# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# In[154]:


# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])


# In[155]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[199]:


# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[200]:


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


# In[201]:


y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1) 


# In[202]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[203]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


# In[204]:


# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1) 


# In[205]:


# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[206]:


# Generate a classification report
class_report = classification_report(y_true, y_pred)
print('Classification Report:')
print(class_report)


# In[207]:


# Calculate and print additional metrics
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')


# In[112]:


# Save the entire model to a HDF5 file
model.save(r"C:\Users\User\Downloads\model2.h5")


# In[166]:


import tensorflow as tf


# In[179]:


model = tf.keras.models.load_model(r"C:\Users\User\Downloads\model2.h5")


# In[189]:


# Path to the new image
image_path = r'C:\Users\User\Downloads\999.jpg'


# In[190]:


# Load and preprocess the image
image = cv2.imread(image_path)
image = cv2.resize(image, (28, 28))  # Resize to the same size as the training images
image = image.astype('float32') / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension


# In[191]:


# Make prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=1)[0]


# In[192]:


# Map the predicted class to label
label_to_int = {'passport': 0, 'id': 1}
int_to_label = {v: k for k, v in label_to_int.items()}
predicted_label = int_to_label[predicted_class]


# In[193]:


# Output the result
print(f'The predicted label for the image is: {predicted_label}')


# In[ ]:




