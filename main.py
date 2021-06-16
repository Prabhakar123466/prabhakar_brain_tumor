
#%%
import tkinter
from tkinter.filedialog import askopenfilename
import filetype
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit([[0], [1]])

# %%
data = []
paths = []
result = []
#%%
os.listdir(r'C:\Users\prabh\Desktop\Prabhakar\brain_tumor_dataset')
# %%
for r, d, f in os.walk(r'C:\Users\prabh\Desktop\Prabhakar\brain_tumor_dataset\yes'):
    for file in f:
        if '.JPG' in file:
            paths.append(os.path.join(r, file))
# %%
for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
# %%
paths = []
for r, d, f in os.walk(r'C:\Users\Prabhakar\brain_tumor_dataset\no'):
    for file in f:
        if '.JPG' in file:
            paths.append(os.path.join(r, file))
#%%
paths
# %%
for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())

# %%
data = np.array(data)
np.shape(data)

# %%
result = np.array(result)
result = result.reshape(74, 2)
result.shape
#result = result.reshape(139,2)
x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)
model = Sequential()
# %%
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))
# %%
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# %%
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
# %%
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# %%
model.add(Flatten())
# %%
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# %%
model.compile(loss = "categorical_crossentropy", optimizer='Adamax')
print(model.summary())
# %%
y_train.shape
# %%
history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1, validation_data = (x_test, y_test))
# %%
plt.plot(history.history['loss'])
#%%
plt.plot(history.history['val_loss'])
#%%
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#%%
plt.legend(['Test', 'Validation'], loc='upper right')
#%%
plt.show()
#%%
def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'
#%%
from matplotlib.pyplot import imshow
#%%

# %%
img = Image.open(r'C:\Users\prabh\Desktop\Prabhakar\brain_tumor_dataset\no\N17.jpg')
x = np.array(img.resize((128,128)))
# %%
x = x.reshape(1,128,128,3)
# %%
res = model.predict_on_batch(x)
# %%
classification = np.where(res == np.amax(res))[1][0]
# %%
plt.imshow(img)
plt.show()
# %%
print(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))
# %%
from matplotlib.pyplot import imshow
# %%

img = Image.open(r'C:\Users\prabh\Desktop\Prabhakar\brain_tumor_dataset\yes\Y3.jpg')
# %%
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
# %%
res = model.predict_on_batch(x)
# %%
classification = np.where(res == np.amax(res))[1][0]
# %%
plt.imshow(img)
plt.show()
# %%
print(str(res[0][classification]*100) + '% Confidence This Is A ' + names(classification))

# %%
