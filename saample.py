import cv2
import os
import matplotlib.pyplot as plt
import itertools
from random import shuffle
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from PIL import Image
import os, sys

path = "data/train/np/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

resize()

base_dir = os.getcwd()
train_tumor ='data/train/p'
train_non ='data/train/np'
images_path = os.path.join(base_dir, train_tumor)
tumor_paths = sorted(os.listdir(images_path))
images_path = os.path.join(base_dir, train_non)
non_paths = sorted(os.listdir(images_path))

test_tumor ='data/test/p'
test_non ='data/test/np'
test_images_path = os.path.join(base_dir, test_tumor)
test_tumor_paths = sorted(os.listdir(test_images_path))
test_images_path = os.path.join(base_dir, test_non)
test_non_paths = sorted(os.listdir(test_images_path))

x_train = []
y_train = []
x_test = []
y_test = []

train_tumor_count = 0
train_non_count = 0
test_tumor_count = 0
test_non_count = 0

for index, i in enumerate(tumor_paths):
    img = cv2.imread("data/train/p/" + i)
    # img = cv2.resize(img, None,fx=0.75, fy=0.75)
    train_tumor_count += 1
    for z in range(4):
        rotated = np.rot90(img)
        flipped = np.fliplr(rotated)
        x_train.append(rotated)
        x_train.append(flipped)
        y_train.append("1")
        y_train.append("1")

for index, i in enumerate(non_paths):
    img = cv2.imread("data/train/np/" + i)
    # img = cv2.resize(img, None,fx=0.5, fy=0.5)
    train_non_count += 1
    for z in range(4):
        rotated = np.rot90(img)
        flipped = np.fliplr(rotated)
        x_train.append(rotated)
        x_train.append(flipped)
        y_train.append("0")
        y_train.append("0")

for index, i in enumerate(test_tumor_paths):
    img = cv2.imread("data/test/p/" + i)
    # img = cv2.resize(img, None,fx=0.75, fy=0.75)
    test_tumor_count += 1
    x_test.append(img)
    y_test.append("1")

for index, i in enumerate(test_non_paths):
    img = cv2.imread("data/test/np/" + i)
    # img = cv2.resize(img, None,fx=0.75, fy=0.75)
    test_non_count += 1
    x_test.append(img)
    y_test.append("0")

num_classes = 2
classes = ["Pothole", "Non-Pothole"]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# opt = keras.optimizers.RMSprop(learning_rate=0.001)  # Learning rate used 0.01
opt = keras.optimizers.Adam(lr=0.001)  # ... can use check format
# compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)  # Training accuracy
#### fit the model    successful run for traning accuracy
history = model.fit(np.asarray(x_train), y_train, epochs=5)  # number of epoch

###Testing###
y_pred = model.predict(np.asarray(x_test))

#### fit the model trying for test accuracy
#history=model.fit(y_pred, y_test, epochs=2)

### confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

print("Training images (p): " + str(train_tumor_count))
print("Training images (np): " + str(train_non_count))
print("Testing images (p): " + str(test_tumor_count))
print("Testing images (np): " + str(test_non_count))

print("confusion Matrix")
print(str(tn) + "\t" + str(fn))
print(str(fp) + "\t" + str(tp))

print("True positive: correctly identified Pothole " + str(tp))
print("True negative:correctly identified nonPothole " + str(tn))
print("False positive:Incorrectly identified Pothole" + str(fp))
print("False negative:Incorrectly identified nonPothole " + str(fn))

print("Accuracy: " + str((tp + tn) / (tn + tp + fp + fn)))
print("Precision: " + str(tp / (tp + fp)))
print("Recall: " + str(tp / (tp + fn)))
print("F1 score: " + str((2 * tp) / (2 * tp + fp + fn)))
print("Specificity: " + str(tn / (tn + fp)))
print("Sensitivity: " + str(tp / (tp + fn)))
print("Error: " + str((fp + fn) / (tn + tp + fp + fn)))

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## print("classification_report"(y_true, y_pred, target_names=num_classes))
def plot_confusion_matrix(cm, labels,
                          normalize=True,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.rcParams.update({'font.size': 20})


plt.figure(figsize=(5, 5))
plot_confusion_matrix(cm, labels=classes)
plt.show()
