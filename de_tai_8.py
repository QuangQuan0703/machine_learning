from gc import callbacks
import os
import warnings
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



data_path = '/home/nguyenquan/Desktop/Baitaplon_MachinLearning'

train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(data_path, 'train-labels-idx1-ubyte.gz')

test_images_path = os.path.join(data_path, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_path, 't10k-labels-idx1-ubyte.gz')



#read gzip
def get_mnist_data (images_path, labels_path, num_images, shuffle = False, _is=True, image_size= 28):

    #read data
    import gzip       # to decompress gz (zip) file

    # open file training to read training data
    f_images = gzip.open(images_path, 'r')

    # skip 16 first bytes because these are not data, only deader infor
    f_images.read(16)


    # general: read num_images data samples if this parameter is set;
    # if not, read all (60000 training or 10000 test)
    real_num = num_images if not shuffle else (60000 if _is else 10000)

    # read all data to buf_images (28x28xreal_num)
    buf_images = f_images.read(image_size * image_size * real_num)

    # images
    images = np.frombuffer(buf_images, dtype=np.uint8).astype(np.float32)
    images = images.reshape(real_num, image_size, image_size,)

    # Read labels
    f_labels = gzip.open(labels_path,'r')
    f_labels.read(8)

    labels = np.zeros((real_num)).astype(np.int64)

    # rearrange to correspond the images and labels
    for i in range(0, real_num):
        buf_labels = f_labels.read(1)
        labels[i] = np.frombuffer(buf_labels, dtype=np.uint8).astype(np.int64)


    # shuffle to get random images data
    if shuffle is True:
        rand_id = np.random.randint(real_num, size=num_images)

        images = images[rand_id, :]
        labels = labels[rand_id,]

    # change images data to type of vector 28x28 dimentional
    images = images.reshape(num_images, image_size * image_size)
    return images, labels



train_images, train_labels = get_mnist_data(train_images_path, train_labels_path, 60000)
test_images, test_labels = get_mnist_data(test_images_path, test_labels_path, 10000)
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.decomposition import PCA

pca3D = PCA(n_components=3)
pca3D.fit(train_images)
pac_transform3D = pca3D.transform(train_images)

fig3 = plt.figure()
fig3.set_size_inches(10,10)
ax = plt.axes(projection='3d')
colors = ['red','blue','green','brown', 'orange', 'black', 'pink','purple','gray','yellow']

for label in range(10):
    ax.scatter(pac_transform3D[train_labels==label,0],
                pac_transform3D[train_labels==label,1],
                pac_transform3D[train_labels==label, 2],
                s= 5, c = colors[label])
    
ax.set_title('decomposition')
plt.show()


pca2D = PCA(n_components=2)
pca2D.fit(train_images)
pac_transform2D = pca2D.transform(train_images)

fig2 = plt.figure()
fig2.set_size_inches(10,10)
for label in range(10):
    plt.scatter(pac_transform2D[train_labels==label,0],
                pac_transform2D[train_labels==label,1],
                s=5, c = colors[label])
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
clf =MultinomialNB()
clf.fit(train_images, train_labels)


predictTest = clf.predict(test_images)
print(predictTest)

sum = 0
for i in predictTest:
    if (predictTest[i]-test_labels[i]==0):
        sum= sum +1

percentCorrect = sum/len(predictTest)
print(percentCorrect)




#-----------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

accuracyMultinomial = accuracy_score(test_labels, predictTest)
confusionMatrixMultinomial= confusion_matrix(test_labels, predictTest)
precisionMultinomial = precision_score(test_labels, predictTest, average='macro')
recallMultinomial = recall_score(test_labels, predictTest, average='macro')

print("Accuracy:", accuracyMultinomial)
print("Confusion matrix:\n", confusionMatrixMultinomial)
print("Precision:", precisionMultinomial)
print("Recall:", recallMultinomial)


##----------------------------------------------------------------------------------------------------------------------------------------------------------
##mean-K clustering
#from scipy.spatial import distance as dist
#def kmeans_creat_centers(X, k):
#    return X[np.random.choice(X.shape[0], k, replace=False),:]
#
#def kmeans_assign_labels(X, centers):
#    D = dist.cdist(X, centers)
#    return np.argmin(D, axis=1)
#
#def kmeans_update_centers(X, labels, k):
#    centers = np.zeros((k, X.shape[1]))
#    for i in range(k):
#        Xk = X[labels == i, :]
#        centers[i,:] = np.mean(Xk, axis=0)
#    return centers
#
#def has_converged(centers, new_centers):
#    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))
#
#def kmeans(X, k):
#    centers = [kmeans_creat_centers(X,k)]
#    labels = []
#    it = 0
#    while True:
#        labels.append(kmeans_assign_labels(X, centers[-1]))
#        new_centers = kmeans_update_centers(X, labels[-1], k)
#        if has_converged(centers[-1], new_centers):
#            break
#        centers.append(new_centers)
#        it +=1
#    return(centers, labels, it)
#
#
#def kmeans_display(train_image, label_iamge, k):
#    fig2 = plt.figure()
#    fig2.set_size_inches(10,10)
#    for label in range(k):
#        plt.scatter(train_image[label_iamge == label,0],
#                train_image[label_iamge == label,1],
#                s=5, c = colors[label])
#    plt.show()
#
#
#(centers, label_image, it)= kmeans(pac_transform3D, 3)
#print(it)
#a = np.array(centers[-1])
#print(a[0])
#print(label_image[-1])
#kmeans_display(pac_transform2D, label_image[-1], 3)
#

#---------------------------------------------------------------------------------------------------------------------------------------
#CNN

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)

train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1) 
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
print(train_labels[0])

model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3),padding="same", activation= 'relu', input_shape =(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(rate= 0.15),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu' , padding="valid"),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(rate=0.10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu', use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(units=10, activation='softmax', use_bias=True, bias_initializer='zeros') 
])
model.compile(optimizer= RMSprop(learning_rate = 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])
model.summary()

 

history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=1000,
    callbacks=[callbacks],
    steps_per_epoch=60,
    epochs=15,
    verbose=2
)



plt.imshow(test_images[0].reshape(28,28))
plt.show

