import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

folder = "C:\\github\\pca-eigenfaces\\ORL"

img_files = os.listdir(folder)

height = 80 #height in pixels of images
weight = 70 #Weight in pixels of images

data = np.ones((len(img_files), height*weight))
labels = []

for i, l in enumerate(img_files):

	# Full image path
	path = os.path.join(folder, l)

	img = np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

	# Make a vector from an image
	img = img.reshape(-1, img.size)

	# store this vector
	data[i,:]  = img

	# get label from file name ex.: 363_37.jpg, returns 37, which is the label id
	label_id = int(l.split("_")[1].split(".")[0])
	labels.append(label_id)

labels = np.array(labels)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

for i in range(10, 21):

	# Compute a PCA 
	n_components = i
	pca = PCA(n_components=n_components, whiten=True).fit(X_train)

	# apply PCA transformation to training data
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)

	# train a classifier
	clf = KNeighborsClassifier(n_neighbors=3).fit(X_train_pca, y_train)

	y_pred = clf.predict(X_test_pca)

	score = accuracy_score(y_test, y_pred)

	print(str(i)+' Componentes Principais, Acurácia: '+f'{score*100:.2f}'+'%')