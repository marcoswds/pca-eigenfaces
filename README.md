# Face Recognition using PCA

In this project you can find an implementation of Face Recognition using PCA to train the model.

The ORL database of faces is used to train the model, with an addition of 10 of my pictures in grayscale and 70x80 pixels.
For the classifier I used the KNN implementation from sklearn.

### Instructions
**Python 3+** needed

The following libraries are required:
* pip install numpy
* pip install sklearn
* pip install opencv-python

The folder in which you download this project needs to be updated in the **folder** variable:
In my case it was "C:\\github\\pca-eigenfaces\\ORL", set it to your folder accordingly

Finally you can run eigenfaces.py and that's it.

### Result Example
* 10 Componentes Principais, Acurácia: 78.05%
* 11 Componentes Principais, Acurácia: 82.93%
* 12 Componentes Principais, Acurácia: 81.30%
* 13 Componentes Principais, Acurácia: 79.67%
* 14 Componentes Principais, Acurácia: 81.30%
* 15 Componentes Principais, Acurácia: 83.74%
* 16 Componentes Principais, Acurácia: 85.37%
* 17 Componentes Principais, Acurácia: 83.74%
* 18 Componentes Principais, Acurácia: 83.74%
* 19 Componentes Principais, Acurácia: 81.30%
* 20 Componentes Principais, Acurácia: 82.93%
