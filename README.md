# smile_recognition

Designing and training the image recognition model
Because the task's data is not large, I can upload all the files and folders needed to Google Colab and train the Machine Learning model.
1.1.	Preprocessing
I used Pillow library to get the image and convert them into RGB color grading.
I also linearized the image so that it would be in the range of 0 and 1.
For the loop, I used the Tqdm library to see the processing bar running.
 
As can be seen in the figure, the number of images is 4000.

Split train and test images
I used sklearn.model_selection.train_test_split to split the training data into training and validation data. Moreover, I used `stratify` method to distribute the data class into train and validation nicely.
 
Because this task has only two classes to predict, I didn't turn them into one-hot-encoding and use binary cross-entropy for the loss calculation function later.

I tested a training data with its label to see if I have assigned everything correctly
 

1.2.	Model selection and training
I used the Convolutional Neural Network (CNN) for the image recognition part of this task because it uses convolutional layers to easily extract the features and edges.
The base model includes  
However, I tried to change a little bit from this by adding:
-	BatchNormalization layer: The whole image batch is going to normalize. Therefore, it will increase the generalization and make the model predict the features better.
-	Dropout: I added a dropout layer of 0.5, which, by experience, make my model perform better.
-	Add more convolution layers than the base model in the above figure.
-	Add more dense layers. Moreover, I added `kernel_initializer` so that the model would have the parameters to train faster.
Train the model:
-	I added callbacks to make the model more efficient. 
-	Firstly, the ModelCheckpoint, so that I can save the weight and the model to re-train purposes or my computer got shut down, I still have the weight file to load and train again.
-	Secondly, ReduceLROnPlateau is the callback that helps me reduce the optimizer's learning rate. If my model stucks in local minima because of a large learning rate, I can reduce it with patience.
-	Thirdly, EarlyStopping allows my model to stop if the validation accuracy is converged.
This is the result of the training:
 
As can be seen, I trained the machine learning model in 50 epochs. However, because I have the EarlyStopping, and in the 8th epoch, it already converged, and I don't have to wait and can stop the learning process there.
 
Test the machine learning model with a webcam video
I have the weight file from the training process above to plug it into the webcam test code. I do not use a pre-trained machine learning network because it would be too overkill for this task. I already achieved an 89% validation accuracy with a simple tweak, and I want to keep it short so that the inference test code should be fast enough.
Moreover, this model ensures that the test video would be real-time and high enough.

The code snippet includes the OpenCV library with the `VideoCapture` method to get the video capture, and get the image video by showing the label "Smile" or "Not Smile" in the top left of the video.
 
The video speed is fast because the network is lightweight. Moreover, the accuracy is high, so that it would ensure to predict the right facial status.
 
 
