
# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project


---


[//]: # (Image References)

[image1]: ./images/NVIDIA.jpeg 
[image2]: ./images/aug.png 


## Introduction



The objective of this project is to teach the computer to drive car on on the basis of data collected in simulator provided by Udacity .
I have used data provided by Udacity and also the data collected by me, driving around different tracks.

Udacity_data  -  [udacity data](https://.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip).

Collected_data  - [collected data](https://.behaviourcloning1.s3.us-east-2.amazonaws.com/data.zip).

Here I have used the concepts of Deep Learning and Convolutional Neural Networks to teach the computer to drive car autonomously.

We feed the data collected from Simulator to our model, this data is fed in the form of images captured by 3 dashboard cams center, left and right. The output data contains a file out.csv which has the mappings of center, left and right images and the corresponding steering angle, throttle, brake and speed. 

Using Keras Deep learning framework we can create a model.h5 file which we can test later on simulator with the command 

```python drive.py model.h5```

This drive.py connects your model to simulator. The challenge in this project is to collect all sorts of training data so as to train the model to respond correctly in any type of situation.

---


#### 1. Project files:

My project includes the following files:

* model_behaviour.py containing the script to create and train the model
* model_behaviour.ipynb python notebook containing the code to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results


---

## Model Architecture and Training Strategy


###  Model Overview


* I tried implementing my  own network but I didn't provide the result I was looing for so I changed to the nvidia model as suggested by udacity

* I decided to test the model provided by NVIDIA as suggested by Udacity. The model architecture is described by NVIDIA [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

![alt text][image1]



```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________

```
### Loading Data 

* I used the the dataset provided by Udacity as well as collected my owndata using simulator.
* I am using OpenCV to load the images, by default the images are read by OpenCV in BGR format but we need to convert to RGB as in drive.py it is processed in RGB format.
* Since we have a steering angle associated with three images we introduce a correction factor for left and right images since the steering angle is captured by the center angle.
* I decided to introduce a correction factor of 0.2
* For the left images I increase the steering angle by 0.2 and for the right images I decrease the steering angle by 0.2

```python

def generator(sets, batch = 64):
  
  len_sets = len(sets)
  
  while 1:
    shuffle(sets)
    
    for offset in range(0, len_sets , batch):
      
      batch_sets = sets[offset:offset+batch]
      
      images = []
      angles = []
      
      for line in batch_sets: 
        
        for i in range(0,3): #to get centre, left and right images
          
          name = './data/IMG/' + line[i].split('/')[-1]
          
          centre_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
          
          centre_angle = float(line[3])
          
          images.append(centre_image)
          
          if(i == 0):
            angles.append(centre_angle)
          elif(i == 1):
            angles.append(centre_angle + 0.2)
          elif (i== 2):
            angles.append(centre_angle - 0.2)
            
          images.append(cv2.flip(centre_image,1))
          
          if(i==0):
            angles.append(centre_angle * -1)
          elif(i == 1):
            angles.append((centre_angle + 0.2) * -1)
          elif(i == 2):
            angles.append((centre_angle - 0.2) * -1)
            
          
          aug_img = cv2.cvtColor(centre_image,cv2.COLOR_RGB2HSV)  #randomising brighness value
          brightness = .25 + np.random.uniform()
          aug_img[::2] =  aug_img[::2] * brightness
          aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
          
          images.append(aug_img)
          
          if(i == 0):
            angles.append(centre_angle)
          elif(i == 1):
            angles.append(centre_angle + 0.2)
          elif (i== 2):
            angles.append(centre_angle - 0.2)
          
          aug_img_eq = np.copy(centre_image)
          for channel in range(aug_img_eq.shape[2]):
            aug_img_eq[:,:, channel] = exposure.equalize_hist(aug_img_eq[:,:, channel])* 255
            
          images.append(aug_img_eq)
          
          if(i==0):
            angles.append(centre_angle * -1)
          elif(i == 1):
            angles.append((centre_angle + 0.2) * -1)
          elif(i == 2):
            angles.append((centre_angle - 0.2) * -1)
          
          
          aug_img_noise = img_as_ubyte(random_noise(centre_image, mode = 'gaussian'))
          
          images.append(aug_img_noise)
          
          if(i==0):
            angles.append(centre_angle * -1)
          elif(i == 1):
            angles.append((centre_angle + 0.2) * -1)
          elif(i == 2):
            angles.append((centre_angle - 0.2) * -1)    
          
           
      X_train = np.array(images)
      y_train = np.array(angles)
      
      yield sklearn.utils.shuffle(X_train,y_train)
      
```


### Preprocessing

* I decided to shuffle the images so that the order in which images comes doesn't matters to the CNN
* Augmenting the data- i decided to flip the image horizontally and adjust steering angle accordingly, I used cv2 to flip the images.
* In augmenting after flipping multiply the steering angle by a factor of -1 to get the steering angle for the flipped image.
* Then, I augmented by adding random brightness by randomising the value of v in hsv space.
* Followed by equalizing hist using skimage and also introduced random noise to the images using random noise from skimage
* So according to this approach we were able to generate 15  images corresponding to one entry in .csv file
* I have used Generator to create these steps as this is more efficient and faster compared to applying these augmwntation separately and saving them to increase the dataset.

![alt text][image2]




### Creation of the Training Set & Validation Set

* I analyzed the Udacity Dataset and found out that it contains 9 laps of track 1 with recovery data and I have created a dataset with 2 laps around each track with one recovery lap in each track.
* I decided to split the dataset into training and validation set using sklearn preprocessing library.
* I decided to keep 20% of the data in Validation Set and remaining in Training Set
* I am using generator to generate the data so as to avoid loading all the images in the memory and instead generate it at the run time in batches of 64. Even Augmented images are generated inside the generators.

```python

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_set, validation_set = train_test_split(sets , test_size = 0.3)

```

### Final Model Architecture

* I made a little changes to the original NVIDIA architecture, my final architecture .
* As it is clear from the model summary my first step is to apply normalization to the all the images.
* Second step is to crop the image 70 pixels from top and 25 pixels from bottom. The image was cropped from top because I did not wanted to distract the model with trees and sky and 25 pixels from the bottom so as to remove the dashboard that is coming in the images.
* Next Step is to define the first convolutional layer with filter depth as 24 and filter size as (5,5) with (2,2) stride followed by ELU activation function
* Moving on to the second convolutional layer with filter depth as 36 and filter size as (5,5) with (2,2) stride followed by ELU activation function 
* The third convolutional layer with filter depth as 48 and filter size as (5,5) with (2,2) stride followed by ELU activation function
* Next we define two convolutional layer with filter depth as 64 and filter size as (3,3) and (1,1) stride followed by ELU activation funciton
* Next step is to flatten the output from 2D to side by side
* Here we apply first fully connected layer with 100 outputs
* Here is the first time when we introduce Dropout with Dropout rate as 0.25 to combact overfitting
* Next we introduce second fully connected layer with 50 outputs
* Then comes a third connected layer with 10 outputs
* And finally the layer with one output.

```python

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

#normalizing the images and mean centering
model.add(Lambda(lambda x: (x/ 255.0) - 0.5 ,input_shape = (160,320,3)))

#cropping the images while passing into the model(selecting required part of image)
model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample = (2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(36,5,5,subsample = (2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(48,5,5,subsample = (2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('elu'))

model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('elu'))

model.add(Dense(10))
model.add(Activation('elu'))

model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator,samples_per_epoch = len(train_set),validation_data = validation_generator, nb_val_samples = len(validation_set),nb_epoch = 5, verbose = 1)

model.summary()
model.save('model.h5')

print('Model Saved!')

```



Here we require one output just because this is a regression problem and we need to predict the steering angle.


### Attempts to reduce overfitting in the model
* After the full connected layer I have used a dropout so that the model generalizes on a track that it has not seen. I decided to keep the Dropoout rate as 0.25 to combact overfitting.

### Model parameter tuning

* No of epochs= 5
* Optimizer Used- Adam
* Learning Rate- Default 0.001
* Validation Data split- 0.2
* Generator batch size= 64
* Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem).

After a lot of testing I found that with 64 batch size model took nearly 17 hr to train , but reducing the batch size resulted in faster training, but the model was not as accurate as before.



### Final result

My model performed well in first track and also performed well in second track, but the vehicle stops in the middle of the second track sometimes , I am not able to understand, why is it happening sometimes. As I trained with only steering angles, it is able to predict the steering angle but not the speed. I hope that might be the problem.

 



