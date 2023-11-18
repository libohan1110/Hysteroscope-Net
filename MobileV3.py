import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
list_1 = os.listdir('0/')
img_552= []
for img_ in list_1:
    imgname = '0/' + img_
    img = image.load_img(imgname,target_size=(336,672))
    img = image.img_to_array(img)
    img_552.append(img)
list_2 = os.listdir('1/')
img_553= []
for img_ in list_2:
    imgname = '1/' + img_
    img = image.load_img(imgname,target_size=(336,672))
    img = image.img_to_array(img)
    img_553.append(img)
list_3 = os.listdir('2/')
img_554= []
for img_ in list_3:
    imgname = '2/' + img_
    img = image.load_img(imgname,target_size=(336,672))
    img = image.img_to_array(img)
    img_554.append(img)
img_552 = np.array(img_552)
img_553 = np.array(img_553)
img_554 = np.array(img_554)
syn = pd.read_excel('Syntax1.xlsx',sheet_name=0)
syn = syn.sort_values(by=['No.'])
y_552 = np.array(syn.iloc[:,::2])
syn1 = pd.read_excel('Syntax2.xlsx',sheet_name=0)
syn1 = syn1.sort_values(by=['No.'])
y_553 = np.array(syn1.iloc[:,::2])
syn2 = pd.read_excel('Syntax3.xlsx',sheet_name=0)
syn2 = syn2.sort_values(by=['No.'])
y_554 = np.array(syn2.iloc[:,::2])
train_x=img_552.copy()
train_y_=y_552.copy()
train_y=train_y_[:,1]
valid_x=img_553.copy()
valid_y_=y_553.copy()
valid_y=valid_y_[:,1]
test_x=img_554.copy()
test_y_=y_554.copy()
test_y=test_y_[:,1]
train_x_ = train_x.astype("float32") * 1/255
train_x_ = train_x_[:,::2,::2]
test_x_ = test_x.astype("float32") * 1/255
test_x_ = test_x_[:,::2,::2]
valid_x_ = valid_x.astype("float32") * 1/255
valid_x_ = valid_x_[:,::2,::2]
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
conv_base = MobileNetV3Small(include_top=False, weights="imagenet", input_shape=(168,336,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation ='selu'))
model.add(layers.Dense(1, activation ='sigmoid'))
model.compile(loss='binary_crossentropy', metrics='acc')
conv_base.trainable = False
model.fit(train_x_,train_y,epochs=300,batch_size=16,
         validation_data=(test_x_,test_y),
         validation_batch_size=16)
input = test_x_
last_conv_layer = model.get_layer('dense')
iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
model_out, last_conv_layer = iterate(input)
outcomes = np.array(last_conv_layer)
np.save('test.npy', outcomes)
model.save("output/qingliangji/mobileV3-1.h5")
conv_base.save("output/qingliangji/mobileV3conv_base.h5")
result=model.predict(valid_x_)
result.reshape(102,)
result=pd.DataFrame(result)
test_result=pd.DataFrame(test_y_)
test_ROC=pd.concat([test_result,result], axis=1)
test_ROC.to_csv('output/test_ROC.csv')