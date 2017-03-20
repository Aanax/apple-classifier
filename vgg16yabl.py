# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''


from __future__ import print_function



import numpy as np
import warnings

import seaborn as sns

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float64"


from keras.models import Model, Sequential
from keras.layers import Flatten,Dropout, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input
from sklearn.externals import joblib
from scipy import misc 
import os
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
from keras.preprocessing import image
from PIL import Image
import shutil
import six

from keras.regularizers import l2, activity_l2



TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=False, weights='imagenet',
          input_tensor=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)


    return model



def sequential_from_Model(m,n_classes):

    s=Sequential()
    s.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',W_regularizer=l2(0.05),input_shape = (3, 224, 224)))
    s.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.05), name='block1_conv2'))
    s.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # Block 2
    s.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.02), name='block2_conv1'))
    s.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.02), name='block2_conv2'))
    s.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # Block 3
    s.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.02), name='block3_conv1'))
    s.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.02) ,name='block3_conv2'))
    s.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.03) ,name='block3_conv3'))
    s.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # Block 4
    s.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.03), name='block4_conv1'))
    s.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.03), name='block4_conv2'))
    s.add( Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.05), name='block4_conv3'))
    s.add( MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    # Block 5
    s.add( Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.03), name='block5_conv1'))
    s.add( Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.05), name='block5_conv2'))
    s.add( Convolution2D(512, 3, 3, activation='relu', border_mode='same',W_regularizer=l2(0.05), name='block5_conv3'))
    s.add( MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    


    #s.set_weights(m.get_weights())
    


    s.add(Flatten())
    

    s.add(Dense(100,activation = 'sigmoid',W_regularizer=l2(0.05),activity_regularizer=activity_l2(0.05)))

    s.add(Dense(30,activation = 'sigmoid',W_regularizer=l2(0.05),activity_regularizer=activity_l2(0.05)))
    s.add(Dropout(0.5))
    s.add(Dense(n_classes,activation='softmax',W_regularizer=l2(0.05),activity_regularizer=activity_l2(0.05)))

    
    


    return s





def trainme(add_model,path,n_classes,n_epochs):
    #trains 

    opt = SGD(lr = 0.001, decay = 0.00001)

    add_model.compile(optimizer = opt, loss='categorical_crossentropy',metrics = ['accuracy'])
    
    print("ALADsaf")
    
    iteration = 0
    trnbl =0
    accs=[0]
    res=0
    dirs=[]
    pathn=[]
    Namefils=[]
    
    print("FOLDERS----",os.listdir(path))
    for folder in os.listdir(path):
        pathn.append(path+folder)
    #path2 = path+"Cropped3/"

    print("PATHS----",pathn)
    for i in pathn:
        dirs.append(os.listdir(i))

    #print("DIRS--------------",dirs)
    print("DIRLEN",len(dirs))





    for epoch in range(n_epochs):
        files = six.moves.zip_longest(*dirs)
        
        for filename in files:
            Namefils=[]
            for j in range(0,len(pathn)):
                if filename[j] is not None:
                    Namefils.append(os.path.join(pathn[j],filename[j]))

            for Namefil in Namefils:
                if not os.path.isfile(Namefil):
                    print("DENIED  " + Namefil)
                    #print ("No file "+Namefil)
                    continue

                iteration+=1
                imag = image.load_img(Namefil, target_size=(224, 224))
                imag = image.img_to_array(imag)

                imag = np.expand_dims(imag, axis=0)
                imag = preprocess_input(imag)

                #print("n_classes",n_classes)
                label=np.zeros((n_classes))
                #print("Zeros",label)
                for i in range(0,len(pathn)):
                    if pathn[i] in Namefil:
                        label[i] = 1.0

                label = np.reshape(label,(-1,n_classes))
  
                    for k in range(0,len(add_model.layers)):
                        if add_model.layers[k].trainable == True and k >= 2 and add_model.layers[k-1].trainable == False:
                            add_model.layers[k-1].trainable=True
                            add_model.layers[k-2].trainable=True
                            trnbl+=2
                            weights = add_model.get_weights()
                            print("TRYING TO PUT IN!!!",trnbl)
                            add_model.compile(optimizer = opt, loss='categorical_crossentropy',metrics = ['accuracy'])
                            add_model.set_weights(weights)
       


                if iteration % 70 == 0:
                    res = testme(add_model,"/home/aanax/Desktop/YABLOKI/Totest/")
                    accs.append(res)
                    if res > 0.650:
                        print("OVER65!!!!!!!!!!!!!!!!!!!!")
                        if (res)>max(accs):
                            print("SAVING!!!!!!!)))))!001!!" , max(accs))
                            joblib.dump(add_model.get_weights(),"OVER65.pkl",compress =9 )







                print(add_model.train_on_batch(imag,label))
                
                print("Iteration",iteration)
      
                print("CURRENT LR ", K.get_value(opt.lr))
                print("CURRENT SCORE", res)
                

        K.set_value(opt.lr, 0.5 * K.get_value(opt.lr))

             
    sns.plt.plot(range(0,len(accs)),accs)
    sns.plt.show()
    return add_model


def sample(eps,pathto):
    #moves pics from train directory to tst
    print("Sampling test...")
    path = "/home/aanax/Desktop/YABLOKI/Totrain/"
    kol1max = 76
    totalvid = 315
    heal=0
    ill=0
    #eps=0.2
    var =['healthy','ill']
    for kol in range(0,kol1max):
        for i in range(0,totalvid,45):
            for name in var:

                Namefil = path + name+"_fan/"+name+str(kol)+'_'+str(i)+'.png'
                if not os.path.isfile(Namefil):
                    #print ("No file "+Namefil)
                    continue
               
                if np.random.random() < eps:
                    shutil.move(Namefil,pathto)
                    if name == 'healthy':
                        heal +=1
                    else:
                        ill+=1
    print("Heal",heal)
    print("Ill",ill)

def testme(seqmodel,path):
    #tests model
    
    kolmax = 76
    totalvid =315
    ok=0.0
    notok=0.0
    tp=0.0
    fp=0.0
    tn=0.0
    fn=0.0
    correct=0
    iteration = 0
    #res=0
    dirs=[]
    pathn=[]
    Namefils=[]
    
    print("FOLDERS----",os.listdir(path))
    for folder in os.listdir(path):
        pathn.append(path+folder)
    #path2 = path+"Cropped3/"

    print("PATHS----",pathn)
    for i in pathn:
        dirs.append(os.listdir(i))

    print("DIRLEN",len(dirs))

    files = six.moves.zip_longest(*dirs)
    
    for filename in files:
        Namefils=[]
        for j in range(0,len(pathn)):
            if filename[j] is not None:
                Namefils.append(os.path.join(pathn[j],filename[j]))

        for Namefil in Namefils:
            if not os.path.isfile(Namefil):
                print("DENIED  " + Namefil)
                #print ("No file "+Namefil)
                continue

            iteration+=1
            imag = image.load_img(Namefil, target_size=(224, 224))
            imag = image.img_to_array(imag)

            imag = np.expand_dims(imag, axis=0)
            imag = preprocess_input(imag)
                
       
            pr=seqmodel.predict(imag)
            #print(pr)
            print("Iteration",iteration)
            
 
            for i in range(0,len(pathn)):
                if pathn[i] in Namefil and np.argmax(pr[0])==i:
                    correct+=1.0
                           

    print("Accuracy",correct/iteration)
    return(correct/iteration)
    
    
if __name__ == '__main__':
    model = VGG16(include_top=False, weights='imagenet')

    n_classes=3
    n_epochs=5

    seqmodel = sequential_from_Model(model, n_classes)


    if not os.path.isfile('rewrite.pkl'):
        print("TRAINING")
        
        path = "/home/aanax/Desktop/YABLOKI"

        if not os.path.isdir(path+"tst"):
            print("No tst directory")
        



        path = "/home/aanax/Desktop/YABLOKI/Totrain/"


        seqmodel = trainme(seqmodel,path,n_classes,n_epochs)





        train_datagen = image.ImageDataGenerator(
            horizontal_flip=False)
     
        


        train_generator = train_datagen.flow_from_directory(
        #"/media/aanax/5ADEB1D6DEB1AAA1/Documents and Settings/Andrew/Desktop/MACHINE_LEARNING/CARDIOHER (1)/ToMakeDataset/PNG",
        "/home/aanax/Desktop/YABLOKI/Totrain/", 
        target_size= (224,224) ,
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical')


        print("Success")

        joblib.dump(seqmodel.get_weights(),"rewrite.pkl",compress=9)



    else:
        print("Testing model")



        opt = SGD(lr = 0.001, decay = 0.00001)

        seqmodel.compile(optimizer = opt, loss='categorical_crossentropy',metrics = ['accuracy'])

        seqmodel.set_weights(joblib.load('rewrite.pkl'))


        pass

        #Totest
        path = "/home/aanax/Desktop/YABLOKI/Totest/"
        
        if not os.path.isdir(path+"tst"):
            print("No tst dir")
            #os.makedirs(path+"tst")
            #sample(0.2,path+"tst")
        else:
            print("Already sampled!!!!")

        acc1 = testme(seqmodel,path)
        acc2 = testme(seqmodel,"/home/aanax/Desktop/YABLOKI/Totrain/")

        print("TESTSCORE",acc1)
        print("TRAINSCORE",acc2)



