# _*_ coding:utf-8 _*_
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
from keras import callbacks
from keras.optimizers import Adam
from data_generate import Generate
#from resnet50 import res_Net50
from models.sepwised_Conv import res_Net50
from keras.utils import multi_gpu_model
#from models.lambda_net_3 import lambda_net
#from models.sepwised_group_conv import res_Net50

class ModelTrain():
    def __init__(self,log_dir='./log/02',batch_size=32):
        self.log_dir=log_dir
        self.batch_size=batch_size
        self.input=Input(shape=[240,320,25])
        self.ge=Generate(batch_size=self.batch_size)
        pass

    def train(self):
        train_examples=self.ge.get_TrainExamples()
        test_examples=self.ge.get_TestExamples()

        model=res_Net50(self.input,classes=self.ge.get_Classes())
        # 实现gpu并行
        #model = multi_gpu_model(model, gpus=2)
        logging = callbacks.TensorBoard(log_dir=self.log_dir)
        checkpoint = callbacks.ModelCheckpoint(
           self.log_dir + '/' + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
           monitor='val_loss', save_best_only=True, mode='min',
           save_weights_only=True, period=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=9, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=19, verbose=1)

        # 指定训练方式
        # for i in range(len(model.layers)-3):
        #     model.layers[i].trainable = False
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        if os.path.exists(self.log_dir + '/' + 'train_weights.h5'):
            model.load_weights( self.log_dir + '/' + 'train_weights.h5',by_name=True)
            #model.load_weights()

        model.fit_generator(self.ge.train_Generate(),
                            steps_per_epoch=max(1, train_examples // self.batch_size),
                            validation_data=self.ge.test_Generate(),
                            validation_steps=max(1, test_examples // self.batch_size),
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])#
        model.save_weights(self.log_dir + '/' + 'train_weights.h5')
        model.save(self.log_dir + '/' + 'train_models.h5')

if __name__=='__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    train=ModelTrain()
    train.train()
