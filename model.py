import os
import numpy as np

from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.metrics import Precision, Recall
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KIVY_NO_CONSOLELOG"] = "1"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

class modelHandler():
    def __init__(self, working_path):
        self.binary_cross_loss = tf.losses.BinaryCrossentropy()
        self.opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

        self.WORKING_PATH = working_path

        self.POS_PATH = os.path.join(self.WORKING_PATH, 'data', 'positive')
        self.NEG_PATH = os.path.join(self.WORKING_PATH, 'data', 'negative')
        self.ANC_PATH = os.path.join(self.WORKING_PATH, 'data', 'anchor')

        self.app_input_path = os.path.join(self.WORKING_PATH, 'verification_data', 'input_image')
        self.app_verification_path = os.path.join(self.WORKING_PATH, 'verification_data', 'verification_image')

        self.model_path = os.path.join(self.WORKING_PATH, 'model')
        self.checkpoint_path = os.path.join(self.model_path, 'training_checkpoints')

        self.siamese_model = self.getSiameseModel()

    def __make_embedding(self):
        inp = Input(shape=(100, 100, 3))

        c1 = Conv2D(64, (10, 10), activation='relu')(inp)
        m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

        c2 = Conv2D(128, (7, 7), activation='relu')(m1)
        m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

        c3 = Conv2D(128, (4, 4), activation='relu')(m2)
        m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

        c4 = Conv2D(256, (4, 4), activation='relu')(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation='sigmoid')(f1)

        return Model(inputs=[inp], outputs=[d1], name='embedding')

    def __make_siamese_model(self):
        input_image = Input(name='input_img', shape=(100, 100, 3))
        validation_image = Input(name='validation_img', shape=(100, 100, 3))

        embedding = self.__make_embedding()

        siamese_layer = L1Dist()
        siamese_layer._name = 'distance'
        distances = siamese_layer(embedding(input_image), embedding(validation_image))

        classifier = Dense(1, activation='sigmoid')(distances)

        return Model(inputs=[input_image, validation_image], outputs=[classifier], name='SiameseNetwork')

    def getSiameseModel(self):
        if os.path.exists(os.path.join(self.model_path, 'siamesemodel.h5')):
            model = tf.keras.models.load_model(os.path.join(self.model_path, 'siamesemodel.h5'),
                                               custom_objects={'L1Dist': L1Dist,
                                                               'BinaryCrossentrophy': tf.losses.BinaryCrossentropy}, compile=False)
            return model
        else:
            self.siamese_model = self.__make_siamese_model()
            self.saveModel()
            return self.siamese_model

    @tf.function
    def __train_step(self, batch):
        # record all of our operations
        with tf.GradientTape() as tape:
            # grabbing anc and pos/neg image
            X = batch[:2]
            # grabbing label
            y = batch[2]

            # forward pass
            yhat = self.siamese_model(X, training=True)
            # calculate loss
            loss = self.binary_cross_loss(y, yhat)

        grad = tape.gradient(loss, self.siamese_model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.siamese_model.trainable_variables))

        return loss

    def train(self, data, EPOCHS):
        if self.siamese_model is not None:
            checkpoint_prefix = os.path.join(self.checkpoint_path, 'ckpt')
            checkpoint = tf.train.Checkpoint(opt=self.opt, siamese_model=self.siamese_model)
            for epoch in range(1, EPOCHS + 1):
                print('\n Epoch {}/{}'.format(epoch, EPOCHS))
                print('')
                progbar = tf.keras.utils.Progbar(len(data))

                r = Recall()
                p = Precision()

                for idx, batch in enumerate(data):
                    loss = self.__train_step(batch)
                    yhat = self.siamese_model.predict(batch[:2])
                    r.update_state(batch[2], yhat)
                    p.update_state(batch[2], yhat)
                    progbar.update(idx + 1)
                    print('\n')
                    print(loss.numpy(), r.result().numpy(), p.result().numpy())
                    print('')
                    print('')
                if epoch % 10 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)
            self.saveModel()

    def saveModel(self):
        if not self.siamese_model is None:
            self.siamese_model.save(os.path.join(self.model_path, 'siamesemodel.h5'))

    def deleteModel(self):
        if os.path.exists(os.path.join(self.model_path, 'siamesemodel.h5')):
            os.remove(os.path.join(self.model_path, 'siamesemodel.h5'))
            self.siamese_model = None
            self.getSiameseModel()

    def predict_ver(self, input_img, validation_img):
        result = self.siamese_model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        return result

    def predict_test(self, test_input, validation_img):
        result = self.siamese_model.predict([test_input, validation_img])
        return result