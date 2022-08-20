import cv2
import os

import numpy as np

import tarfile
import uuid
import shutil
import random

import requests

from keras.metrics import Precision, Recall
import tensorflow as tf

from kivy.core.window import Window
Window.hide()
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KIVY_NO_CONSOLELOG"] = "1"

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from model import modelHandler
import mediapipe

from twilio.rest import Client

import configparser



#modification params
gNAME = 'FaceID'
gPARENT_PATH = r'C:\Users\lucab\Desktop'
gDETECTION_THRESHOLD = 0.5
gVERIFICATION_THRESHOLD = 0.7
gnumVerImg = 100
gEPOCHS = 2
gnumAncPosImg = 100
gnumPuffer = 5
gSendSMS = 1





account_sid = 'AC324b9368ae996de4a39f2c3ed5ddcf75'
auth_token = '36daf680edbc6535ac644f70390aa3bc'
client = Client(account_sid, auth_token)

gWORKING_PATH = r''

gPOS_PATH = r''
gNEG_PATH = r''
gANC_PATH = r''

gapp_input_path = r''
gapp_verification_path = r''

gmodel_path = r''
gcheckpoint_path = r''


Window.set_icon("icon_request.svg")

# Window.borderless = True

def ConfigSetup(NAME):
    config = configparser.ConfigParser()
    PARENT_PATH = str(gPARENT_PATH)
    DETECTION_THRESHOLD = str(gDETECTION_THRESHOLD)
    VERIFICATION_THRESHOLD = str(gVERIFICATION_THRESHOLD)
    numVerImg = str(gnumVerImg)
    EPOCHS = str(gEPOCHS)
    numAncPosImg = str(gnumAncPosImg)
    numPuffer = str(gnumPuffer)
    SendSMS = str(gSendSMS)

    if not os.path.exists('conf.ini'):
        config[NAME] = {}
        config[NAME]['PARENT_PATH'] = PARENT_PATH
        config[NAME]['DETECTION_THRESHOLD'] = DETECTION_THRESHOLD
        config[NAME]['VERIFICATION_THRESHOLD'] = VERIFICATION_THRESHOLD
        config[NAME]['numVerImg'] = numVerImg
        config[NAME]['EPOCHS'] = EPOCHS
        config[NAME]['numAncPosImg'] = numAncPosImg
        config[NAME]['numPuffer'] = numPuffer
        config[NAME]['SendSMS'] = SendSMS
        with open('conf.ini', 'w') as configfile:
            config.write(configfile)

        PARENT_PATH = str(gPARENT_PATH)
        DETECTION_THRESHOLD = float(gDETECTION_THRESHOLD)
        VERIFICATION_THRESHOLD = float(gVERIFICATION_THRESHOLD)
        numVerImg = int(gnumVerImg)
        EPOCHS = int(gEPOCHS)
        numAncPosImg = int(gnumAncPosImg)
        numPuffer = int(gnumPuffer)
        SendSMS = int(gSendSMS)

    else:
        config.read('conf.ini')
        config.sections()
        PARENT_PATH = config[gNAME]['PARENT_PATH']
        DETECTION_THRESHOLD = float(config[gNAME]['DETECTION_THRESHOLD'])
        VERIFICATION_THRESHOLD = float(config[gNAME]['VERIFICATION_THRESHOLD'])
        numVerImg = int(config[gNAME]['numVerImg'])
        EPOCHS = int(config[gNAME]['EPOCHS'])
        numAncPosImg = int(config[gNAME]['numAncPosImg'])
        numPuffer = int(config[gNAME]['numPuffer'])
        SendSMS = int(config[gNAME]['SendSMS'])

    return PARENT_PATH, DETECTION_THRESHOLD, VERIFICATION_THRESHOLD, numVerImg, EPOCHS, numAncPosImg, numPuffer, SendSMS

def FolderSetup(PARENT_PATH, NAME):
    print('Setting up work directory')
    WORKING_PATH = os.path.join(PARENT_PATH, NAME)
    if not os.path.exists(WORKING_PATH):
        os.makedirs(WORKING_PATH)

    POS_PATH = os.path.join(WORKING_PATH, 'data', 'positive')
    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)

    NEG_PATH = os.path.join(WORKING_PATH, 'data', 'negative')
    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)

    if not os.listdir(NEG_PATH):

        if not os.path.exists('lfw'):
            fname = os.path.join(WORKING_PATH, 'data', "lfw.tgz")
            fname = fname.replace('\\', chr(92))
            if not os.path.exists(fname):
                URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
                response = requests.get(URL)
                with open(fname, 'wb') as f:
                    f.write(response.content)
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(os.path.join(WORKING_PATH, 'data'))
            tar.close()

        for directory in os.listdir(os.path.join(WORKING_PATH, 'data', 'lfw')):
            for file in os.listdir(os.path.join(WORKING_PATH, 'data', 'lfw', directory)):
                EX_PATH = os.path.join(WORKING_PATH, 'data', 'lfw', directory, file)
                NEW_PATH = os.path.join(NEG_PATH, file)
                try:
                    os.replace(EX_PATH, NEW_PATH)
                except:
                    pass
        shutil.rmtree(os.path.join(os.path.join(WORKING_PATH, 'data', 'lfw')))

    ANC_PATH = os.path.join(WORKING_PATH, 'data', 'anchor')
    if not os.path.exists(ANC_PATH):
        os.makedirs(ANC_PATH)

    app_input_path = os.path.join(WORKING_PATH, 'verification_data', 'input_image')
    if not os.path.exists(app_input_path):
        os.makedirs(app_input_path)

    app_verification_path = os.path.join(WORKING_PATH, 'verification_data', 'verification_image')
    if not os.path.exists(app_verification_path):
        os.makedirs(app_verification_path)

    model_path = os.path.join(WORKING_PATH, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpoint_path = os.path.join(model_path, 'training_checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return (WORKING_PATH, POS_PATH, NEG_PATH, ANC_PATH, app_input_path, app_verification_path, model_path, checkpoint_path)

gWORKING_PATH, gPOS_PATH, gNEG_PATH, gANC_PATH, gapp_input_path, gapp_verification_path, gmodel_path, gcheckpoint_path = FolderSetup(PARENT_PATH=gPARENT_PATH, NAME=gNAME)
gPARENT_PATH, gDETECTION_THRESHOLD, gVERIFICATION_THRESHOLD, gnumVerImg, gEPOCHS, gnumAncPosImg, gnumPuffer, gSendSMS = ConfigSetup(gNAME)

# build app and layout
class FaceIDAppMain(App):
    def __init__(self,  **kwargs):
        super(FaceIDAppMain, self).__init__()
        global gWORKING_PATH
        global gapp_input_path
        global gapp_verification_path
        global gDETECTION_THRESHOLD
        global gVERIFICATION_THRESHOLD
        global gnumVerImg
        global gEPOCHS
        global gnumAncPosImg
        global gNAME
        global gSendSMS
        global gnumPuffer


        self.mpFaceDetection = mediapipe.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)
        self.mpDraw = mediapipe.solutions.drawing_utils

        self.capture = cv2.VideoCapture(0)

        self.handler = None

    def build(self):

        self.title = gNAME

        self.web_cam = Image(size_hint=(1, .7))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        self.button2 = Button(text='Verify', size_hint=(1, .1), background_color=(0,1,0,1))
        self.button2.bind(on_press=self.verify)

        self.creator = Label(text="FaceID - Created by Luca Bäck", size_hint=(1, .1))
        self.info = Label(text="", size_hint=(1, .1))

        self.button3 = Button(text="Train Siamese Network", size_hint=(.25, 1), background_color=(0, 0, 1, 1))
        self.button3.bind(on_press=self.TrainModel)

        self.button4 = Button(text="Test Siamese Network", size_hint=(.25, 1), background_color=(0, 0, 1, 1))
        self.button4.bind(on_press=self.getStats)

        self.button1 = Button(text='Set New Face', size_hint=(.5, 1), background_color=(1, 0, 0, 1))
        self.button1.bind(on_press=self.setNewFace)

        floatlayout = BoxLayout(orientation='horizontal', size_hint=(1, .1))
        floatlayout.add_widget(self.button1)
        floatlayout.add_widget(self.button3)
        floatlayout.add_widget(self.button4)



        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.creator)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.info)
        layout.add_widget(floatlayout)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button2)

        Clock.schedule_interval(self.update, 1.0 / 33.0)

        self.handler = modelHandler(gWORKING_PATH)

        return layout

    def on_start(self):
        results, verified = None, None
        while results is None and verified is None:
            results, verfified = self.verify()
        Window.show()
        #Window.size = (900, 600)
        Window.raise_window()

    def update(self, *args):
        ret, frame = self.capture.read()
        bbox, frame = self.findFaces(frame, True)

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        self.info.text = ""
        SAVE_PATH = os.path.join(gapp_input_path, 'input_image.jpg')
        ret, frame = self.capture.read()
        bbox, frame = self.findFaces(frame, False)
        frame = self.cropFrame(bbox, frame)
        if frame is False:
            self.info.text = "No Face Detected"
            return None, None

        cv2.imwrite(SAVE_PATH, frame)
        results = []
        for image in os.listdir(gapp_verification_path):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join(gapp_verification_path, image))

            result = self.handler.predict_ver(input_img, validation_img)
            results.append(result)

        detection = np.sum(np.array(results) > gDETECTION_THRESHOLD)
        verification = 0
        if(len(os.listdir(gapp_verification_path)) != 0):
            verification = detection / len(os.listdir(gapp_verification_path))
        verfified = verification > gVERIFICATION_THRESHOLD

        if verfified:
            self.verification_label.text = 'Verified'
            Window.set_icon("icon_complet.svg")
            Window.clearcolor =(0, 1, 0, 0)
            if gSendSMS:
                try:
                    message = client.messages \
                        .create(
                        body='Hey Luca, dein Computer wird gerade von einer verifizierten Person benutzt',
                        from_='+16693222484',
                        to='+4915163496563'
                    )
                except:
                    pass
            print('')
            print('Verified')
            print('')
        else:
            self.verification_label.text = 'Unverified'
            Window.set_icon("icon_request.svg")
            Window.clearcolor =(1, 0, 0, 0)
            if gSendSMS:#
                try:
                    message = client.messages \
                        .create(
                        body='Hey Luca, dein Computer wird gerade von einer nicht verifizierten Person benutzt',
                        from_='+16693222484',
                        to='+4915163496563'
                    )
                except:
                    pass
            print('')
            print('Unverified')
            print('')


        return results, verfified

    def findFaces(self, frame, draw):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        rect = None
        if (results.detections):

            h, w, c = frame.shape

            bboxC = results.detections[0].location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h),


            #position of rectangle
            scale_factor = 1.75

            width = bbox[2] * scale_factor
            height = bbox[3] * scale_factor

            diff_width = width - bbox[2]
            diff_width = diff_width / 2

            pos_x = bbox[0] - diff_width

            diff_height = height - bbox[3]
            diff_height = diff_height / 1.25

            pos_y = bbox[1] - diff_height

            if (pos_y + height) > h:
                pos_y = h - height
            if (pos_y < 0):
                pos_y = 0

            if (height > h):
                height = h
            if (width > w):
                width = w

            if (pos_x + width) > w:
                pos_x = w - width
            if(pos_x < 0):
                pos_x = 0


            rect = int(pos_x), int(pos_y), \
                   int(width), int(height),

            if (draw):
                cv2.rectangle(frame, rect, (255, 0, 255), 2)
                cv2.circle(frame, (rect[0], rect[1]), 1, (255,255,0), 4)
                cv2.circle(frame, (rect[0], rect[1] + rect[3]), 1, (255, 255, 0), 4)
                cv2.circle(frame, (rect[0] + rect[2], rect[1]), 1, (255, 255, 0), 4)
                cv2.circle(frame, (rect[0] + rect[2], rect[1] + rect[3]), 1, (255, 255, 0), 4)

        return rect, frame

    def cropFrame(self, bbox, frame):
        if not bbox:
            return False
        frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]: bbox[0] + bbox[2], :]

        frame = cv2.resize(frame, (250,250))
        return frame

    def saveIMG(self, imgname):
        ret, frame = self.capture.read()
        bbox, frame = self.findFaces(frame, False)
        frame = self.cropFrame(bbox, frame)
        if frame is False:
            return False
        cv2.imwrite(imgname, frame)
        return True

    def setNewFace(self, *args):
        for f in os.listdir(gANC_PATH):
            os.remove(os.path.join(gANC_PATH, f))
        for f in os.listdir(gPOS_PATH):
            os.remove(os.path.join(gPOS_PATH, f))
        for f in os.listdir(gapp_verification_path):
            os.remove(os.path.join(gapp_verification_path, f))

        for i in range(0, gnumAncPosImg+gnumPuffer):
            imgname = os.path.join(gANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            b = False
            while not b:
                try:
                    b = self.saveIMG(imgname)
                except:
                    pass
            imgname = os.path.join(gPOS_PATH, '{}.jpg'.format(uuid.uuid1()))
            c = False
            while not c:
                try:
                    c = self.saveIMG(imgname)
                except:
                    pass
            self.info.text = "Took {}/{} Pictures".format(i+1, gnumAncPosImg+gnumPuffer)
            print("Took {}/{} Pictures".format(i+1, gnumAncPosImg+gnumPuffer))

        print('Creating variation images')

        for file_name in os.listdir(os.path.join(gPOS_PATH)):
            img_path = os.path.join(gPOS_PATH, file_name)
            img = cv2.imread(img_path)
            augmented_images = self.data_aug(img)

            for image in augmented_images:
                cv2.imwrite(os.path.join(gPOS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

        for file_name in os.listdir(os.path.join(gANC_PATH)):
            img_path = os.path.join(gANC_PATH, file_name)
            img = cv2.imread(img_path)
            augmented_images = self.data_aug(img)

            for image in augmented_images:
                cv2.imwrite(os.path.join(gANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

        print('Getting poitive images for app verification')

        pos_imgs = []
        counter = 0
        while counter < gnumVerImg:
            file = random.choice(os.listdir(gPOS_PATH))
            if not file in pos_imgs:
                pos_imgs.append(file)
                counter = counter + 1
        for file in pos_imgs:
            shutil.copy(os.path.join(gPOS_PATH, file), os.path.join(gapp_verification_path, file))

        self.handler.deleteModel()

    def preprocess_twin(self, input_img, validation_img, label):
        return (self.preprocess(input_img), self.preprocess(validation_img), label)

    def data_aug(self, img):
        data = []
        for i in range(9):
            img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
            img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
            # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
            img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
            img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100,
                                                         seed=(np.random.randint(100), np.random.randint(100)))
            img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                       seed=(np.random.randint(100), np.random.randint(100)))

            data.append(img)

        return data

    def getData(self, *args):
        if (len(os.listdir(gPOS_PATH)) < gnumAncPosImg*9 or len(os.listdir(gANC_PATH)) < gnumAncPosImg*9):
            self.info.text = "Not enough POSITIVE or ANCHOR images available"
            return None, None

        anchor = tf.data.Dataset.list_files(gANC_PATH + '\*.jpg').take(gnumAncPosImg * 9)
        positive = tf.data.Dataset.list_files(gPOS_PATH + '\*.jpg').take(gnumAncPosImg * 9)
        negative = tf.data.Dataset.list_files(gNEG_PATH + '\*.jpg').take(gnumAncPosImg * 9)

        positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
        negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
        data = positives.concatenate(negatives)

        data = data.map(self.preprocess_twin)
        data = data.cache()
        data = data.shuffle(buffer_size=1024)

        train_data = data.take(round(len(data) * .7))
        train_data = train_data.batch(16)
        train_data = train_data.prefetch(8)

        test_data = data.skip(round(len(data) * .7))
        test_data = test_data.take(round(len(data) * .3))
        test_data = test_data.batch(16)
        test_data = test_data.prefetch(8)

        return train_data, test_data

    def TrainModel(self, *args):
        self.info.text = "Training..."
        train_data, test_data = self.getData()
        if(train_data is None):
            return None
        self.handler.train(train_data, gEPOCHS)
        self.info.text = ""

    def getStats(self, *args):
        train_data, test_data = self.getData()
        if(test_data is None):
            return None
        test_input, test_val, y_true = test_data.as_numpy_iterator().next()
        yhat = self.handler.predict_test(test_input, test_val)
        r = Recall()
        r.update_state(y_true, yhat)
        rec = r.result().numpy()

        p = Precision()
        p.update_state(y_true, yhat)
        prec = p.result().numpy()

        self.info.text = 'Recall: ' + str(rec) + ' Precision: ' + str(prec)


if __name__ == '__main__':
    FaceIDAppMain().run()
