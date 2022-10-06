# import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock

from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
class CamApp(App):

    def build(self):
        self.web_cam = Image(size_hint = (1, .8))
        self.button = Button(text = 'Verify', on_press = self.verify, size_hint = (1, .1))
        self.verification_label = Label(text = 'Verification Uninitiated', size_hint = (1,.1))

        # Add items to layout
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)


        # Load keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, *args):
        # Read the frame from opencv
        ret, frame = self.capture.read()
        frame = frame[420:670, 970:1220, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
        self.web_cam.texture = img_texture

    #  Load image from file and convert 
    def preprocess(self, file_path  ):

        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        #Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0

        return img

    # Verification function to verify
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[420:670, 970:1220, :]
        cv2.imwrite(SAVE_PATH, frame)
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)
        
        detection = np.sum(np.array(results) > detection_threshold)
        
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verification == True else "Unverfied"
        return results, verified

if __name__ == '__main__':
    CamApp().run()
    