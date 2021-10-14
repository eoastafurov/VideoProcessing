import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from tqdm import tqdm


class Model:
    """
    Class for operations with frames classification net
    """
    class ModelFrameOutput:
        """
        Class for storing model output for each frame.
        P.S. on this stage it contains only probabilities
        """
        def __init__(
                self,
                probs: [],
        ):
            self.probs = probs

    def __init__(
            self,
            path: str,
            height=128,
            width=128,
            class_names=None
    ):
        """
        :param path: dir with model assets, weights and meta
        :param height: input layer dim
        :param width: should equals height
        :param class_names: labels in order model was taught on
            default value is ['indoor', 'outdoor']
        """
        if class_names is None:
            class_names = [
                'indoor',
                'outdoor',
            ]

        self.class_names = class_names
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width

        self.model = keras.models.load_model(path)
        self.probability_model = tf.keras.Sequential(
            [
                self.model,
                tf.keras.layers.Softmax()
            ]
        )

    def process_image(
            self,
            img: np.array
    ) -> ModelFrameOutput:
        """
        Get model prediction on this image
        :param img: np.array regardless to its shape
        :return: ModelFrameOutput
        """
        if img.shape[0] != self.IMG_HEIGHT \
                or img.shape[1] != self.IMG_WIDTH:
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))

        probs = self.probability_model.predict((np.expand_dims(img, 0)))
        return self.ModelFrameOutput(probs)

    def process_frames_sequence(
            self,
            frames: []
    ) -> [ModelFrameOutput]:
        """
        Get model predictions for all frames in video.
        \nP.S. Without smoothing, just probs sequence
        \nP.S. To use smoothing take look at
        VideoProcessing SmoothingTools
        :param frames: images
        :return: list of ModelFramesOutput's
        """
        return [
            self.process_image(frame) for frame in
            tqdm(frames, desc='Processing images')
        ]

    def convert_prediction_to_label(self, probs: []) -> str:
        """
        Get most likely label from prediction
        (list of probabilities)
        :param probs: list of labels probs
        :return: most likely label
        """
        return self.class_names[int(np.argmax(probs))]

    # TODO: function to process VideoCap
    # TODO: function to process video between timecodes
