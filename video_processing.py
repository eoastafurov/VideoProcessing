from model_api import Model
from video_utils import VideoUtils
import numpy as np
from video_utils import Video


class VideoProcessing:
    """
    Tools for processing exact videos
    """
    class SmoothingTools:
        """
        Smoothing tools to operate with sequence of predictions
        """
        @staticmethod
        def moving_average(
                predictions: np.array,
                window_size=35
        ) -> []:
            """
            Apply moving average method to sequence of predictions.
            \nLook here: https://cutt.ly/0Ry9ETy
            :param predictions: np.array, dtype is [float]
            :param window_size: (ODD!) width of the window to smooth
            :return: smoothed predictions
            """
            assert window_size % 2 == 1, 'Window size should be odd'
            assert predictions.shape[0] != 0, 'Predictions list is empty'

            padding = int(window_size / 2)
            output = list(predictions[:padding])
            number_of_frames = predictions.shape[0]

            for i in range(padding, number_of_frames - padding):
                probs_slice = predictions[i - padding:][:window_size]
                averaged_proba = probs_slice.sum(axis=0) / probs_slice.shape[0]
                output.append(averaged_proba)

            for elem in predictions[number_of_frames - padding:]:
                output.append(elem)

            return np.array(output)

    @staticmethod
    def process(
            video: Video,
            model_path: str,
            smoothing_method=SmoothingTools.moving_average,
            draw=True,
            width=128,
            height=128,
            class_names=None
    ) -> ([], [], Model):
        """
        function to process given video,
        calculate predictions, draw predictions, ..
        :param video: instance of the Video to process
        :param model_path: path to model u want to use.
            P.S. manually saving
        :param smoothing_method: method from SmoothingMethods
            to smooth predictions
        :param draw: is it necessary to draw model output
            to file out.avi
        :param width: size of model input layer
        :param height: size of model input layer
        :param class_names: labels in order model was taught on
        :return: probabilities, labels, model
        """
        model = Model(model_path, height, width, class_names)

        model_output = model.process_frames_sequence(video.frames)

        probs_list = np.array([video_meta.probs for video_meta in model_output])

        probs_list = smoothing_method(probs_list)
        labels = [model.convert_prediction_to_label(probs) for probs in probs_list]

        if draw:
            video = VideoUtils.draw_probs_rects_whole_frames(
                video=video,
                probs=[int(100 * np.max(probs)) for probs in probs_list],
                labels=labels
            )
            VideoUtils.save_video(
                video=video,
                case='to file',
                path='./',
                name='out.avi'
            )

        return probs_list, labels, Model


def process():
    """
    Testing
    """
    video_path = 'video.mp4'
    model_path = 'model_IO_Baseline2/'

    video = VideoUtils.read_videocap(
        VideoUtils.load_videocap_from_file(video_path)
    )

    _, _, model = VideoProcessing.process(
        video,
        model_path,
        width=128,
        height=128,
        class_names=['indoor', 'outdoor']
    )


if __name__ == '__main__':
    process()
