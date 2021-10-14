import cv2
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class Video:
    """
    Class for storing video frames
    and meta
    """
    class VideoMetaInformation:
        def __init__(self, fps, height, width):
            self.fps = fps
            self.height = height
            self.width = width

    def __init__(self, frames: [], meta: VideoMetaInformation):
        self.frames = frames
        self.meta = meta


class VideoUtils:
    """
    This class contains some useful utils to work with videos.
    E.g. loading from file, reading VideoCapture
    (regardless to its origin), drawing utils
    """
    @staticmethod
    def transform_resolution(
            video: Video,
            new_height: int,
            new_width: int
    ) -> Video:
        for i in range(len(video.frames)):
            video.frames[i] = cv2.resize(
                src=video.frames[i],
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LANCZOS4
            )
        return video

    @staticmethod
    def save_video(
            video: Video,
            case: str,
            path: str,
            name='out.avi'
    ):
        """
        Save given video in 'case' way.
        :param video: video to save
        :param case: possible cases:
            'to frames':
                frames will be saved in path/frames/ dir
                with names equal to frame number in video
            'to video':
                form a video file and save to path/name
        :param path: ends with '/'
        :param name: should be given only if case='to video'
        :return: None
        """
        if not path.endswith('/'):
            warnings.warn('\npath should ends with /\n')
        if case == 'to frames':
            if not os.path.exists(path + '/frames'):
                warnings.warn('Given path does not exist')
                os.mkdir(path + '/frames')

            for i in range(len(video.frames)):
                cv2.imwrite(path + 'frames/' + str(i) + '.jpg', video.frames[i])

        elif case == 'to file':
            out = cv2.VideoWriter(
                filename=path + name,
                fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                fps=video.meta.fps,
                frameSize=(int(video.meta.width), int(video.meta.height))
            )
            for i in range(len(video.frames)):
                out.write(video.frames[i])
            out.release()

        else:
            warnings.warn('Unexpected case')

    @staticmethod
    def load_videocap_from_file(path: str) -> cv2.VideoCapture:
        """
        Make VideoCapture (stream) from video  file
        :param path: to video file
        :return: stream to read
        """
        videocap = cv2.VideoCapture(path)
        assert videocap.isOpened(), 'Error opening video stream or file'
        return videocap

    @staticmethod
    def read_videocap(videocap: cv2.VideoCapture) -> Video:
        """
        Collect frames from VideoCapture (stream) and
        form Video  instance
        :param videocap: stream to read
        :return: formed Video with filled frames
        """
        frames = []
        meta = Video.VideoMetaInformation(
            fps=videocap.get(cv2.CAP_PROP_FPS),
            height=videocap.get(4),
            width=videocap.get(3)
        )

        while videocap.isOpened():
            stat, frame = videocap.read()
            if stat is True:
                frames.append(frame)
            else:
                break
        videocap.release()

        return Video(frames, meta)

    @staticmethod
    def draw_probs_rects_whole_frames(
            video: Video,
            probs: [],
            labels: []
    ) -> Video:
        """
        Process all frames in Video and draw calculated
        labels with probs
        :param video:
        :param probs: list of probs, each prob is int from 0 to 100
        :param labels: list of text labels
        :return: processed video
        """
        assert len(probs) == len(video.frames), 'probs number should equal frames number'
        frames = video.frames.copy()
        meta = video.meta
        for i in range(len(frames)):
            VideoUtils.draw_prob_rect_on_frame(
                frame=frames[i],
                text=labels[i],
                prob=probs[i]
            )
        return Video(frames, meta)

    @staticmethod
    def draw_prob_rect_on_frame(
            frame: np.array,
            text: str,
            prob: int
    ) -> np.array:
        """
        This method draws white rectangular with text
        information about detected class and its probability
        :param frame:
        :param text: label name
        :param prob: label probability [0..100]
        :return: frame with label in upper left pos
        """
        epsilon = 0.15
        height, width = frame.shape[1], frame.shape[0]

        # TODO: find coefficients to rectangular shape 144x255
        p1 = int(height * (1 - epsilon * 2)), int(width * 0.0)
        p2 = int(height * 1.0), int(width * epsilon)
        cv2.rectangle(
            img=frame,
            pt1=p1,
            pt2=p2,
            color=(255, 255, 255),
            thickness=-1
        )
        cv2.rectangle(
            img=frame,
            pt1=p1,
            pt2=p2,
            color=(0, 0, 0),
            thickness=5
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (int(height * (1 - epsilon * 1.5)), int(width * epsilon * 0.4))
        cv2.putText(
            img=frame,
            text=text,
            org=text_pos,
            fontFace=font,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        text_pos = (int(height * (1 - epsilon * 1.2)), int(width * epsilon * 0.75))
        cv2.putText(
            img=frame,
            text='{} %'.format(prob),
            org=text_pos,
            fontFace=font,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        return frame

    # TODO: probably delete this
    def draw_inner_image(self, frame: np.array, img: np.array, pos: str) -> np.array:
        pass

    # TODO: deal with this method
    @staticmethod
    def draw_horizontal_chart(
            frame: np.array,
            probs: np.array,
            class_names: [str],
            dpi=300
    ) -> np.array:
        fig, axes = plt.subplots(figsize=(frame.shape[0] / dpi, 2 * frame.shape[0] / dpi))
        sns.set_color_codes("pastel")
        sns.set_color_codes("pastel")
        sns.barplot(x=probs, y=class_names, label="Probability", color="b")
        axes.legend(ncol=2, loc="lower right", frameon=True)


def process():
    """
    Testing method
    """
    video = VideoUtils.read_videocap(
        VideoUtils.load_videocap_from_file('video2.mov')
    )
    probs = []
    for i in range(len(video.frames)):
        probs.append(i)

    video = VideoUtils.draw_probs_rects_whole_frames(
        video=video,
        probs=probs,
        labels=['outdoor' for _ in range(len(probs))]
    )
    video.save(case='to_video', path='.')


if __name__ == '__main__':
    process()
