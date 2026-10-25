# Vendored from duckietown/lib-dt-computer-vision, commit c5ae3a2de91d012a0c27d1a453e8a0ba9cbd1719.
# Only imports / annotation-only camera types adapted for standalone MaixPy.
# See UPSTREAM.json. Detection and projection methods are unchanged.
import dataclasses

import numpy as np
import cv2


@dataclasses.dataclass
class Detections:
    """
    This is a data class that can be used to store the results of the line detection procedure
    performed by :py:class:`LineDetector`.
    """
    # An ``Nx4`` array with every row representing a line ``[x1, y1, x2, y2]``
    lines: np.ndarray
    # An ``Nx2`` array with every row representing the normal of a line ``[nx, ny]``
    normals: np.ndarray
    # An ``Nx2`` array with every row representing the center of a line ``[cx, cy]``
    centers: np.ndarray
    # A binary map of the area from which the line segments were extracted
    map: np.ndarray


class ColorRange:
    """
    The Color Range class holds one or multiple color ranges. It can easily be generated with
    the :py:meth:`fromDict` class method and extends the `OpenCV's inRange <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981>`_
    method to work with multiple color ranges.

    All colours must be given in ``HSV`` space.

    Args:
        low (:obj:`numpy array`): An ``Nx3`` array with the low ends of ``N`` colour ranges.
        high (:obj:`numpy array`): An ``Nx3`` array with the high ends of ``N`` colour ranges.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high

    @classmethod
    def fromDict(cls, dictionary):
        """

        Generates a :py:class:`ColorRange` object from a dictionary. Expects the colors to be given in ``HSV`` space.
        If multi-entry ranges are provided (e.g. if you are interested in yellow and white), then each should have
        keys ``high_X`` and ``low_X``, see example bellow:

        Examples:

            Single-entry color range::

                { 'low': [0,0,150], 'high': [180,60,255] }

            Multi-entry color range::

                { 'low_1': [0,0,150], 'high_1': [180,60,255], 'low_2': [165,140,100], 'high_2': [180,255,255] }


        Args:
            dictionary (:obj:`dict`): The yaml dictionary describing the color ranges.

        Returns:
            :obj:`ColorRange`: the generated ColorRange object
        """

        # if only two entries: single-entry, if more: multi-entry
        if len(dictionary) == 2:
            assert "low" in dictionary, "Key 'low' must be in dictionary"
            assert "high" in dictionary, "Key 'high' must be in dictionary"
            low = np.array(dictionary["low"]).reshape((1, 3))
            high = np.array(dictionary["high"]).reshape((1, 3))

        elif len(dictionary) % 2 == 0:

            # make the keys tuples with `low` or `high` and the id of the entry
            dictionary = {tuple(k.split("_")): v for k, v in list(dictionary.items())}
            entry_indices = set([k[1] for k, _ in list(dictionary.items())])

            assert len(entry_indices) == len(dictionary) / 2, (
                "The multi-entry definition doesn't " "follow the requirements"
            )

            # build an array for the low and an array for the high range bounds
            low = np.zeros((len(entry_indices), 3))
            high = np.zeros((len(entry_indices), 3))
            for idx, entry in enumerate(entry_indices):
                low[idx] = dictionary[("low", entry)]
                high[idx] = dictionary[("high", entry)]

        else:
            raise ValueError(
                "The input dictionary has two have an even number of "
                "entries: a low and high value for each color range."
            )

        return cls(low=low, high=high)

    def inRange(self, image):
        """
        Applies the `OpenCV inRange
        <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981>`_
        method to every color range entry. Returns the bitwise OR of the results.
        In other words, returns a binary map with 1 for the pixels of the input image that fall
        in at least one of the color ranges.

        Args:
            image (:obj:`numpy array`): an ``HSV`` image

        Returns:
            :obj:`numpy array`: a two-dimensional binary map
        """

        selection = cv2.inRange(image, self.low[0], self.high[0])
        for i in range(1, len(self.low)):
            current = cv2.inRange(image, self.low[i], self.high[i])
            selection = cv2.bitwise_or(current, selection)

        return selection

    @property
    def representative(self):
        """
        Provides an representative color for this color range.
        This is the average color of the first range (if more than one ranges are set).

        Returns:
            :obj:`list`: a list with 3 entries representing an HSV color
        """

        return list(0.5 * (self.high[0] + self.low[0]).astype(int))

from abc import abstractmethod, ABC
from typing import Tuple, List

import cv2
import numpy as np


BGRImage = HSVImage = np.ndarray


class LineDetectorInterface(ABC):

    @abstractmethod
    def detect(self, image: BGRImage, color: ColorRange) -> Detections:
        """Returns a tuple of class Detections"""
        pass


class LineDetector(LineDetectorInterface):
    """
    The Line Detector can be used to extract line segments from a particular color range in
    an image. It combines edge detection, color filtering, and line segment extraction.

    This class was created for the goal of extracting the white, yellow, and red lines in the
    Duckiebot's camera stream as part of the lane localization pipeline. It is setup in a way
    that allows efficient detection of line segments in different color ranges.

    The detect function is called with an image color ranges. It converts it to `HSV color space
    <https://en.wikipedia.org/wiki/HSL_and_HSV>`_, which is much better for color segmentation,
    and applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_.

    Then, to do the actual line segment extraction, a call to :py:meth:`detect` with a
    :py:class:`ColorRange` object must be made.
    Multiple such calls with different colour ranges can be made and these will reuse the
    precomputed HSV image and Canny edges.

    Args:

        canny_thresholds (:obj:`list` of :obj:`int`): a list with two entries that specify the
        thresholds for the hysteresis procedure, details `here
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
        #canny>`__, default is ``[80, 200]``

        canny_aperture_size (:obj:`int`): aperture size for a Sobel operator, details `here
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
        #canny>`__, default is 3

        dilation_kernel_size (:obj:`int`): kernel size for the dilation operation which fills
        in the gaps in the color filter result, default is 3

        hough_threshold (:obj:`int`): Accumulator threshold parameter. Only those lines are
        returned that get enough votes, details `here
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
        =houghlinesp#houghlinesp>`__, default is 2

        hough_min_line_length (:obj:`int`): Minimum line length. Line segments shorter than that
        are rejected, details `here
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
        =houghlinesp#houghlinesp>`__, default is 3

        hough_max_line_gap (:obj:`int`): Maximum allowed gap between points on the same line to
        link them, details `here
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
        =houghlinesp#houghlinesp>`__, default is 1

    """

    def __init__(
            self,
            canny_thresholds=None,
            canny_aperture_size=3,
            dilation_kernel_size=3,
            hough_threshold=2,
            hough_min_line_length=3,
            hough_max_line_gap=1,
    ):
        self.canny_thresholds = canny_thresholds or [80, 200]
        self.canny_aperture_size = canny_aperture_size
        self.dilation_kernel_size = dilation_kernel_size
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

    @staticmethod
    def find_edges(image: BGRImage, canny_thr_low: int, canny_thr_high: int,
                   canny_aperture_size: int) -> np.ndarray:
        """
        Applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_
        to a ``BGR`` image.

        Args:
            image: (:obj:`numpy array`): BGR image to process.

            canny_thr_low (:obj:`int`): low threshold for the hysteresis procedure, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
            #canny>`__.

            canny_thr_high (:obj:`int`): high threshold for the hysteresis procedure, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
            #canny>`__.

            canny_aperture_size (:obj:`int`): aperture size for a Sobel operator, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
            #canny>`__.

        Returns:
            :obj:`numpy array`: a binary image with the edges
        """
        # perform edge detection
        edges = cv2.Canny(
            image,
            canny_thr_low,
            canny_thr_high,
            apertureSize=canny_aperture_size,
        )
        return edges

    @staticmethod
    def hough_line(edges: np.ndarray, hough_threshold: int,
                   hough_min_line_length: int,
                   hough_max_line_gap: int, ) -> np.ndarray:
        """
        Finds line segments in a binary image using the probabilistic Hough transform.
        Based on the OpenCV function
        `HoughLinesP
        <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`_.

        Args:
            edges (:obj:`numpy array`): binary image with edges

            hough_threshold (:obj:`int`): Accumulator threshold parameter. Only those lines are
            returned that get enough votes, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
            =houghlinesp#houghlinesp>`__.

            hough_min_line_length (:obj:`int`): Minimum line length. Line segments shorter than
            that are rejected, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
            =houghlinesp#houghlinesp>`__.

            hough_max_line_gap (:obj:`int`): Maximum allowed gap between points on the same line to
            link them, details `here
            <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight
            =houghlinesp#houghlinesp>`__.

        Returns:
             :obj:`numpy array`: An ``Nx4`` array where each row represents a
             line ``[x1, y1, x2, y2]``. If no lines
             were detected, returns an empty list.

        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=hough_min_line_length,
            maxLineGap=hough_max_line_gap,
        )
        if lines is not None:
            lines = lines.reshape((-1, 4))  # it has an extra dimension
        else:
            lines = np.array([])

        return lines

    @staticmethod
    def color_filter(image: HSVImage, color: ColorRange, edges: np.ndarray,
                     dilation_kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains the regions of the image that fall in the provided color range and the subset
        of the detected Canny edges which are in these regions.
        Applies a `dilation <https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm>`_
        operation to smooth and grow the regions map.

        Args:
            image: (:obj:`numpy array`): HSV image to process.

            color (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying
            the desired colors.

            edges (:obj:`numpy array`): binary image with the edges as returned by ``find_edges``.

            dilation_kernel_size (:obj:`int`): kernel size for the dilation operation which fills
            in the gaps in the color filter result.

        Returns:
            :obj:`tuple`: a tuple containing:

                :obj:`numpy array`: binary image with the regions of the image that fall in the
                color range

                :obj:`numpy array`: binary image with the edges in the image that fall in the
                color range
        """
        # threshold colors in HSV space
        map = color.inRange(image)
        # binary dilation: fills in gaps and makes the detected regions grow
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        map = cv2.dilate(map, kernel)
        # extract only the edges which come from the region with the selected color
        edge_color = cv2.bitwise_and(map, edges)
        # ---
        return map, edge_color

    @staticmethod
    def find_normal(map: np.ndarray, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the centers of the line segments and their normals.

        Args:
            map (:obj:`numpy array`):  binary image with the regions of the image that fall
            in a given color range

            lines (:obj:`numpy array`): An ``Nx4`` array where each row represents a line.
            If no lines were detected,
            returns an empty list.

        Returns:
            :obj:`tuple`: a tuple containing:

                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the center
                 point of a line. If no lines were detected returns an empty list.

                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the normal
                 of a line. If no lines were detected returns an empty list.
        """
        normals = np.array([])
        centers = np.array([])
        if len(lines) > 0:
            length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
            dx = 1.0 * (lines[:, 3:4] - lines[:, 1:2]) / length
            dy = 1.0 * (lines[:, 0:1] - lines[:, 2:3]) / length

            centers = np.hstack(
                [(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
            x3 = (centers[:, 0:1] - 3.0 * dx).astype("int")
            y3 = (centers[:, 1:2] - 3.0 * dy).astype("int")
            x4 = (centers[:, 0:1] + 3.0 * dx).astype("int")
            y4 = (centers[:, 1:2] + 3.0 * dy).astype("int")

            np.clip(x3, 0, map.shape[1] - 1, out=x3)
            np.clip(y3, 0, map.shape[0] - 1, out=y3)
            np.clip(x4, 0, map.shape[1] - 1, out=x4)
            np.clip(y4, 0, map.shape[0] - 1, out=y4)

            flag_signs = (np.logical_and(map[y3, x3] > 0, map[y4, x4] == 0)).astype("int") * 2 - 1
            normals = np.hstack([dx, dy]) * flag_signs

        return centers, normals

    def detect(self, image: BGRImage, colors: List[ColorRange]) -> List[Detections]:
        """
        Detects the line segments in the currently set image that occur in and the edges of
        the regions of the image
        that are within the provided colour ranges.

        Args:
            image (:obj:`numpy array`):         BGR image as numpy array
            colors (:obj:`List[ColorRange]`):   A list of :py:class:`ColorRange` objects specifying
                                                the desired colors to detect.

        Returns:
            :obj:`List[Detections]`:    A list of :py:class:`Detections` objects, one for each
                                        color range given, containing the detected lines for the
                                        corresponding color.

        """
        # find edges
        edges = self.find_edges(image,
                                self.canny_thresholds[0],
                                self.canny_thresholds[1],
                                self.canny_aperture_size)
        # bgr -> hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # detect colors
        output: List[Detections] = []
        for color in colors:
            # color filter
            map, edge_color = self.color_filter(hsv, color, edges, self.dilation_kernel_size)
            # hough lines
            lines = self.hough_line(edge_color, self.hough_threshold,
                                    self.hough_min_line_length,
                                    self.hough_max_line_gap)
            # find center and normals
            centers, normals = self.find_normal(map, lines)
            # pack detections
            dets = Detections(lines=lines, normals=normals, map=map, centers=centers)
            output.append(dets)
        # ---
        return output

#!/usr/bin/env python3

import numpy as np

CameraModel = object

@dataclasses.dataclass
class GroundPoint:
    x: float
    y: float

NormalizedImagePoint = GroundPoint


class GroundProjector:
    """
        Handles the Ground Projection operations.

        This class projects points in the image to the ground plane and in the robot's
        reference frame.
        It enables lane localization in the 2D ground plane.
        This projection uses the homography obtained from the extrinsic calibration procedure.

        Note:
            All pixel and image operations in this class assume that the pixels and images are
            *already rectified*.
            If unrectified pixels or images are supplied, the outputs of these operations will
            be incorrect.

        Args:
            camera (``CameraModel``): Object describing the camera model

        """

    def __init__(self, camera: CameraModel):
        self.camera = camera
        # store homography
        if self.camera.H is None:
            raise ValueError("You need to set a homography `H` in your CameraModel object before "
                             "you can use it to create an instance of GroundProjector.")
        self.H: np.ndarray = self.camera.H.reshape((3, 3))
        # invert homography
        self.Hinv: np.ndarray = np.linalg.inv(self.H)
        # cache/support objects
        self.debug_img_bg = None

    def vector2ground(self, point: NormalizedImagePoint) -> GroundPoint:
        """
        Projects a normalized point (``[-1, 1] X [-1, 1]``) to the ground
        plane using the homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.

        Returns:
            :py:class:`GroundPoint` : A :py:class:`GroundPoint` object on the ground plane.

        """
        point = np.array([point.x, point.y, 1.0])
        ground_point = np.dot(self.H, point)
        ground_point = ground_point / ground_point[2]
        return GroundPoint(*ground_point[:2])

    def ground2vector(self, point: GroundPoint) -> NormalizedImagePoint:
        """
        Projects a point on the ground plane to a normalized pixel (``[-1, 1] X [-1, 1]``) using the
        homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object on the ground plane.

        Returns:
            :py:class:`NormalizedImagePoint` : A point in normalized coordinates.

        """
        ground_point = np.array([point.x, point.y, 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]
        return NormalizedImagePoint(*image_point[:2])

    def distorted_to_ground(self, point: NormalizedImagePoint) -> GroundPoint:
        """
        Creates a :py:class:`ground_projection.types.GroundPoint` object from a normalized point
        message from a distorted image. It converts it to pixel coordinates and rectifies it.
        Then projects it to the ground plane.

        Args:
            point (:obj:`ground_projection.types.NormalizedImagePoint`): Normalized point
            coordinates from a distorted image.

        Returns:
            :obj:`ground_projection.types.GroundPoint`: Point coordinates in the ground
            reference frame.

        """
        # rectify point [distorted point -> rectified point]
        point_rect = self.camera.rectifier.rectify_pixel(self.camera.vector2pixel(point))
        # project on ground [rectified point -> ground point]
        ground_pt = self.vector2ground(self.camera.pixel2vector(point_rect))
        # ---
        return ground_pt

    # def lineseglist_cb(self, seglist_msg):
    #     """
    #     Projects a list of line segments on the ground reference frame point by point by
    #     calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes the projected list of segments.
    #
    #     Args:
    #         seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
    #         unrectified images
    #
    #     """
    #     if self.camera_info_received:
    #         seglist_out = SegmentList()
    #         seglist_out.header = seglist_msg.header
    #         for received_segment in seglist_msg.segments:
    #             new_segment = Segment()
    #             new_segment.points[0] = self.pixel_msg_to_ground_msg(received_segment.pixels_normalized[0])
    #             new_segment.points[1] = self.pixel_msg_to_ground_msg(received_segment.pixels_normalized[1])
    #             new_segment.color = received_segment.color
    #             # TODO: what about normal and points?
    #             seglist_out.segments.append(new_segment)
    #         self.pub_lineseglist.publish(seglist_out)
    #
    #         if not self.first_processing_done:
    #             self.log("First projected segments published.")
    #             self.first_processing_done = True
    #
    #         if self.pub_debug_img.get_num_connections() > 0:
    #             debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.debug_image(seglist_out))
    #             debug_image_msg.header = seglist_out.header
    #             self.pub_debug_img.publish(debug_image_msg)
    #     else:
    #         self.log("Waiting for a CameraInfo message", "warn")
