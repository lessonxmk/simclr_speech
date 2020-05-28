import numpy as np
import random
import torch
import tensorflow as tf
import functools
from absl import flags
FLAGS = flags.FLAGS

def audio_Crop(audioData, proportion=0.5):
    '''
    Randomly select 2 points and crop the audio between the points
    audio_data: [Batch size, Time * Rate]
    proportion: (0, 1). control the max crop proportion of the audio
    '''
    batchSize = audioData.shape[0]
    audioLength = audioData.shape[1]
    cropMatrix = []
    for audio in batchSize:
        begin = np.random.randint(1, audioLength * (1 - proportion))
        end = np.random.randint(begin, begin + audioLength * proportion)
        left = np.ones(begin)
        mid = np.zeros(end - begin)
        right = np.ones(audioLength - end)
        mask = np.concatenate((left, mid, right), 0)
        cropMatrix.append(mask)
    cropMatrix = np.array(cropMatrix)
    cropMatrix = torch.from_numpy(cropMatrix)
    audioData = audioData.mul(cropMatrix)
    return audioData


def Gaussian_white_noise(audio_data, intensity=1):
    noise = torch.randn(audio_data.shape) * intensity
    audio_data += noise
    return audio_data


def random_crop_with_resize(image, height, width, p=1.0):
    """Randomly crop and resize an image.

      Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        p: Probability of applying this transformation.

      Returns:
        A preprocessed image `Tensor`.
      """

    def _transform(image):  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image

    return random_apply(_transform, p=p, x=image)


def crop_and_resize(image, height, width):
    """Make a random crop and resize it to height `height` and width `width`.

    Args:
      image: Tensor representing the image.
      height: Desired image height.
      width: Desired image width.

    Returns:
      A `height` x `width` x channels Tensor holding a random crop of `image`.
    """
    bbox = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100,
        scope=None)

    return tf.image.resize_bicubic([image], [height, width])[0]


def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    if (np.random.rand() > p):
        return x
    else:
        return func(x)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: `Tensor` of image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      (cropped image `Tensor`, distorted bbox `Tensor`).
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, target_height, target_width)

        return image

def random_color_jitter(image, p=1.0):
  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter, strength=FLAGS.color_jitter_strength)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)
  return random_apply(_transform, p=p, x=image)

def color_jitter(image,
                 strength,
                 random_order=True):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)

def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = tf.image.random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return tf.image.random_brightness(x, max_delta=brightness)
      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random_shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image

def to_grayscale(image, keep_channels=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3])
  return image

def flip(image):
    return tf.image.random_flip_left_right(image)