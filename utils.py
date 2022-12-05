import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array, array_to_img
import math
from typing import Tuple

from torch.jit import load as load_script
from torchvision.transforms import ToTensor, ToPILImage

load_dotenv()

SALGAN_WEIGHT_DIR = os.getenv("SALGAN_WEIGHT_DIR")
SALGAN_ARCH_DIR = os.getenv("SALGAN_ARCH_DIR")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
RESULT_FOLDER = os.getenv("RESULT_FOLDER")


def load_resized_img() -> Tuple[bool, int, int]:
    original = load_img(
        os.path.join(UPLOAD_FOLDER, "original.png"), interpolation="bilinear"
    )
    resized = load_img(
        os.path.join(UPLOAD_FOLDER, "resized.png"), interpolation="bilinear"
    )

    # return True, Int, Int if resized height
    if original.size[0] == resized.size[0]:
        if (resized.size[1] - original.size[1]) % 2 == 0:
            p = int((resized.size[1] - original.size[1]) / 2)
            return True, p, p
        else:
            p1 = int((resized.size[1] - original.size[1] + 1) / 2)
            p2 = resized.size[1] - original.size[1] - p1
            return True, p1, p2
    else:
        # return False, Int, Int if resized width
        print(resized.size[0], original.size[0])
        if (resized.size[0] - original.size[0]) % 2 == 0:
            p = int((resized.size[0] - original.size[0]) / 2)
            return False, p, p
        else:
            p1 = int((resized.size[0] - original.size[0] + 1) / 2)
            p2 = resized.size[0] - original.size[0] - p1
            return False, p1, p2


def resize(path: str, save=False):
    img = load_img(path, interpolation="bilinear")
    img_array = img_to_array(img)
    width, height = img.size

    if save == True:

        print("Original image size is {} {}".format(width, height))

        if int(4 * height / 3) > width:
            _width = math.ceil(4 * height / 3)
            print("Resizing width with estimated new width {}".format(_width))
            if (_width - width) % 2 == 0:
                padding_value = int((_width - width) / 2)
                pad = tf.constant([[0, 0], [padding_value, padding_value], [0, 0]])
                img_array = tf.pad(img_array, pad, "CONSTANT")
            else:
                t_pad = int((_width - width + 1) / 2)
                b_pad = int(_width - width - t_pad)
                pad = tf.constant([[0, 0], [t_pad, b_pad], [0, 0]])
                img_array = tf.pad(img_array, pad, "CONSTANT")
        else:
            _height = math.ceil(3 * width / 4)
            print("Resizing height with estimated new value {}".format(_height))
            if (_height - height) % 2 == 0:
                padding_value = int((_height - height) / 2)
                pad = tf.constant([[padding_value, padding_value], [0, 0], [0, 0]])
                img_array = tf.pad(img_array, pad, "CONSTANT")
            else:
                t_pad = int((_height - height + 1) / 2)
                b_pad = int(_height - height - t_pad)
                pad = tf.constant([[t_pad, b_pad], [0, 0], [0, 0]])
                img_array = tf.pad(img_array, pad, "CONSTANT")

    resized_img = array_to_img(img_array)

    if save == True:
        print(
            "Resized image size is {} {}".format(
                resized_img.size[0], resized_img.size[1]
            )
        )
        RESIZED_IMG_DIR = os.path.join(UPLOAD_FOLDER, "resized.png")
        resized_img.save(RESIZED_IMG_DIR, format="PNG")
        return
    else:
        return resized_img.size[0], resized_img.size[1]
