import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array, array_to_img
import math

load_dotenv()

SALGAN_WEIGHT_DIR = os.getenv("SALGAN_WEIGHT_DIR")
SALGAN_ARCH_DIR = os.getenv("SALGAN_ARCH_DIR")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
RESULT_FOLDER = os.getenv("RESULT_FOLDER")


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
        RESIZED_IMG_DIR = os.path.join(UPLOAD_FOLDER, "resized_img.png")
        resized_img.save(RESIZED_IMG_DIR, format="PNG")
        return
    else:
        return resized_img.size[0], resized_img.size[1]


class SalGAN:
    def __init__(self, img_path: str) -> str:
        self.model = load_model(SALGAN_WEIGHT_DIR, compile=False)
        self.path = img_path
        self.width, self.height = resize(self.path, save=False)
        self.RESIZED_IMG_DIR = os.path.join(UPLOAD_FOLDER, "resized_img.png")

    def gen(self) -> str:
        img = tf.convert_to_tensor(
            np.array(
                [
                    img_to_array(
                        load_img(
                            self.RESIZED_IMG_DIR,
                            interpolation="bilinear",
                            target_size=(192, 256),
                        )
                    )
                ]
            )
        )
        generated_tensor = tf.image.resize(
            self.model.predict(img), (self.height, self.width)
        )
        generated_img = array_to_img(generated_tensor[0])
        GENERATED_IMG_DIR = os.path.join(RESULT_FOLDER, "result.png")
        generated_img.save(GENERATED_IMG_DIR, format="PNG")
        return GENERATED_IMG_DIR
