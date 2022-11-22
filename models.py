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


class SalGAN:
    def __init__(self, img_path: str) -> str:
        self.model = load_model(SALGAN_WEIGHT_DIR, compile=False)
        self.path = img_path
        self.width, self.height = resize(self.path, save=False)
        self.c, self.p1, self.p2 = load_resized_img()
        print("SalGAN resized image value is {} {} {}".format(self.c, self.p1, self.p2))
        self.RESIZED_IMG_DIR = os.path.join(UPLOAD_FOLDER, "resized.png")

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
        # -> Crop saliency map to original size
        if self.c == False:
            cropped_array = tf.image.crop_to_bounding_box(
                generated_tensor[0], 0, self.p1, self.height, self.width - self.p1 * 2
            )
            generated_img = array_to_img(cropped_array)
            GENERATED_IMG_DIR = os.path.join(RESULT_FOLDER, "result.png")
            generated_img.save(GENERATED_IMG_DIR, format="PNG")
            return GENERATED_IMG_DIR
        else:
            cropped_array = tf.image.crop_to_bounding_box(
                generated_tensor[0], self.p1, 0, self.height - self.p1 * 2, self.width
            )
            generated_img = array_to_img(cropped_array)
            GENERATED_IMG_DIR = os.path.join(RESULT_FOLDER, "result.png")
            generated_img.save(GENERATED_IMG_DIR, format="PNG")
            return GENERATED_IMG_DIR


class TranSalNetModel:
    def __init__(self, script_path: str):
        self.to_tensor = ToTensor()
        self.to_pil_image = ToPILImage()
        self.script = load_script(script_path)

    def predict(self, image, cuda: bool = False):
        image = image.convert("RGB")
        x = self.to_tensor(image)

        if cuda:
            self.script.cuda()
            x = x.cuda()

        pred = self.script(x)
        pred = pred.detach()
        if cuda:
            pred = pred.cpu()

        pred = self.to_pil_image(pred)
        return pred
