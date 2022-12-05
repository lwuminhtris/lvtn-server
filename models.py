import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array, array_to_img
import math
from typing import Tuple
from utils import resize, load_resized_img

from torch.jit import load as load_script
from torchvision.transforms import ToTensor, ToPILImage

load_dotenv()

SALGAN_WEIGHT_DIR = os.getenv("SALGAN_WEIGHT_DIR")
SALGAN_ARCH_DIR = os.getenv("SALGAN_ARCH_DIR")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
RESULT_FOLDER = os.getenv("RESULT_FOLDER")


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


class MSINetModel:
    def __init__(self, script_path: str):
        print("haha")
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
