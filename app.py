import os
from dotenv import load_dotenv
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from models import SalGAN, MSI_n_TransalNetModel
from utils import resize
from PIL import Image
from io import BytesIO

load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content_Type"

# Don't change this one
@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return "<h1>SERVER IS RUNNING</h1>"


@app.route("/save", methods=["POST"])
@cross_origin()
def save_handler():
    if "image" not in request.files:
        return {"msg": "Your request is missing image field", "code": 400}
    else:
        f = request.files["image"]
        # -> Save Image
        path = os.path.join(UPLOAD_FOLDER, secure_filename("original.png"))
        f.save(path)

        # -> Save Resized Image
        resize(path, save=True)
        return {"msg": "Image saved and resized successfully", "code": 200}


@app.route("/salgan", methods=["GET"])
@cross_origin()
def salgan_handler():
    path = os.path.join(UPLOAD_FOLDER, "resized.png")
    GENERATED_IMG_DIR = SalGAN(path).gen()
    return send_file(GENERATED_IMG_DIR, mimetype="image/png")


transalnet = MSI_n_TransalNetModel(os.getenv("TRANSALNET_WEIGHT_DIR"))
msinet = MSI_n_TransalNetModel(os.getenv("MSINET_WEIGHT_DIR"))

@app.route("/transalnet", methods=["POST"])
@cross_origin()
def transalnet_handler():
    if request.method == "POST":
        if "image" not in request.files:
            return {"msg": "Your request is missing image field", "code": 400}
        else:
            image = request.files["image"]
            image = Image.open(image)
            image = transalnet.predict(image, cuda=False)
            image_io = BytesIO()
            image.save(image_io, "PNG")
            image_io.seek(0)
            return send_file(image_io, mimetype="image/png")
    else:
        return {"msg": "Unsupported method", "code": 400}


@app.route('/msinet', methods=["POST"])
@cross_origin
def msinet_handler():
    if request.method == "POST":
        if "image" not in request.files:
            return {"msg": "Your request is missing image field", "code": 400}
        else:
            image = request.files["image"]
            image = Image.open(image)
            image = transalnet.predict(image, cuda=False)
            image_io = BytesIO()
            image.save(image_io, "PNG")
            image_io.seek(0)
            return send_file(image_io, mimetype="image/png")
    else:
        return {"msg": "Unsupported method", "code": 400}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
