# LVTN Server
## How to use
Quickstart as below instructions:
- Upload your model weights (ie `h5` file) to `weights` folder then set the directory in `.env` file. 

- Build your own class `YourModel` in `models.py`

- Create a route with GET method in app.py to send client your model's prediction
## Data Flow
### Upload Image
```sequence
Client->Server: Send image 
Server: Save image to images folder
Server: Resize saved image
Server: Save resized image to images folder
Server->Client: Send resized image
```
### Get Prediction
```sequence
Client->Server: Get prediction 
Server: Get resized image
Server: Generate prediction
Server: Save prediction to results folder
Server->Client: Send prediction
```