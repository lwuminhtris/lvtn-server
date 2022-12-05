# LVTN Server
## How To Use
Quickstart as below instructions:
- Upload your model weights (ie `h5` file) to `weights` folder then set the directory in `.env` file. 

- Build your own class `YourModel` in `models.py`

- Create a route with GET method in app.py to send client your model's prediction
## Data Flow
### Upload Image
`[POST]: /save`
```sequence
Client->Server: Send image 
Server: Save image to images folder
Server: Resize saved image
Server: Save resized image to images folder
Server->Client: OK
```
### Get Prediction
`[GET]: /salgan`
```sequence
Client->Server: Get prediction 
Server: Get resized image
Server: Generate prediction
Server: Save prediction to results folder
Server->Client: Send prediction
```

`[POST]: /transalnet`
```sequence
Client->Server: Send image
Server: Generate prediction
Server->Client: Send prediction
```

`[POST]: /msinet`
```sequence
Client->Server: Send image
Server: Generate prediction
Server->Client: Send prediction
```
