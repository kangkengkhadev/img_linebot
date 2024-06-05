from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
from linebot.exceptions import InvalidSignatureError

app = FastAPI()

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]
input_name = input_details[0]['name']

classify_lite = interpreter.get_signature_runner('serving_default')
class_names = ['Gingivitis', 'Healthy']

# LINE Messaging API setup
channel_secret = '933e52a41edd1ba1ba245383ebf2dd40'
channel_access_token = 'PuoiQtsxmhQAdd9SUGKhHaSxvRbgaVzjdV52poklAZSSYm261mdUJBebnjMWQhDLT0wSj6/XETFkEWL5aT5JeAbwzJkGLnk7DJzeUTmaF15qTTCRE19VAdnWGKlQ7E1WJqZgEwKL9VztbUAsO9R8UgdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.keras.utils.load_img(io.BytesIO(contents), target_size=input_shape)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions_lite = classify_lite(**{input_name: img_array})['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)
    return JSONResponse(content={"predicted_class": class_names[np.argmax(score_lite)]})

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        return JSONResponse(status_code=400, content={"message": "Invalid signature"})
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="Send me an image to analyze.")
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event: MessageEvent):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = io.BytesIO(message_content.content)
    image = Image.open(image_bytes)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"Predicted class: 3")
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
