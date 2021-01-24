from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import os


from app.sentiment_model import SentimentModel
from app.image_model import ImageModel
from keras.models import load_model

middleware = [
    Middleware(CORSMiddleware, allow_origins=['*'])
]

print(os.listdir("./app"))
modela = load_model("./app/Sentiment.h5")
modelb = load_model("./app/lungs(1).h5")
im = ImageModel(modelb)
sm = SentimentModel(modela)



app = Starlette(middleware=middleware)


@app.route("/")
async def home(_request):
    return JSONResponse({"hello": "world"})


@app.route("/sentence", methods=['GET'])
async def proc_sen(request: Request):
    sentence: str = request.query_params['sentence']
    if not sentence:
        return JSONResponse({"error": "no sentence"},status_code=400)
    else:
        return JSONResponse(sm.predict(sentence))


@app.route("/image", methods=['POST'])
async def img_pred(request: Request):
    form = await request.form()
    byt = await form["File"].read()
    open("./topred.png","wb").write(byt)
    return JSONResponse(im.model_predict("./topred.png"))


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)
