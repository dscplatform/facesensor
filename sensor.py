import sys
import os
import asyncio
import dscframework
import json
import keras
from keras.models import load_model
from data import build_dataset
import numpy as np

version = 1
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
model = load_model(("export/mdl_v%d.h5")%(version))

async def on_facedetect(head, data):
    print("got sub data", flush=True)
    print(json.dumps(head), flush=True)
    print(data, flush=True)
    # Batch predict
    #X, y = build_dataset()
    #batch = np.asarray([X[0]])
    #result = model.predict(batch, batch_size=1, verbose=1)
    #print(result)
    #print(y[0])

async def on_connect(cli):
    await cli.subscribe("facebuffer", on_facedetect)

async def main():
    cli = dscframework.Client("ws://localhost:8080")
    await cli.start(on_connect)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
