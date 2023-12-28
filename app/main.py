from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from segment_anything import sam_model_registry
from app.routers import inference

async def lifespan(app: FastAPI):

    app.state.ml_models = {}

    if torch.cuda.is_available():
       device = "cuda"
    else:
       device = "cpu"

    # Load the SAM model
    sam = sam_model_registry["vit_l"](checkpoint="./sam_images/sam_vit_l_0b3195.pth")
    
    sam.to(device=device)

    app.state.ml_models["sam"] = sam
    yield

    # Clean up the ML models and release the resources
    app.state.ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.include_router(inference.router)

@app.get("/healthcheck")
def check_working():
    return {"online": True}
