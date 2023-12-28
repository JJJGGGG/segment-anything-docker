import base64
from fastapi import APIRouter, Request
from base64 import b64decode, b64encode
import json
from segment_anything import SamPredictor
import numpy as np
import io
import cv2
from ..schemas import SegmentBody

router = APIRouter()

@router.post("/segment")
def segment_image(request: Request, body: SegmentBody):
    image = b64decode(body.image.encode())

    file_bytes = np.fromstring(image, np.uint8)    
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        predictor = SamPredictor(request.app.state.ml_models["sam"])
        predictor.set_image(image)

        input_box = np.array(body.box) if body.box else None
        input_points = np.array(body.input_points) if body.input_points else None
        input_labels = np.array(body.input_labels) if body.input_labels else None
        multimask_output=bool(body.multimask_output)

        masks, _, _ = predictor.predict(
            box=input_box,
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output
        )
    except Exception as ex:
        return {
            "detail": str(ex)
        }

    single_mask = np.repeat(masks[0][:, :, np.newaxis], 3, axis=2)

    tobyte = lambda t: 255 if t else 0
    vfunc = np.vectorize(tobyte)

    mask_image_vect = vfunc(single_mask)
    
    is_success, buffer = cv2.imencode(".png", mask_image_vect)
    return {
        "encoded_binary_mask": b64encode(buffer).decode()
    }
