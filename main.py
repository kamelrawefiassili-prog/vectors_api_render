import os
import gc
from io import BytesIO
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response
from PIL import Image, ImageOps
import uvicorn
import onnxruntime as ort

# تقييد استهلاك ONNX لموارد المعالج للعمل بسلاسة على Render
ort_options = ort.SessionOptions()
ort_options.intra_op_num_threads = 1
ort_options.inter_op_num_threads = 1
ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

app = FastAPI(title="rembg-api")

current_session = None
current_model_name = None

def get_session(model_name):
    global current_session, current_model_name
    from rembg import new_session
    
    # إعادة استخدام الجلسة إذا كانت نفس الموديل المطلوب
    if current_session is not None and current_model_name == model_name:
        return current_session
    
    # تفريغ الذاكرة من الموديل السكني القديم لمنع تجاوز 512MB RAM
    current_session = None
    gc.collect()
    
    current_session = new_session(model_name, providers=['CPUExecutionProvider'], session_options=ort_options)
    current_model_name = model_name
    return current_session

@app.on_event("startup")
def startup_event():
    # تحميل الموديل الأساسي في الذاكرة فور تشغيل السيرفر
    try:
        print("Pre-loading default model into memory...")
        get_session("isnet-general-use")
        print("Model pre-loaded successfully.")
    except Exception as e:
        print(f"Pre-loading failed: {e}")

def analyze_alpha_content(png_bytes):
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    alpha = np.array(img)[:, :, 3]

    total_pixels = alpha.size
    if total_pixels == 0:
        return 0.0, 0.0

    visible_mask = alpha > 10
    visible_count = int(np.sum(visible_mask))

    if visible_count == 0:
        return 0.0, 0.0

    visible_fraction = visible_count / total_pixels

    ys, xs = np.where(visible_mask)
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    box_area = box_width * box_height
    bbox_density = visible_count / box_area if box_area > 0 else 0.0

    return visible_fraction, bbox_density

def is_result_acceptable(visible_fraction, bbox_density):
    MIN_VISIBLE_FRACTION = 0.005
    MIN_BBOX_DENSITY = 0.15
    return visible_fraction >= MIN_VISIBLE_FRACTION and bbox_density >= MIN_BBOX_DENSITY

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/remove_bg")
def remove_bg(url: str):
    from rembg import remove

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()

        input_image = Image.open(BytesIO(res.content))
        input_image = ImageOps.exif_transpose(input_image)
        input_image = input_image.convert("RGB")

        # تصغير الصورة لحماية السيرفر من الامتلاء ولتسريع المعالجة
        MAX_DIMENSION = 800
        if max(input_image.size) > MAX_DIMENSION:
            input_image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download/open source image: {str(e)}")

    MODEL_CHAIN = ["isnet-general-use", "u2net"]
    last_error = None
    diagnostics = []

    for model_name in MODEL_CHAIN:
        try:
            session = get_session(model_name)
            output_image = remove(input_image, session=session)

            buffer = BytesIO()
            output_image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            visible_fraction, bbox_density = analyze_alpha_content(png_bytes)
            diagnostics.append(f"{model_name}: visible={visible_fraction:.4f} density={bbox_density:.4f}")

            if is_result_acceptable(visible_fraction, bbox_density):
                return Response(content=png_bytes, media_type="image/png")

            last_error = f"Model '{model_name}' produced blank/noisy result"

        except Exception as e:
            last_error = f"Model '{model_name}' raised an error: {str(e)}"
            continue

    raise HTTPException(
        status_code=422,
        detail=f"Background removal failed. {last_error}. Diagnostics: {' | '.join(diagnostics)}"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
