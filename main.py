import os
from io import BytesIO
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response
from PIL import Image, ImageOps
import uvicorn

app = FastAPI(title="rembg-api")

# حفظ الجلسات في الذاكرة لعدم إعادة تحميل الموديل عند كل طلب
sessions = {}
MODEL_CHAIN = ["isnet-general-use", "u2net"]


def get_session(model_name):
    global sessions
    if model_name not in sessions:
        from rembg import new_session
        sessions[model_name] = new_session(model_name)
    return sessions[model_name]


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


# [إضافة أساسية]: مسار فحص الحالة لإبقاء السيرفر نشطاً عبر UptimeRobot
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
        res = requests.get(url, headers=headers, timeout=20)
        res.raise_for_status()

        input_image = Image.open(BytesIO(res.content))
        input_image = ImageOps.exif_transpose(input_image)
        input_image = input_image.convert("RGB")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download/open source image: {str(e)}")

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

            last_error = f"Model '{model_name}' produced blank/noisy result (visible={visible_fraction:.4f}, density={bbox_density:.4f})"

        except Exception as e:
            last_error = f"Model '{model_name}' raised an error: {str(e)}"
            continue

    raise HTTPException(
        status_code=422,
        detail=f"Background removal failed for all models. {last_error}. Diagnostics: {' | '.join(diagnostics)}"
    )


if __name__ == "__main__":
    # تعديل البورت الافتراضي إلى 10000 ليتطابق مع Render و Dockerfile
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
