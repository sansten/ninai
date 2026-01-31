"""Generate a scalable deployment diagram PNG.

Creates docs/scalable_deployment.png.

Run:
  python docs/generate_scalable_deployment_diagram.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "scalable_deployment.png"


def _font(size: int) -> ImageFont.ImageFont:
    # Default bitmap font is fine and portable.
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, r: int, fill, outline, width: int = 2):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1, y1, x2, y2), radius=r, fill=fill, outline=outline, width=width)


def _center_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(0, 0, 0)):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx = x1 + (w - tw) / 2
    ty = y1 + (h - th) / 2
    draw.text((tx, ty), text, font=font, fill=fill)


def _arrow(draw: ImageDraw.ImageDraw, p1, p2, color=(40, 40, 40), width: int = 3, head: int = 10):
    x1, y1 = p1
    x2, y2 = p2
    draw.line((x1, y1, x2, y2), fill=color, width=width)

    # Simple arrow head
    if abs(x2 - x1) >= abs(y2 - y1):
        # horizontal
        if x2 >= x1:
            tip = (x2, y2)
            left = (x2 - head, y2 - head // 2)
            right = (x2 - head, y2 + head // 2)
        else:
            tip = (x2, y2)
            left = (x2 + head, y2 - head // 2)
            right = (x2 + head, y2 + head // 2)
    else:
        # vertical
        if y2 >= y1:
            tip = (x2, y2)
            left = (x2 - head // 2, y2 - head)
            right = (x2 + head // 2, y2 - head)
        else:
            tip = (x2, y2)
            left = (x2 - head // 2, y2 + head)
            right = (x2 + head // 2, y2 + head)

    draw.polygon([tip, left, right], fill=color)


def main() -> None:
    W, H = 1600, 900
    bg = (248, 250, 252)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    title_f = _font(28)
    h_f = _font(18)
    t_f = _font(14)

    # Header
    draw.text((40, 25), "Ninai2 — Scalable Deployment (API + Workers + Stores)", font=title_f, fill=(15, 23, 42))
    draw.text((40, 60), "Scale API and Celery workers independently; isolate state in Postgres/Redis/Qdrant; throttle+cache LLM calls.", font=t_f, fill=(51, 65, 85))

    # Palette
    c_compute = (219, 234, 254)  # blue-100
    c_compute_border = (59, 130, 246)
    c_queue = (254, 249, 195)  # amber-100
    c_queue_border = (245, 158, 11)
    c_store = (220, 252, 231)  # green-100
    c_store_border = (34, 197, 94)
    c_ext = (243, 232, 255)  # purple-100
    c_ext_border = (168, 85, 247)

    # Boxes
    users = (60, 140, 280, 220)
    lb = (320, 140, 540, 220)

    api_group = (600, 100, 980, 260)
    api1 = (620, 140, 790, 220)
    api2 = (810, 140, 960, 220)

    worker_group = (600, 300, 980, 520)
    w1 = (620, 350, 790, 430)
    w2 = (810, 350, 960, 430)

    beat = (600, 550, 790, 630)

    redis = (1040, 310, 1510, 380)
    postgres = (1040, 410, 1510, 480)
    qdrant = (1040, 510, 1510, 580)
    es = (1040, 610, 1510, 680)

    llm = (1040, 140, 1280, 220)
    ocr = (1310, 140, 1510, 220)

    # Draw compute
    _rounded_rect(draw, users, 16, c_compute, c_compute_border)
    _center_text(draw, users, "Clients\n(Web/CLI)", h_f, fill=(30, 41, 59))

    _rounded_rect(draw, lb, 16, c_compute, c_compute_border)
    _center_text(draw, lb, "Ingress /\nLoad Balancer", h_f, fill=(30, 41, 59))

    # API group
    _rounded_rect(draw, api_group, 18, (239, 246, 255), (147, 197, 253), width=2)
    draw.text((api_group[0] + 14, api_group[1] + 10), "Backend API (FastAPI) — scale horizontally", font=t_f, fill=(30, 64, 175))

    for b, label in [(api1, "API Pod 1"), (api2, "API Pod N")]:
        _rounded_rect(draw, b, 14, c_compute, c_compute_border)
        _center_text(draw, b, f"{label}\n/health, /api", t_f, fill=(30, 41, 59))

    # Worker group
    _rounded_rect(draw, worker_group, 18, (239, 246, 255), (147, 197, 253), width=2)
    draw.text((worker_group[0] + 14, worker_group[1] + 10), "Celery Workers — scale horizontally", font=t_f, fill=(30, 64, 175))

    for b, label in [(w1, "Worker 1"), (w2, "Worker N")]:
        _rounded_rect(draw, b, 14, c_compute, c_compute_border)
        _center_text(draw, b, f"{label}\nAgents + Pipeline", t_f, fill=(30, 41, 59))

    _rounded_rect(draw, beat, 14, c_compute, c_compute_border)
    _center_text(draw, beat, "Celery Beat\n(Schedules)", t_f, fill=(30, 41, 59))

    # Queue
    _rounded_rect(draw, redis, 14, c_queue, c_queue_border)
    _center_text(draw, redis, "Redis\nBroker + Caching", h_f, fill=(120, 53, 15))

    # Stores
    for b, label in [
        (postgres, "Postgres\n(RLS tenant data + cache table)"),
        (qdrant, "Qdrant\n(Vector store)"),
        (es, "Elasticsearch\n(Audit search)"),
    ]:
        _rounded_rect(draw, b, 14, c_store, c_store_border)
        _center_text(draw, b, label, t_f, fill=(20, 83, 45))

    # External services
    _rounded_rect(draw, llm, 14, c_ext, c_ext_border)
    _center_text(draw, llm, "Ollama\n(LLM)", h_f, fill=(88, 28, 135))

    _rounded_rect(draw, ocr, 14, c_ext, c_ext_border)
    _center_text(draw, ocr, "OCR Sidecar\n(Tesseract)", t_f, fill=(88, 28, 135))

    # Arrows
    _arrow(draw, (users[2], (users[1] + users[3]) // 2), (lb[0], (lb[1] + lb[3]) // 2))
    _arrow(draw, (lb[2], (lb[1] + lb[3]) // 2), (api_group[0], (api1[1] + api1[3]) // 2))

    # API -> stores
    _arrow(draw, (api_group[2], 180), (llm[0], 180), color=(107, 33, 168))
    _arrow(draw, (api_group[2], 200), (ocr[0], 200), color=(107, 33, 168))

    _arrow(draw, (api_group[2], 230), (postgres[0], 440), color=(20, 83, 45))
    _arrow(draw, (api_group[2], 240), (qdrant[0], 540), color=(20, 83, 45))
    _arrow(draw, (api_group[2], 250), (es[0], 650), color=(20, 83, 45))

    # API -> redis (enqueue)
    _arrow(draw, (api_group[2], 215), (redis[0], 345), color=(120, 53, 15))

    # beat -> redis
    _arrow(draw, (beat[2], (beat[1] + beat[3]) // 2), (redis[0], 330), color=(120, 53, 15))

    # redis -> workers
    _arrow(draw, (redis[0], 360), (worker_group[2], 400), color=(120, 53, 15))

    # workers -> stores + LLM
    _arrow(draw, (worker_group[2], 390), (postgres[0], 430), color=(20, 83, 45))
    _arrow(draw, (worker_group[2], 420), (qdrant[0], 540), color=(20, 83, 45))
    _arrow(draw, (worker_group[2], 450), (es[0], 640), color=(20, 83, 45))
    _arrow(draw, (worker_group[2], 370), (llm[0], 200), color=(107, 33, 168))
    _arrow(draw, (worker_group[2], 385), (ocr[0], 215), color=(107, 33, 168))

    # Legend
    legend_x, legend_y = 60, 760
    draw.text((legend_x, legend_y), "Legend", font=h_f, fill=(15, 23, 42))

    def legend_item(y, fill, outline, label):
        _rounded_rect(draw, (legend_x, y, legend_x + 26, y + 20), 6, fill, outline, width=2)
        draw.text((legend_x + 36, y + 2), label, font=t_f, fill=(51, 65, 85))

    legend_item(legend_y + 34, c_compute, c_compute_border, "Compute (scale out via replicas)")
    legend_item(legend_y + 60, c_queue, c_queue_border, "Queue/Broker (Redis)")
    legend_item(legend_y + 86, c_store, c_store_border, "Stateful stores (Postgres/Qdrant/Elasticsearch)")
    legend_item(legend_y + 112, c_ext, c_ext_border, "External/sidecar services (Ollama/OCR)")

    # Scaling notes
    notes = (
        "Scalability highlights:\n"
        "• Scale API pods for request throughput\n"
        "• Scale Celery workers for enrichment throughput\n"
        "• LLM throttling: OLLAMA_MAX_CONCURRENCY\n"
        "• Result caching: agent_result_cache (Postgres + RLS)"
    )
    draw.text((620, 720), notes, font=t_f, fill=(51, 65, 85))

    img.save(OUT_PATH)


if __name__ == "__main__":
    main()
