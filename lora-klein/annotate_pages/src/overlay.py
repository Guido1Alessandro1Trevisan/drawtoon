"""Render red numbered bboxes onto a manga page (right-to-left, top-to-bottom)."""

from __future__ import annotations

import io
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


BORDER_COLOR = (220, 30, 30, 255)
LABEL_BG_COLOR = (220, 30, 30, 230)
LABEL_TEXT_COLOR = (255, 255, 255, 255)
BORDER_WIDTH_FRAC = 0.005   # of the page diagonal
LABEL_FONT_FRAC = 0.024      # of the page diagonal


def manga_reading_order(bboxes: list[list[int]]) -> list[int]:
    """Sort indices into right-to-left, top-to-bottom reading order.

    Group bboxes into rows by vertical overlap, then sort each row by x_max
    descending (rightmost first). Returns the new ordering of original indices.
    """
    n = len(bboxes)
    if n == 0:
        return []
    rows: list[list[tuple[int, list[int]]]] = []
    # process top to bottom by y_min
    by_top = sorted(range(n), key=lambda i: bboxes[i][1])
    for idx in by_top:
        b = bboxes[idx]
        placed = False
        for row in rows:
            # if vertical overlap with any member of the row is significant, join
            row_y_min = min(item[1][1] for item in row)
            row_y_max = max(item[1][3] for item in row)
            overlap = max(0, min(b[3], row_y_max) - max(b[1], row_y_min))
            row_height = row_y_max - row_y_min
            if row_height > 0 and overlap / row_height >= 0.35:
                row.append((idx, b))
                placed = True
                break
        if not placed:
            rows.append([(idx, b)])
    # rows: sort each row right-to-left (descending x_max), rows sorted top-to-bottom
    ordered: list[int] = []
    for row in rows:
        row.sort(key=lambda item: -item[1][2])  # rightmost first
        ordered.extend(idx for idx, _ in row)
    return ordered


def _font_for_size(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # PIL ships a default bitmap font. Lambda image won't have system fonts.
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", px)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_numbered_overlay(
    image_bytes: bytes,
    *,
    bboxes: list[list[int]],
    ordering: list[int],
    output_format: str = "JPEG",
    jpeg_quality: int = 85,
) -> bytes:
    """Draw red numbered rectangles on the page.

    `bboxes` is the raw panel list. `ordering` is the reading-order indices
    (`manga_reading_order(bboxes)`). The drawn label is the ordering position
    (0..N-1), not the original index — so panel 0 is always the first-read.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        page = img.convert("RGBA")

    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = page.size
    diag = (w * w + h * h) ** 0.5
    border_w = max(2, int(round(diag * BORDER_WIDTH_FRAC)))
    font_px = max(14, int(round(diag * LABEL_FONT_FRAC)))
    font = _font_for_size(font_px)

    for read_pos, orig_idx in enumerate(ordering):
        bbox = bboxes[orig_idx]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        # Clamp to page
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        draw.rectangle(((x1, y1), (x2, y2)), outline=BORDER_COLOR, width=border_w)

        # Label badge in the top-right corner of each panel (matches manga reading start)
        label = str(read_pos)
        bbox_text = font.getbbox(label) if hasattr(font, "getbbox") else (0, 0, font_px, font_px)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = max(2, font_px // 3)
        badge_w = tw + 2 * pad
        badge_h = th + 2 * pad
        bx1 = x2 - badge_w
        by1 = y1
        bx2 = x2
        by2 = y1 + badge_h
        draw.rectangle(((bx1, by1), (bx2, by2)), fill=LABEL_BG_COLOR)
        draw.text(
            (bx1 + pad - bbox_text[0], by1 + pad - bbox_text[1]),
            label,
            fill=LABEL_TEXT_COLOR,
            font=font,
        )

    out = Image.alpha_composite(page, overlay).convert("RGB")
    buf = io.BytesIO()
    if output_format.upper() == "JPEG":
        out.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        out.save(buf, format=output_format.upper())
    return buf.getvalue()
