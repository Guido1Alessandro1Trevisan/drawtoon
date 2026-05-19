from __future__ import annotations

import io
import os
from typing import Any, Callable


MANHWA_RAW_GUTTER_MIN_HEIGHT_PX = int(os.environ.get("MANHWA_RAW_GUTTER_MIN_HEIGHT_PX", "18"))
MANHWA_RAW_GUTTER_COLOR_TOLERANCE = int(os.environ.get("MANHWA_RAW_GUTTER_COLOR_TOLERANCE", "10"))
MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX = int(os.environ.get("MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX", "96"))
MANHWA_RAW_SHEET_PADDING_PX = int(os.environ.get("MANHWA_RAW_SHEET_PADDING_PX", "3"))
MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX = int(os.environ.get("MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX", "1280"))
MANHWA_STRIP_JPEG_QUALITY = int(os.environ.get("MANHWA_STRIP_JPEG_QUALITY", "92"))


def raw_sheet_key(output_prefix: str, chapter_key: str, sheet_index: int, segment_ids: list[str], image_format: str) -> str:
    if segment_ids:
        span = f"segments_{segment_ids[0].split('_')[-1]}-{segment_ids[-1].split('_')[-1]}"
    else:
        span = "segments_empty"
    ext = "png" if image_format == "png" else "jpg"
    return f"{output_prefix.rstrip('/')}/{chapter_key.strip('/')}/sheet_{int(sheet_index):04d}_{span}.{ext}"


def _row_is_uniform_gutter(image: Any, *, y: int, sample_xs: list[int], tolerance: int) -> bool:
    pixels = [image.getpixel((x, y))[:3] for x in sample_xs]
    if not pixels:
        return False
    mins = [min(pixel[channel] for pixel in pixels) for channel in range(3)]
    maxs = [max(pixel[channel] for pixel in pixels) for channel in range(3)]
    horizontal_range = max(maxs[channel] - mins[channel] for channel in range(3))
    if horizontal_range > tolerance:
        return False
    means = [sum(pixel[channel] for pixel in pixels) / float(len(pixels)) for channel in range(3)]
    luma = 0.2126 * means[0] + 0.7152 * means[1] + 0.0722 * means[2]
    saturation_proxy = max(means) - min(means)
    return luma >= 238.0 or luma <= 35.0 or saturation_proxy <= 18.0


def detect_gutter_cuts(image: Any, *, config: dict[str, Any]) -> list[int]:
    manhwa_config = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    min_band_height = max(4, int(manhwa_config.get("raw_gutter_min_height_px") or MANHWA_RAW_GUTTER_MIN_HEIGHT_PX))
    tolerance = max(0, int(manhwa_config.get("raw_gutter_color_tolerance") or MANHWA_RAW_GUTTER_COLOR_TOLERANCE))
    min_segment_height = max(16, int(manhwa_config.get("raw_min_segment_height_px") or MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX))
    width, height = image.size
    if width <= 0 or height <= min_segment_height * 2:
        return []
    step = max(1, width // 96)
    sample_xs = sorted(set([0, width - 1, width // 2] + list(range(0, width, step))))
    flat_rows = [
        _row_is_uniform_gutter(image, y=y, sample_xs=sample_xs, tolerance=tolerance)
        for y in range(height)
    ]
    cuts: list[int] = []
    y = 0
    while y < height:
        if not flat_rows[y]:
            y += 1
            continue
        start = y
        while y < height and flat_rows[y]:
            y += 1
        end = y
        if end - start < min_band_height:
            continue
        midpoint = (start + end) // 2
        if midpoint < min_segment_height or height - midpoint < min_segment_height:
            continue
        cuts.append(midpoint)
    deduped: list[int] = []
    for cut in cuts:
        if not deduped or cut - deduped[-1] >= min_segment_height:
            deduped.append(cut)
    return deduped


def _segment_slices_for_range(
    pages: list[dict[str, Any]],
    *,
    start_y: int,
    end_y: int,
    target_width: int,
) -> tuple[list[dict[str, Any]], int]:
    slices: list[dict[str, Any]] = []
    rendered_height = 0
    for page in pages:
        page_start = int(page.get("global_y_start") or 0)
        page_end = int(page.get("global_y_end") or 0)
        overlap_start = max(int(start_y), page_start)
        overlap_end = min(int(end_y), page_end)
        if overlap_end <= overlap_start:
            continue
        source_y_start = overlap_start - page_start
        source_y_end = overlap_end - page_start
        source_width = max(1, int(page.get("width") or target_width))
        source_height = max(1, int(source_y_end - source_y_start))
        rendered_slice_height = max(1, int(round(source_height * float(target_width) / float(source_width))))
        slices.append(
            {
                "page_index": int(page.get("page_index") or 0),
                "accepted_page_index": int(page.get("accepted_page_index") or 0),
                "relative_path": str(page.get("relative_path") or ""),
                "source_key": str(page.get("source_key") or ""),
                "source_x_start": 0,
                "source_x_end": source_width,
                "source_y_start": int(source_y_start),
                "source_y_end": int(source_y_end),
                "source_width": source_width,
                "source_height": source_height,
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_slice_height),
            }
        )
        rendered_height += rendered_slice_height
    return slices, rendered_height


def build_segments(
    *,
    pages: list[dict[str, Any]],
    config: dict[str, Any],
    get_bytes: Callable[[str], bytes],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    from PIL import Image

    if not pages:
        return [], 0, {"total_virtual_height": 0, "target_width": 0, "detected_gutter_cuts": 0}
    target_width = max(1, max(int(page.get("width") or 0) for page in pages))
    if target_width <= 1:
        target_width = 720
    cuts: set[int] = {0}
    global_y = 0
    detected_cuts = 0
    for accepted_index, page in enumerate(pages):
        source_key = str(page.get("source_key") or "")
        data = get_bytes(source_key)
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            page["accepted_page_index"] = accepted_index
            page["width"] = int(width)
            page["height"] = int(height)
            page["global_y_start"] = int(global_y)
            page["global_y_end"] = int(global_y + height)
            for local_cut in detect_gutter_cuts(image, config=config):
                cuts.add(int(global_y + local_cut))
                detected_cuts += 1
            global_y += height
    cuts.add(int(global_y))
    sorted_cuts = sorted(cuts)
    min_segment_height = max(
        16,
        int((config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}).get("raw_min_segment_height_px") or MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX),
    )
    segments: list[dict[str, Any]] = []
    for start_y, end_y in zip(sorted_cuts, sorted_cuts[1:]):
        source_height = int(end_y - start_y)
        if source_height < min_segment_height:
            continue
        source_slices, rendered_height = _segment_slices_for_range(
            pages,
            start_y=int(start_y),
            end_y=int(end_y),
            target_width=int(target_width),
        )
        if not source_slices:
            continue
        segment_index = len(segments)
        segments.append(
            {
                "segment_id": f"segment_{segment_index:04d}",
                "segment_index": segment_index,
                "global_y_start": int(start_y),
                "global_y_end": int(end_y),
                "source_height": source_height,
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_height),
                "source_slices": source_slices,
                "page_indices": sorted({int(item["page_index"]) for item in source_slices}),
                "page_span": [int(source_slices[0]["page_index"]), int(source_slices[-1]["page_index"])],
            }
        )
    stats = {
        "total_virtual_height": int(global_y),
        "target_width": int(target_width),
        "detected_gutter_cuts": int(detected_cuts),
        "cut_count": len(sorted_cuts),
    }
    return segments, int(target_width), stats


def _render_segment_image(
    *,
    segment: dict[str, Any],
    target_width: int,
    image_cache: dict[str, Any],
    get_bytes: Callable[[str], bytes],
) -> Any:
    from PIL import Image

    crops: list[Any] = []
    try:
        for source_slice in segment.get("source_slices") or []:
            if not isinstance(source_slice, dict):
                continue
            source_key = str(source_slice.get("source_key") or "")
            image = image_cache.get(source_key)
            if image is None:
                data = get_bytes(source_key)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                image_cache[source_key] = image
            crop = image.crop(
                (
                    int(source_slice.get("source_x_start") or 0),
                    int(source_slice.get("source_y_start") or 0),
                    int(source_slice.get("source_x_end") or image.width),
                    int(source_slice.get("source_y_end") or image.height),
                )
            )
            if crop.width != target_width:
                target_height = max(1, int(round(crop.height * float(target_width) / float(max(1, crop.width)))))
                crop = crop.resize((int(target_width), target_height), Image.Resampling.LANCZOS)
            crops.append(crop)
        rendered = Image.new("RGB", (int(target_width), max(1, sum(int(crop.height) for crop in crops))), "white")
        y = 0
        for crop in crops:
            rendered.paste(crop, (0, y))
            y += int(crop.height)
        return rendered
    except Exception:
        for crop in crops:
            try:
                crop.close()
            except Exception:
                pass
        raise


def pack_segments_into_sheets(segments: list[dict[str, Any]], *, span_threshold_px: int) -> list[dict[str, Any]]:
    sheets: list[dict[str, Any]] = []
    current: dict[str, Any] = {"columns": [{"used_units": 0, "segments": []}, {"used_units": 0, "segments": []}]}

    def flush_current() -> None:
        nonlocal current
        if any(column["segments"] for column in current["columns"]):
            current["sheet_index"] = len(sheets)
            sheets.append(current)
        current = {"columns": [{"used_units": 0, "segments": []}, {"used_units": 0, "segments": []}]}

    for segment in segments:
        row_span = 2 if int(segment.get("rendered_height") or 0) > int(span_threshold_px) else 1
        for column in current["columns"]:
            if int(column["used_units"]) + row_span <= 2:
                column["segments"].append({**segment, "row_span": row_span})
                column["used_units"] = int(column["used_units"]) + row_span
                break
        else:
            flush_current()
            current["columns"][0]["segments"].append({**segment, "row_span": row_span})
            current["columns"][0]["used_units"] = row_span
    flush_current()
    return sheets


def write_sheets(
    *,
    output_prefix: str,
    chapter_key: str,
    segments: list[dict[str, Any]],
    target_width: int,
    config: dict[str, Any],
    get_bytes: Callable[[str], bytes],
    put_bytes: Callable[[str, bytes, str], None],
    make_uri: Callable[[str], str],
) -> dict[str, Any]:
    from PIL import Image

    manhwa_config = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    padding = max(0, int(manhwa_config.get("raw_sheet_padding_px") or MANHWA_RAW_SHEET_PADDING_PX))
    span_threshold_px = max(
        64,
        int(manhwa_config.get("raw_sheet_span_threshold_px") or MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX),
    )
    jpeg_quality = max(40, min(95, int(manhwa_config.get("strip_jpeg_quality") or MANHWA_STRIP_JPEG_QUALITY)))
    sheets = pack_segments_into_sheets(segments, span_threshold_px=span_threshold_px)
    written_sheets: list[dict[str, Any]] = []
    for sheet_index, sheet in enumerate(sheets):
        image_cache: dict[str, Any] = {}
        try:
            used_columns = [column for column in sheet["columns"] if column["segments"]]
            if not used_columns:
                continue
            column_width = int(target_width)
            column_heights = [
                sum(int(segment.get("rendered_height") or 0) for segment in column["segments"])
                + padding * max(0, len(column["segments"]) - 1)
                for column in used_columns
            ]
            canvas_width = column_width * len(used_columns) + padding * max(0, len(used_columns) - 1)
            canvas_height = max(1, max(column_heights) if column_heights else 1)
            canvas = Image.new("RGB", (int(canvas_width), int(canvas_height)), "white")
            sheet_segments: list[dict[str, Any]] = []
            for column_index, column in enumerate(used_columns):
                x = column_index * (column_width + padding)
                y = 0
                for segment in column["segments"]:
                    rendered = _render_segment_image(
                        segment=segment,
                        target_width=column_width,
                        image_cache=image_cache,
                        get_bytes=get_bytes,
                    )
                    canvas.paste(rendered, (x, y))
                    placement = {
                        "sheet_index": int(sheet_index),
                        "column": int(column_index),
                        "x": int(x),
                        "y": int(y),
                        "width": int(rendered.width),
                        "height": int(rendered.height),
                        "row_span": int(segment.get("row_span") or 1),
                    }
                    sheet_segments.append(
                        {
                            "segment_id": str(segment.get("segment_id") or ""),
                            "segment_index": int(segment.get("segment_index") or 0),
                            "placement": placement,
                        }
                    )
                    segment["sheet_index"] = int(sheet_index)
                    segment["sheet_placement"] = placement
                    y += int(rendered.height) + padding
                    rendered.close()
            output = io.BytesIO()
            image_format = "jpeg"
            if canvas.width > 65000 or canvas.height > 65000:
                canvas.save(output, format="PNG", optimize=True)
                image_format = "png"
            else:
                canvas.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
            image_bytes = output.getvalue()
            segment_ids = [str(item["segment_id"]) for item in sheet_segments]
            sheet_key = raw_sheet_key(output_prefix, chapter_key, sheet_index, segment_ids, image_format)
            put_bytes(sheet_key, image_bytes, "image/png" if image_format == "png" else "image/jpeg")
            for segment in segments:
                if int(segment.get("sheet_index") or -1) == sheet_index:
                    segment["sheet_key"] = sheet_key
                    segment["sheet_uri"] = make_uri(sheet_key)
            written_sheets.append(
                {
                    "sheet_id": f"sheet_{sheet_index:04d}",
                    "sheet_index": int(sheet_index),
                    "sheet_key": sheet_key,
                    "sheet_uri": make_uri(sheet_key),
                    "image_format": image_format,
                    "image_bytes": len(image_bytes),
                    "width": int(canvas.width),
                    "height": int(canvas.height),
                    "padding_px": int(padding),
                    "span_threshold_px": int(span_threshold_px),
                    "segments": sheet_segments,
                }
            )
            canvas.close()
        finally:
            for image in image_cache.values():
                try:
                    image.close()
                except Exception:
                    pass
    return {
        "status": "ok",
        "sheet_count": len(written_sheets),
        "sheets": written_sheets,
        "sheet_span_threshold_px": int(span_threshold_px),
        "sheet_padding_px": int(padding),
    }


def build_and_write_chapter_sheets(
    *,
    output_prefix: str,
    chapter_key: str,
    pages: list[dict[str, Any]],
    config: dict[str, Any],
    get_bytes: Callable[[str], bytes],
    put_bytes: Callable[[str, bytes, str], None],
    make_uri: Callable[[str], str],
) -> dict[str, Any]:
    segments, target_width, segmentation = build_segments(pages=pages, config=config, get_bytes=get_bytes)
    sheet_result = write_sheets(
        output_prefix=output_prefix,
        chapter_key=chapter_key,
        segments=segments,
        target_width=target_width,
        config=config,
        get_bytes=get_bytes,
        put_bytes=put_bytes,
        make_uri=make_uri,
    )
    return {
        "status": sheet_result["status"],
        "segments": segments,
        "segment_count": len(segments),
        "target_width": int(target_width),
        "segmentation": segmentation,
        **sheet_result,
    }
