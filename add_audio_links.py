#!/usr/bin/env python3
"""
Add invisible clickable audio links to a PDF based on green speaker icons
and adjacent chapter numbers.

Requires: PyMuPDF (pymupdf) and Pillow.
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required. Install with: pip install pymupdf\n"
        f"Import error: {exc}"
    )

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install with: pip install pillow\n"
        f"Import error: {exc}"
    )

import numpy as np


CHAPTER_RE = re.compile(r"^\d+\.\d+$")
CHAPTER_INNER_RE = re.compile(r"(\d+[\.,]\d+)")


def build_audio_map(audio_dir: Path) -> dict[str, Path]:
    audio_map: dict[str, Path] = {}
    for path in audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp3":
            parts = path.name.split()
            if len(parts) >= 2 and parts[0].lower() == "kap":
                audio_map[parts[1]] = path.resolve()
    return audio_map


def image_similarity(a: Image.Image, b: Image.Image) -> float:
    """Return mean absolute pixel diff between two RGB images."""
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    return float(np.mean(np.abs(arr_a - arr_b)))


def render_page_image(page: fitz.Page, zoom: float) -> Image.Image:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def edges_from_gray(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="constant")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="constant")
    return gx + gy


def fft_conv2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    fa = np.fft.rfftn(a, shape, axes=(0, 1))
    fb = np.fft.rfftn(b, shape, axes=(0, 1))
    out = np.fft.irfftn(fa * fb, shape, axes=(0, 1))
    return out


def match_template_ssd(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    h, w = template.shape
    if h > image.shape[0] or w > image.shape[1]:
        return np.empty((0, 0), dtype=np.float32)
    template_flipped = np.flipud(np.fliplr(template))
    ones = np.ones_like(template, dtype=np.float32)
    sum_img2 = fft_conv2d(image * image, ones)
    sum_img_t = fft_conv2d(image, template_flipped)
    sum_t2 = float(np.sum(template * template))
    y0 = h - 1
    x0 = w - 1
    y1 = y0 + image.shape[0] - h + 1
    x1 = x0 + image.shape[1] - w + 1
    sum_img2_valid = sum_img2[y0:y1, x0:x1]
    sum_img_t_valid = sum_img_t[y0:y1, x0:x1]
    ssd = sum_img2_valid - (2.0 * sum_img_t_valid) + sum_t2
    return ssd


def find_speaker_rects(
    page: fitz.Page,
    speaker_png: Path,
    diff_threshold: float,
    debug_dir: Path | None = None,
) -> list[fitz.Rect]:
    speaker_img = Image.open(speaker_png).convert("RGB")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        speaker_img.save(debug_dir / "speaker_reference.png")
    rects: list[fitz.Rect] = []
    blocks = page.get_text("dict").get("blocks", [])
    if debug_dir is not None:
        render_page_image(page, 2.0).save(debug_dir / "page_render_2x.png")
    for block in blocks:
        if block.get("type") != 1:
            continue
        img_bytes = block.get("image")
        if not img_bytes:
            continue
        try:
            block_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            continue
        diff = image_similarity(speaker_img, block_img)
        if debug_dir is not None:
            bbox = block.get("bbox")
            bbox_str = "_".join(str(int(round(x))) for x in bbox)
            block_img.save(debug_dir / f"img_block_{bbox_str}_diff_{diff:.2f}.png")
        if diff <= diff_threshold:
            rects.append(fitz.Rect(block.get("bbox")))
    return rects


def find_chapter_near(page: fitz.Page, icon_rect: fitz.Rect) -> tuple[str | None, fitz.Rect | None]:
    search_rect = fitz.Rect(
        icon_rect.x1 + (icon_rect.width * 0.1),
        icon_rect.y0 - (icon_rect.height * 0.4),
        icon_rect.x1 + (icon_rect.width * 8.0),
        icon_rect.y1 + (icon_rect.height * 0.4),
    )
    words = page.get_text("words", clip=search_rect)
    candidates: list[tuple[float, str, fitz.Rect]] = []
    icon_center = ((icon_rect.x0 + icon_rect.x1) / 2.0, (icon_rect.y0 + icon_rect.y1) / 2.0)

    for w in words:
        text = w[4]
        match = CHAPTER_INNER_RE.search(text)
        if match:
            chapter = match.group(1).replace(",", ".")
        else:
            continue
        word_rect = fitz.Rect(w[:4])
        word_center = ((word_rect.x0 + word_rect.x1) / 2.0, (word_rect.y0 + word_rect.y1) / 2.0)
        dist = (word_center[0] - icon_center[0]) ** 2 + (word_center[1] - icon_center[1]) ** 2
        candidates.append((dist, chapter, word_rect))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def find_speaker_rects_template(
    page: fitz.Page,
    speaker_png: Path,
    render_zoom: float,
    search_scale: float,
    template_scales: list[float],
    max_matches: int,
    match_threshold: float | None,
    list_matches: bool,
    debug_dir: Path | None = None,
) -> list[fitz.Rect]:
    page_img = render_page_image(page, render_zoom)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        page_img.save(debug_dir / f"page_render_{render_zoom:.1f}x.png")

    page_gray = np.asarray(page_img.convert("L"), dtype=np.float32) / 255.0
    page_small = np.asarray(
        page_img.resize(
            (max(1, int(page_img.width * search_scale)), max(1, int(page_img.height * search_scale))),
            Image.Resampling.LANCZOS,
        ).convert("L"),
        dtype=np.float32,
    ) / 255.0

    speaker_img = Image.open(speaker_png).convert("L")
    speaker_base = np.asarray(speaker_img, dtype=np.float32) / 255.0

    page_small_edges = edges_from_gray(page_small)
    page_small_edges /= (page_small_edges.max() + 1e-6)

    matches: list[tuple[float, fitz.Rect]] = []

    for scale in template_scales:
        sw = max(5, int(round(speaker_img.width * scale)))
        sh = max(5, int(round(speaker_img.height * scale)))
        if sw <= 0 or sh <= 0:
            continue
        tmpl_scaled = speaker_img.resize((sw, sh), Image.Resampling.LANCZOS)
        tmpl_small = tmpl_scaled.resize(
            (max(5, int(round(sw * search_scale))), max(5, int(round(sh * search_scale)))),
            Image.Resampling.LANCZOS,
        )

        tmpl_small_arr = np.asarray(tmpl_small, dtype=np.float32) / 255.0
        tmpl_small_edges = edges_from_gray(tmpl_small_arr)
        tmpl_small_edges /= (tmpl_small_edges.max() + 1e-6)

        ssd = match_template_ssd(page_small_edges, tmpl_small_edges)
        if ssd.size == 0:
            continue

        flat = ssd.ravel()
        k = min(max_matches, flat.size)
        idxs = np.argpartition(flat, k - 1)[:k]

        for idx in idxs:
            y, x = divmod(int(idx), ssd.shape[1])
            score = float(flat[idx] / (tmpl_small_edges.size + 1e-6))
            x_render = x / search_scale
            y_render = y / search_scale
            w_render = sw
            h_render = sh
            rect = fitz.Rect(
                x_render / render_zoom,
                y_render / render_zoom,
                (x_render + w_render) / render_zoom,
                (y_render + h_render) / render_zoom,
            )
            matches.append((score, rect))

    if not matches:
        return []

    matches.sort(key=lambda x: x[0])
    best_score = matches[0][0]
    score_limit = match_threshold if match_threshold is not None else (best_score * 1.5)
    filtered = [m for m in matches if m[0] <= score_limit]

    kept: list[tuple[float, fitz.Rect]] = []
    for score, rect in filtered:
        too_close = False
        for existing_score, existing in kept:
            if rect.intersects(existing) or rect.tl.distance_to(existing.tl) < min(rect.width, rect.height) * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append((score, rect))
        if len(kept) >= max_matches:
            break

    if list_matches:
        print("Template matches (score, rect):")
        for score, rect in kept:
            print(f"  {score:.6f} -> {rect}")

    if debug_dir is not None:
        for i, (score, rect) in enumerate(kept, start=1):
            x0 = int(round(rect.x0 * render_zoom))
            y0 = int(round(rect.y0 * render_zoom))
            x1 = int(round(rect.x1 * render_zoom))
            y1 = int(round(rect.y1 * render_zoom))
            crop = page_img.crop((x0, y0, x1, y1))
            crop.save(debug_dir / f"match_{i:02d}_score_{score:.6f}_crop.png")

    return [rect for _, rect in kept]


def add_audio_links(
    input_pdf: Path,
    speaker_png: Path,
    audio_dir: Path,
    output_pdf: Path,
    page_number: int,
    diff_threshold: float,
    debug_dir: Path | None,
    render_zoom: float,
    search_scale: float,
    template_scales: list[float],
    max_matches: int,
    match_threshold: float | None,
    list_matches: bool,
    link_kind: str,
    relative_links: bool,
) -> None:
    audio_map = build_audio_map(audio_dir)
    if not audio_map:
        raise SystemExit(f"No audio files found in {audio_dir}")

    doc = fitz.open(str(input_pdf))
    page_index = page_number - 1
    if page_index < 0 or page_index >= doc.page_count:
        raise SystemExit(f"Page {page_number} out of range (1-{doc.page_count})")

    # Create a single-page PDF for easier testing.
    single = fitz.open()
    single.insert_pdf(doc, from_page=page_index, to_page=page_index)
    page_only_pdf = output_pdf.with_name(f'page{page_number}_only' + output_pdf.suffix)
    single.save(str(page_only_pdf))
    single.close()
    doc.close()

    # Re-open single-page PDF and add links.
    out_doc = fitz.open(str(page_only_pdf))
    page = out_doc[0]

    speaker_rects = find_speaker_rects(page, speaker_png, diff_threshold, debug_dir)
    if not speaker_rects:
        speaker_rects = find_speaker_rects_template(
            page,
        speaker_png,
        render_zoom=render_zoom,
        search_scale=search_scale,
        template_scales=template_scales,
        max_matches=max_matches,
        match_threshold=match_threshold,
        list_matches=list_matches,
        debug_dir=debug_dir,
    )
    if not speaker_rects:
        print(
            "No speaker icons detected. Try adjusting --diff-threshold, "
            "--template-scales, or inspect --debug-out crops."
        )

    audio_cache: dict[Path, bytes] = {}

    for rect in speaker_rects:
        chapter, chapter_rect = find_chapter_near(page, rect)
        if not chapter:
            print(f"No chapter found near icon at {rect}")
            continue

        audio_path = audio_map.get(chapter)
        if not audio_path:
            print(f"No audio file found for chapter {chapter}")
            continue

        link_rect = rect
        if chapter_rect is not None:
            link_rect = rect | chapter_rect

        if link_kind == "attach":
            buffer_ = audio_cache.get(audio_path)
            if buffer_ is None:
                buffer_ = audio_path.read_bytes()
                audio_cache[audio_path] = buffer_
            point = fitz.Point(link_rect.x0, link_rect.y0)
            annot = page.add_file_annot(point, buffer_, audio_path.name, desc=f"Kap {chapter}")
            annot.set_rect(link_rect)
            annot.set_border(width=0)
            try:
                annot.set_opacity(0)
            except Exception:
                pass
        elif link_kind == "uri":
            link_def = {
                "kind": fitz.LINK_URI,
                "uri": audio_path.as_uri(),
                "from": link_rect,
                "border": [0, 0, 0],
            }
        else:
            file_target = audio_path
            if relative_links:
                file_target = Path(os.path.relpath(audio_path, start=output_pdf.parent))
            link_def = {
                "kind": fitz.LINK_LAUNCH,
                "file": str(file_target).replace("\\\\", "/"),
                "from": link_rect,
                "border": [0, 0, 0],
            }
        if link_kind != "attach":
            page.insert_link(link_def)
        print(f"Linked chapter {chapter} -> {audio_path.name}")

    out_doc.save(str(output_pdf))
    out_doc.close()
    print(f"Wrote page-only PDF: {page_only_pdf}")
    print(f"Wrote linked PDF: {output_pdf}")


def main() -> None:
    if len(sys.argv) == 1:
        print(
            "Usage: add_audio_links.py <pdf> <speaker_png> <audio_dir> <output_pdf> [--page N] [--diff-threshold D]\n"
            "Example: add_audio_links.py 'BASIC_Svenska i Engelska parken VT26 BASTEXTER.pdf' "
            "green_speaker_symbol.png audio_files page4_with_audio.pdf --page 4\n"
            "Optional: --debug-out DIR --render-zoom 2.0 --search-scale 0.25 --template-scales 0.6,0.8,1.0,1.2,1.4,1.6 "
            "--max-matches 10 --match-threshold 0.05 --list-matches --link-kind launch|uri|attach --relative-links"
        )
        return

    args = list(sys.argv[1:])
    page_number = 4
    diff_threshold = 8.0
    debug_dir: Path | None = None
    render_zoom = 2.0
    search_scale = 0.25
    template_scales = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    max_matches = 10
    match_threshold: float | None = None
    list_matches = False
    link_kind = "launch"
    relative_links = False

    if "--page" in args:
        idx = args.index("--page")
        page_number = int(args[idx + 1])
        del args[idx : idx + 2]

    if "--diff-threshold" in args:
        idx = args.index("--diff-threshold")
        diff_threshold = float(args[idx + 1])
        del args[idx : idx + 2]

    if "--debug-out" in args:
        idx = args.index("--debug-out")
        debug_dir = Path(args[idx + 1])
        del args[idx : idx + 2]

    if "--render-zoom" in args:
        idx = args.index("--render-zoom")
        render_zoom = float(args[idx + 1])
        del args[idx : idx + 2]

    if "--search-scale" in args:
        idx = args.index("--search-scale")
        search_scale = float(args[idx + 1])
        del args[idx : idx + 2]

    if "--template-scales" in args:
        idx = args.index("--template-scales")
        template_scales = [float(x) for x in args[idx + 1].split(",") if x.strip()]
        del args[idx : idx + 2]

    if "--max-matches" in args:
        idx = args.index("--max-matches")
        max_matches = int(args[idx + 1])
        del args[idx : idx + 2]

    if "--match-threshold" in args:
        idx = args.index("--match-threshold")
        match_threshold = float(args[idx + 1])
        del args[idx : idx + 2]

    if "--list-matches" in args:
        list_matches = True
        args.remove("--list-matches")

    if "--link-kind" in args:
        idx = args.index("--link-kind")
        link_kind = args[idx + 1].lower()
        del args[idx : idx + 2]

    if "--relative-links" in args:
        relative_links = True
        args.remove("--relative-links")

    if len(args) != 4:
        raise SystemExit("Expected 4 positional args: <pdf> <speaker_png> <audio_dir> <output_pdf>")

    input_pdf = Path(args[0])
    speaker_png = Path(args[1])
    audio_dir = Path(args[2])
    output_pdf = Path(args[3])

    add_audio_links(
        input_pdf,
        speaker_png,
        audio_dir,
        output_pdf,
        page_number,
        diff_threshold,
        debug_dir,
        render_zoom,
        search_scale,
        template_scales,
        max_matches,
        match_threshold,
        list_matches,
        link_kind,
        relative_links,
    )


if __name__ == "__main__":
    main()
