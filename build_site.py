#!/usr/bin/env python3
"""
Build a static website from the Swedish PDF with clickable audio hotspots.

Outputs:
- pages/page-###.png
- data.json

Requires: pymupdf, pillow, numpy
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

CHAPTER_INNER_RE = re.compile(r"(\d+)\s*[\.,]\s*(\d+)")


def build_audio_map(audio_dir: Path) -> dict[str, Path]:
    audio_map: dict[str, Path] = {}
    for path in audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp3":
            parts = path.name.split()
            if len(parts) >= 2 and parts[0].lower() == "kap":
                audio_map[parts[1]] = path.resolve()
    return audio_map


def render_page_image(page: fitz.Page, zoom: float) -> Image.Image:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


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


def find_speaker_rects_template(
    page: fitz.Page,
    speaker_png: Path,
    render_zoom: float,
    search_scale: float,
    template_scales: list[float],
    max_matches: int,
    match_threshold: float | None,
    force_count: int | None = None,
    debug_dir: Path | None = None,
) -> list[fitz.Rect]:
    page_img = render_page_image(page, render_zoom)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        page_img.save(debug_dir / f"page_render_{render_zoom:.1f}x.png")

    page_small = np.asarray(
        page_img.resize(
            (max(1, int(page_img.width * search_scale)), max(1, int(page_img.height * search_scale))),
            Image.Resampling.LANCZOS,
        ).convert("L"),
        dtype=np.float32,
    ) / 255.0

    speaker_img = Image.open(speaker_png).convert("L")

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
        per_scale = max_matches
        if force_count is not None and per_scale < force_count * 3:
            per_scale = force_count * 3
        k = min(per_scale, flat.size)
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
    if force_count is not None:
        filtered = matches
    else:
        best_score = matches[0][0]
        score_limit = match_threshold if match_threshold is not None else (best_score * 1.5)
        filtered = [m for m in matches if m[0] <= score_limit]

    kept: list[tuple[float, fitz.Rect]] = []
    limit = force_count if force_count is not None else max_matches
    for score, rect in filtered:
        too_close = False
        for _, existing in kept:
            if rect.intersects(existing) or rect.tl.distance_to(existing.tl) < min(rect.width, rect.height) * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append((score, rect))
        if len(kept) >= limit:
            break

    if debug_dir is not None:
        for i, (score, rect) in enumerate(kept, start=1):
            x0 = int(round(rect.x0 * render_zoom))
            y0 = int(round(rect.y0 * render_zoom))
            x1 = int(round(rect.x1 * render_zoom))
            y1 = int(round(rect.y1 * render_zoom))
            crop = page_img.crop((x0, y0, x1, y1))
            crop.save(debug_dir / f"match_{i:02d}_score_{score:.6f}_crop.png")

    return [rect for _, rect in kept]


def normalize_chapter(text: str) -> str | None:
    match = CHAPTER_INNER_RE.search(text)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    cleaned = text.strip().replace(",", ".").replace(" ", "")
    return cleaned or None


def extract_chapter_tokens(words: list[list]) -> list[tuple[str, fitz.Rect]]:
    candidates: list[tuple[str, fitz.Rect]] = []
    for idx, w in enumerate(words):
        text = w[4]
        match = CHAPTER_INNER_RE.fullmatch(text)
        if match:
            chapter = f"{match.group(1)}.{match.group(2)}"
            candidates.append((chapter, fitz.Rect(w[:4])))
            continue

        match_single = re.fullmatch(r"(\d+)[\.,]", text)
        if match_single and idx + 1 < len(words):
            next_text = words[idx + 1][4]
            match_next = re.fullmatch(r"(\d+)", next_text)
            if match_next:
                chapter = f"{match_single.group(1)}.{match_next.group(1)}"
                rect = fitz.Rect(w[:4]) | fitz.Rect(words[idx + 1][:4])
                candidates.append((chapter, rect))
                continue

        if re.fullmatch(r"(\d+)", text) and idx + 2 < len(words):
            mid = words[idx + 1][4]
            next_text = words[idx + 2][4]
            if re.fullmatch(r"[\.,]", mid) and re.fullmatch(r"(\d+)", next_text):
                chapter = f"{text}.{next_text}"
                rect = fitz.Rect(w[:4]) | fitz.Rect(words[idx + 1][:4]) | fitz.Rect(words[idx + 2][:4])
                candidates.append((chapter, rect))
                continue

    return candidates


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

    for chapter, word_rect in extract_chapter_tokens(words):
        word_center = ((word_rect.x0 + word_rect.x1) / 2.0, (word_rect.y0 + word_rect.y1) / 2.0)
        dist = (word_center[0] - icon_center[0]) ** 2 + (word_center[1] - icon_center[1]) ** 2
        candidates.append((dist, chapter, word_rect))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]




def load_manual_overrides(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Manual overrides file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Manual overrides JSON must be an object mapping page -> hotspots")
    manual: dict[str, dict] = {}
    for key, value in raw.items():
        page_key = str(key)
        if isinstance(value, list):
            hotspots = value
            replace = False
        elif isinstance(value, dict):
            hotspots = value.get("hotspots", [])
            replace = bool(value.get("replace", False))
        else:
            continue
        cleaned = []
        for item in hotspots:
            if not isinstance(item, dict):
                continue
            chapter = item.get("chapter")
            audio = item.get("audio")
            x = item.get("x")
            y = item.get("y")
            w = item.get("w")
            h = item.get("h")
            if x is None or y is None or w is None or h is None:
                continue
            try:
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
            except Exception:
                continue
            if chapter:
                chapter = normalize_chapter(str(chapter)) or str(chapter)
            if audio is not None:
                audio = str(audio)
            cleaned.append({"chapter": chapter, "audio": audio, "x": x, "y": y, "w": w, "h": h})
        manual[page_key] = {"replace": replace, "hotspots": cleaned}
    return manual

def parse_pages(pages_arg: str | None, page_count: int) -> list[int]:
    if not pages_arg:
        return list(range(1, page_count + 1))
    pages: list[int] = []
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    pages = [p for p in pages if 1 <= p <= page_count]
    return sorted(set(pages))


def load_overrides(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Overrides file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Overrides JSON must be an object mapping page -> [chapters]")
    overrides: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            continue
        chapters: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = normalize_chapter(item)
            if normalized:
                chapters.append(normalized)
        overrides[str(key)] = chapters
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--pages", default=None)
    parser.add_argument("--render-zoom", type=float, default=2.0)
    parser.add_argument("--search-scale", type=float, default=0.25)
    parser.add_argument("--template-scales", default="0.6,0.8,1.0,1.2,1.4,1.6")
    parser.add_argument("--max-matches", type=int, default=10)
    parser.add_argument("--match-threshold", type=float, default=0.04)
    parser.add_argument("--debug-out", default=None)
    parser.add_argument("--overrides", default=None)
    parser.add_argument("--manual-overrides", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    speaker_png = Path(args.speaker)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    thumbs_dir = out_dir / "thumbs"
    pages_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    audio_map = build_audio_map(audio_dir)
    if not audio_map:
        raise SystemExit(f"No audio files found in {audio_dir}")

    doc = fitz.open(str(pdf_path))
    pages_to_process = parse_pages(args.pages, doc.page_count)
    template_scales = [float(x) for x in args.template_scales.split(",") if x.strip()]
    overrides = load_overrides(Path(args.overrides)) if args.overrides else {}
    manual_overrides = load_manual_overrides(Path(args.manual_overrides)) if args.manual_overrides else {}

    data = {
        "meta": {
            "pdf": pdf_path.name,
            "render_zoom": args.render_zoom,
            "audio_base": str(os.path.relpath(audio_dir, start=out_dir)).replace("\\", "/"),
        },
        "pages": [],
    }

    for page_number in pages_to_process:
        page = doc[page_number - 1]
        page_img = render_page_image(page, args.render_zoom)
        img_name = f"page-{page_number:03d}.png"
        img_path = pages_dir / img_name
        page_img.save(img_path)

        thumb_name = f"thumb-{page_number:03d}.jpg"
        thumb_path = thumbs_dir / thumb_name
        thumb_img = page_img.copy()
        thumb_img.thumbnail((220, 2200), Image.Resampling.LANCZOS)
        thumb_img.save(thumb_path, format="JPEG", quality=72, optimize=True)

        debug_dir = None
        if args.debug_out:
            debug_dir = Path(args.debug_out) / f"page_{page_number:03d}"

        speaker_rects = find_speaker_rects_template(
            page,
            speaker_png,
            render_zoom=args.render_zoom,
            search_scale=args.search_scale,
            template_scales=template_scales,
            max_matches=args.max_matches,
            match_threshold=args.match_threshold,
            debug_dir=debug_dir,
        )
        speaker_rects = sorted(speaker_rects, key=lambda r: (r.y0, r.x0))
        override_list = overrides.get(str(page_number), [])
        manual_entry = manual_overrides.get(str(page_number))
        if override_list and len(speaker_rects) < len(override_list):
            wide_scales = sorted(set(template_scales + [0.35, 0.45, 0.55, 0.65, 0.75, 1.8]))
            speaker_rects = find_speaker_rects_template(
                page,
                speaker_png,
                render_zoom=args.render_zoom,
                search_scale=args.search_scale,
                template_scales=wide_scales,
                max_matches=max(args.max_matches, len(override_list) * 3),
                match_threshold=None,
                force_count=len(override_list),
                debug_dir=debug_dir,
            )
            speaker_rects = sorted(speaker_rects, key=lambda r: (r.y0, r.x0))

        page_entry = {
            "page": page_number,
            "image": f"pages/{img_name}",
            "thumb": f"thumbs/{thumb_name}",
            "width": page_img.width,
            "height": page_img.height,
            "hotspots": [],
        }

        manual_hotspots = []
        manual_replace = False
        if manual_entry:
            manual_hotspots = manual_entry.get("hotspots", [])
            manual_replace = bool(manual_entry.get("replace", False))

        if not manual_replace:
            rect_entries: list[dict] = []
            for rect in speaker_rects:
                chapter, _ = find_chapter_near(page, rect)
                chapter = normalize_chapter(chapter) if chapter else None
                rect_entries.append({"rect": rect, "chapter": chapter})

            missing = [entry for entry in rect_entries if not entry["chapter"]]
            for entry, chapter in zip(missing, override_list):
                entry["chapter"] = normalize_chapter(chapter) or chapter

            for entry in rect_entries:
                chapter = entry["chapter"]
                if not chapter:
                    continue
                audio_path = audio_map.get(chapter)
                if not audio_path:
                    continue

                rect = entry["rect"]
                page_w = page.rect.width
                page_h = page.rect.height
                hotspot = {
                    "chapter": chapter,
                    "audio": f"{data['meta']['audio_base']}/{audio_path.name}",
                    "x": rect.x0 / page_w,
                    "y": rect.y0 / page_h,
                    "w": rect.width / page_w,
                    "h": rect.height / page_h,
                }
                page_entry["hotspots"].append(hotspot)

        # Manual hotspots are appended after auto-detected ones.
        for item in manual_hotspots:
            chapter = item.get("chapter")
            audio = item.get("audio")
            if not audio and chapter:
                audio_path = audio_map.get(chapter)
                if not audio_path:
                    continue
                audio = f"{data['meta']['audio_base']}/{audio_path.name}"
            if not audio:
                continue
            hotspot = {
                "chapter": chapter or "",
                "audio": audio,
                "x": item.get("x"),
                "y": item.get("y"),
                "w": item.get("w"),
                "h": item.get("h"),
            }
            page_entry["hotspots"].append(hotspot)

        data["pages"].append(page_entry)

    doc.close()

    data_path = out_dir / "data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
