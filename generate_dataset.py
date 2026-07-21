#!/usr/bin/env python3
"""
reasonseg — one-file SA-1B referring-expression data engine.

A tidied, self-contained rewrite of the newsam "gemini" pipeline (click_describe.py +
gemini_prompts.py + gemini_negatives/*) collapsed into a single script. It builds a
referring-expression *segmentation* dataset from SA-1B images where every positive sample
is tagged with ONE of eight reasoning categories, and every image also gets verified hard
NEGATIVE phrases.

Pipeline per image (stage `generate`):
  1. SAM3 interactive predictor auto-clicks candidate objects (clean, sensibly sized,
     non-background) and dedups them  -> a handful of distinct objects.
  2. Gemini picks the object that best fits a TARGET reasoning category, and writes a
     referring expression of that category (or declines if the scene can't support it).
  3. The mask is refined: text-segment the image with a plain "simple_prompt" and keep the
     concept mask that best overlaps the click mask (falls back to the click mask).
  4. Verification gates (1 programmatic + 4 Gemini) — grounding, boundary, unique/meaningful,
     and CATEGORY-MATCH. PASS iff all pass. Everything is logged to results.jsonl.

Stage `negatives`: propose plausible-but-absent concept phrases per image (Gemini), verify
absence (Gemini), and assemble an image_negatives.json sidecar.

Stage `export`: pack PASS records into COCO train/test splits (category carries the referring
expression + reasoning_type + simple_prompt) plus the merged negatives sidecar.

Stage `download`: fetch + extract an SA-1B tar of .jpg images.

Run stages independently or `all`. Uses the vendored ./sam3 clone. Requires GEMINI_API_KEY
in the environment.

    GEMINI_API_KEY=... ./generate_dataset.py all --limit 50
"""
from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import io
import json
import os
import random
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------------------------------------------------------
# Paths / config (override via CLI or env)
# --------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
SAM3_ROOT = REPO / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

# The base SAM3 checkpoint. The old newsam const (/workspace/sam/sam3_original.pt) was
# deleted; default to the surviving HuggingFace-cache copy. Override with --checkpoint.
DEFAULT_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/workspace/.cache/huggingface/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
)
DEFAULT_IMAGE_DIR = REPO / "data" / "images"
DEFAULT_OUT_DIR = REPO / "out"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# The first SA-1B tar (provided by the user). Override with --tar-url.
DEFAULT_TAR_URL = (
    "https://scontent.xx.fbcdn.net/m1/v/t6/An_YmP5OIPXun-vu3hkckAZZ2s4lPYoVkiyvCcWiVY21mu1Ng5"
    "_1HeCa2CWiSTsskj8HQ8bN013HxNpYDdSC_7jWQq_svcg.tar?_nc_gid&ccb=10-5&oh=00_AQA5W6jTR3T-"
    "kgikAKOIke1kUvryVixrpWQPmrlMGzwNKQ&oe=6A7551A8&_nc_sid=0fdd51"
)

# candidate sampling (from click_describe.py, unchanged)
N_CANDIDATES = 16
MIN_AREA, MAX_AREA = 0.01, 0.25
MIN_SCORE = 0.80
N_SELECT = 8
DEDUP_IOU = 0.5
TEXT_THRESHOLD = 0.5
MIN_OVERLAP_IOU = 0.10
HILITE = (57, 255, 20)

# --------------------------------------------------------------------------------------
# Reasoning categories — the 8 sample types this engine produces
# --------------------------------------------------------------------------------------
CATEGORIES = [
    {
        "id": "object_attribute",
        "name": "Object & attributes",
        "definition": (
            "The object is identified directly by its category and intrinsic properties "
            "(colour, size, material, state, clothing worn), WITHOUT reasoning about other "
            "objects."
        ),
        "examples": ["the red car", "the tall tree", "the man wearing a blue shirt",
                     "the open window", "the small dog"],
    },
    {
        "id": "spatial",
        "name": "Spatial reasoning",
        "definition": (
            "The object is identified by its location in the image or relative to other "
            "objects (left/right/above/below/behind/in front/between/nearest)."
        ),
        "examples": ["the tree on the right", "the car behind the truck",
                     "the person between the two benches", "the bicycle nearest the camera",
                     "the window above the door"],
    },
    {
        "id": "ordinal_comparative",
        "name": "Ordinal & comparative reasoning",
        "definition": (
            "The object is selected by comparing it with similar objects using order or a "
            "measurable property (nth-from-an-end, largest/tallest/smallest, closest/leftmost)."
        ),
        "examples": ["the second person from the left", "the largest suitcase",
                     "the tallest building", "the leftmost chair",
                     "the closest car to the entrance"],
    },
    {
        "id": "relational",
        "name": "Relational reasoning",
        "definition": (
            "The object is identified through a DIRECT relationship with another object or "
            "person — interaction, possession, or part-whole."
        ),
        "examples": ["the person holding the umbrella", "the dog's leash",
                     "the man's backpack", "the front wheel of the bicycle",
                     "the cup on the table"],
    },
    {
        "id": "multi_hop",
        "name": "Multi-hop reasoning",
        "definition": (
            "Identifying the target requires first locating one or more INTERMEDIATE objects "
            "before reaching the final object (a chain of two or more references)."
        ),
        "examples": ["the backpack belonging to the person sitting on the bench",
                     "the phone held by the woman wearing a red jacket",
                     "the cup on the desk of the man using the laptop",
                     "the suitcase beside the passenger wearing glasses",
                     "the bicycle behind the truck nearest the entrance"],
    },
    {
        "id": "constraint_composition",
        "name": "Constraint composition",
        "definition": (
            "The object must satisfy SEVERAL independent conditions simultaneously "
            "(attribute + position + comparative, etc.)."
        ),
        "examples": ["the red car closest to the building", "the tallest tree on the right",
                     "the woman wearing glasses standing beside the bus",
                     "the smallest dog near the bench",
                     "the second person from the left holding a phone"],
    },
    {
        "id": "commonsense_affordance",
        "name": "Commonsense & affordance reasoning",
        "definition": (
            "Identifying the object requires knowledge about how objects are typically used, "
            "owned, or function, rather than explicit visual relations alone."
        ),
        "examples": ["the object used to unlock the door", "the driver's seat",
                     "the chair most likely being used by the person working",
                     "the container used for drinking water",
                     "the tool used to tighten the bolt"],
    },
    {
        "id": "counterfactual_predictive",
        "name": "Counterfactual & predictive reasoning",
        "definition": (
            "Identifying the object requires reasoning about hypothetical situations, future "
            "outcomes, or physical consequences that are not directly observable."
        ),
        "examples": ["the object that would block the doorway if moved",
                     "the person who would reach the exit first",
                     "the tree that would remain visible if the truck disappeared",
                     "the object most likely to fall if the table were bumped",
                     "the vehicle that would need to move first to let the blue car leave"],
    },
]
CATEGORIES_BY_ID = {c["id"]: c for c in CATEGORIES}

# --------------------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------------------
SELECT_PROMPT_TMPL = """You are choosing which object to annotate for a referring-expression segmentation dataset.

Several candidate objects in this image are each marked with a numbered white badge sitting on the object. The badge is only a marker — judge the object underneath it, not the badge.

We want to write a referring expression of this specific reasoning type:
  TYPE: {cat_name}
  DEFINITION: {cat_def}
  EXAMPLES: {cat_examples}

Pick the ONE numbered object that would make the BEST example of THIS reasoning type — i.e. an object for which a natural, unambiguous expression of this type genuinely exists in the scene. If several work, prefer the clearest single, well-delineated object.

There are {n} numbered objects. Return ONLY a JSON object:
{{"choice": <badge number 1..{n}>, "reasoning": "<one sentence: why this object suits the {cat_name} type>"}}"""

SELECT_AUTO_PROMPT_TMPL = """You are choosing BOTH which object to annotate AND which reasoning type to use, for a referring-expression segmentation dataset.

Several candidate objects in this image are each marked with a numbered white badge sitting on the object. The badge is only a marker — judge the object underneath it, not the badge.

Choose the reasoning type from these {n_cats} options:
{cat_block}

Pick the ONE (object, reasoning type) pairing that makes the BEST, most natural and unambiguous example: an object for which a genuinely clear, grounded referring expression of the chosen type exists in THIS scene. Prefer a clearly delineated single object, and a reasoning type the scene strongly supports. Do not force a type the scene does not naturally afford.

There are {n} numbered objects. Return ONLY a JSON object:
{{"choice": <badge number 1..{n}>, "category": "<one of: {cat_ids}>", "reasoning": "<one sentence: why this object + type is the best example>"}}"""

DESCRIBE_PROMPT_TMPL = """You are building a referring-expression segmentation dataset. ONE object in this image is outlined in bright green (the outline is only a marker drawn on top — NOT part of the object and NOT its colour; describe the object by its own appearance).

Write a referring expression that UNIQUELY identifies THAT outlined object using this SPECIFIC reasoning type:
  TYPE: {cat_name}
  DEFINITION: {cat_def}
  EXAMPLES of this type: {cat_examples}

Hard requirements:
1. UNIQUE — if ten people read your expression and each clicked the object it means, all ten must click the SAME outlined object. Never produce something two objects could satisfy.
2. ON-TYPE — the expression must genuinely exhibit the {cat_name} reasoning above, not a different type. (e.g. for Spatial it must use location/relative position; for Ordinal it must use order or a measurable comparison; for Multi-hop it must route through an intermediate object; etc.)
3. GROUNDED — every object your expression mentions (the target AND any anchor/intermediate objects) must actually be visible in the image.
4. Natural and as short as the type allows. Do not mention the outline or any annotation.

If the outlined object CANNOT be described with this reasoning type in a natural, unambiguous, grounded way, do NOT force it — return {{"possible": false, "reason": "<why not>"}}.

Also give "simple_prompt": a short, plain, lowercase noun phrase naming the object's KIND, suitable for a text object detector to find it and others of its kind (e.g. "car", "chairs", "the lamp"). Category only, no disambiguating detail.

Return ONLY a JSON object:
{{"possible": true, "object": "<plain object name>", "prompt": "<the {cat_name} referring expression, lowercase>", "simple_prompt": "<plain noun phrase of the object's kind>", "reasoning": "<why it uniquely identifies the outlined object and how it exhibits the {cat_name} type>"}}"""

_VERIFY_PREAMBLE = """You are quality-checking a referring-expression annotation for object segmentation. ONE object in this image is outlined in bright green. The outline is only a marker drawn on top — NOT part of any object and NOT its colour; judge the object underneath, never the outline.

Be STRICT and SKEPTICAL: this is a quality gate, most candidates should FAIL, and when in any doubt you FAIL.
"""

VERIFY_GROUNDING_TMPL = _VERIFY_PREAMBLE + """
Check ONE thing only: GROUNDING — do the green-outlined pixels actually show a "{object}"?
1. Look ONLY inside the outline. Ignore every other object, however salient.
2. In "outlined_thing", state in your OWN words what the outlined pixels are.
3. PASS only if that thing genuinely IS a "{object}"; FAIL if it is a different object, background (wall/door/ground/sky/water), or empty area.
Decide "outlined_thing" BEFORE the verdict.
Return ONLY JSON: {{"outlined_thing": "<...>", "verdict": "pass" or "fail", "reason": "<one sentence>"}}"""

VERIFY_DISTINCT_TMPL = _VERIFY_PREAMBLE + """
Check ONE thing only: are the BOUNDARIES of the outlined object clear — is it obvious where it starts and stops?
PASS if it has well-defined edges (being part of something larger, or one of many, is fine). FAIL only if the boundary genuinely merges several things or its extent is ambiguous. Boundary clear -> PASS.
Return ONLY JSON: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""

VERIFY_UNIQUE_TMPL = _VERIFY_PREAMBLE + """
The proposed referring expression for the outlined object is:
    "{prompt}"
Check ONE thing only: does it pick out the outlined object UNIQUELY? Imagine ten readers each click the object it names — PASS only if all ten click the SAME outlined object. FAIL if more than one object fits, or the disambiguator is a bare ordinal that assumes an ordering the viewer doesn't share, or a superlative whose extreme is not obvious (several roughly tied), or a vague/subjective detail. PASS only when it anchors to an unmistakable property/landmark true of this object and no other.
Return ONLY JSON: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""

VERIFY_CATEGORY_TMPL = _VERIFY_PREAMBLE + """
The proposed referring expression for the outlined object is:
    "{prompt}"
It is supposed to be an example of this reasoning type:
  TYPE: {cat_name}
  DEFINITION: {cat_def}
Check ONE thing only: does the expression GENUINELY exhibit this reasoning type (and not merely a simpler type)? For example, "Multi-hop" must route through an intermediate object before reaching the target; "Ordinal & comparative" must use order or a measurable comparison among similar objects; "Spatial" must use location/relative position; "Commonsense & affordance" must rely on how the object is used/owned; "Counterfactual & predictive" must reason about a hypothetical/future/physical consequence. FAIL if the expression is really a different (usually simpler) type, or does not require the intended reasoning to resolve.
Return ONLY JSON: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""


NEG_PROPOSE_TMPL = """You are proposing HARD NEGATIVE object phrases for an instance-segmentation dataset.

Look at this image. Propose up to {k} short, plain, lowercase noun phrases naming object TYPES that a person might plausibly expect in a scene like this, but that are NOT actually visible anywhere in THIS image. They should be tempting-but-absent (e.g. "bicycle" for a street with no bike), not random unrelated things. Do NOT include anything that is present.

These concepts ARE present, so never propose them or synonyms of them:
{present}

Return ONLY a JSON list of strings, e.g. ["fire hydrant", "traffic cone", "dog"]. Nothing else."""

NEG_VERIFY_TMPL = """You are validating HARD NEGATIVE prompts for an instance-segmentation dataset.

For the attached image, decide for EACH candidate phrase below whether that object type is GENUINELY ABSENT (not visible anywhere). Be strict: if even one instance is visible, or you are unsure, mark it NOT absent.

Candidates:
{cands}

Return ONLY a JSON object mapping each candidate phrase to true (absent) or false (present/unsure), e.g. {{"fire hydrant": true, "bicycle": false}}. Nothing else."""

# multi-instance (PCS) — a plain concept that must segment ALL of its instances
# PCS concept proposal. Representative SA-Co-style distribution: most concepts are plain
# noun phrases with 1-8 countable instances; multi-instance concepts are preferred (they
# teach "segment ALL of them") but clean single-instance concepts are allowed, matching
# the instance histogram of real PCS data (~40-60%% single).
MULTI_CONCEPT_TMPL = """You are building a promptable-concept segmentation dataset. Look at this image and name up to 3 CONCEPTS that are clearly visible with COUNTABLE, well-separated instances (1 to 8 of each), as plain lowercase noun phrases (e.g. "chairs", "parked cars", "window", "traffic cone").

Rules:
- STRONGLY prefer concepts with TWO OR MORE instances; include a single-instance concept only when nothing repeats cleanly.
- Every instance of the concept must be clearly visible and delineable — skip amorphous stuff (sky, grass, road surface) and skip concepts with partially hidden/uncountable instances.
- Order by instance count, highest first.

Return ONLY a JSON list, e.g. [{{"concept": "parked cars", "count": 4}}, {{"concept": "street lamp", "count": 1}}]. If nothing qualifies, return []."""

# Combined exhaustiveness check + quality score in ONE call: the overlay must cover ALL
# instances of the concept (missed instances = harmful false negatives in PCS training).
MULTI_CHECK_TMPL = """The {n} highlighted region(s) in this image are claimed to be ALL instances of "{concept}".

Check BOTH, strictly:
(a) CORRECT — every highlighted region genuinely is a "{concept}" (no wrong objects included);
(b) EXHAUSTIVE — no clearly visible "{concept}" is left unhighlighted. Count only distinct, clearly visible instances.

Score 1-10: 9-10 exact and exhaustive; 7-8 minor imperfection (one borderline partial/duplicate); 4-6 one clear miss or one wrong region; 1-3 multiple wrong or missed.

Return ONLY JSON: {{"all_correct": true/false, "exhaustive": true/false, "score": <int 1-10>, "reason": "<one sentence>"}}"""


# --------------------------------------------------------------------------------------
# Gemini call (urllib; no extra deps) + JSON parsing
# --------------------------------------------------------------------------------------
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def get_api_key(cli_key=None):
    key = cli_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it (the newsam pipeline used a shared key) "
            "or pass --api-key."
        )
    return key


# Gate execution mode (see --gate-mode): "cascade" = score-first + ordered short-circuit
# gates (keep-identical to "full", ~37% fewer calls); "full" = original always-all-gates.
GATE_MODE = "cascade"
GATE_MIN_SCORE = 1     # cascade skips gates when score < this (set from --min-score)
NO_PREVIEW = False     # set from --no-preview; suppresses caption-PNG side outputs


def LLM(model, parts, api_key, max_retries=6, temperature=0.4):
    return call_gemini(model, parts, api_key, max_retries=max_retries, temperature=temperature)


def call_gemini(model, parts, api_key, max_retries=6, temperature=0.4):
    """POST a generateContent request; retry on rate-limit/5xx with exponential backoff."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    data = json.dumps(body).encode()
    delay = 4.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
            cand = payload["candidates"][0]
            return "".join(p.get("text", "") for p in cand["content"]["parts"])
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read().decode(errors="replace")[:200]
            if code in (429, 500, 503) and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                log(f"    [retry] HTTP {code}; waiting {wait:.0f}s ({detail})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            raise
    raise RuntimeError("exhausted retries")


def parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def png_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_part(b64):
    return {"inline_data": {"mime_type": "image/png", "data": b64}}


# --------------------------------------------------------------------------------------
# Image / mask helpers
# --------------------------------------------------------------------------------------
def border_sides(mask, thresh=0.12):
    m = np.asarray(mask, dtype=bool)
    edges = (m[0, :], m[-1, :], m[:, 0], m[:, -1])
    return sum(1 for e in edges if e.mean() > thresh)


def iou(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = a.sum() + b.sum() - inter
    return float(inter / union) if union else 0.0


def edge_band(mask, thickness):
    """Boolean band of pixels within `thickness` of the mask boundary (for outlines)."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m)
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            shifted = np.zeros_like(m)
            ys = slice(max(0, dy), m.shape[0] + min(0, dy))
            xs = slice(max(0, dx), m.shape[1] + min(0, dx))
            ys2 = slice(max(0, -dy), m.shape[0] + min(0, -dy))
            xs2 = slice(max(0, -dx), m.shape[1] + min(0, -dx))
            shifted[ys, xs] = m[ys2, xs2]
            out |= shifted
    return out & ~m


def highlight(orig, mask, alpha=0.0):
    """Lime outline (+ optional translucent fill) around the mask. alpha=0 => outline only."""
    arr = np.asarray(orig.convert("RGB")).astype(np.float32)
    m = np.asarray(mask, dtype=bool)
    if alpha > 0:
        arr[m] = (1 - alpha) * arr[m] + alpha * np.array(HILITE, np.float32)
    arr[edge_band(mask, 5)] = (0, 0, 0)
    arr[edge_band(mask, 3)] = HILITE
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


def number_image(orig, masks):
    """Draw a numbered white badge at the centroid of each candidate mask, with a legible
    centered digit (default PIL font is near-invisible at full res, so use a truetype font)."""
    img = orig.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    r = max(16, img.width // 45)
    font = _font(int(r * 1.4))
    for i, m in enumerate(masks, 1):
        ys, xs = np.where(np.asarray(m, dtype=bool))
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        s = str(i)
        tb = d.textbbox((0, 0), s, font=font)
        d.text((cx - (tb[2] - tb[0]) / 2 - tb[0], cy - (tb[3] - tb[1]) / 2 - tb[1]),
               s, fill=(0, 0, 0), font=font)
    return img


_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_w):
    lines, cur = [], ""
    for w in str(text).split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [str(text)]


def render_caption(img, lines):
    """Append a WHITE strip at the bottom of `img` with BLACK wrapped text (one block
    per entry in `lines`). Returns a new image."""
    W, H = img.size
    fs = max(15, W // 48)
    font = _font(fs)
    pad = int(fs * 0.7)
    line_h = fs + int(fs * 0.35)
    probe = ImageDraw.Draw(img)
    wrapped = []
    for ln in lines:
        wrapped += _wrap_text(probe, ln, font, W - 2 * pad)
    strip_h = 2 * pad + line_h * len(wrapped)
    canvas = Image.new("RGB", (W, H + strip_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = H + pad
    for ln in wrapped:
        d.text((pad, y), ln, font=font, fill=(0, 0, 0))
        y += line_h
    return canvas


# Per-sample quality score (1-10), used by every generate/auto/rescore path. Count, necessity
# and uniqueness in ONE call; forced instance enumeration (STEP 1) de-anchors the count so the
# model can't invent instances to justify over-specifying a one-of-a-kind object.
SCORE_SINGLE_TMPL = """You are scoring a referring-expression annotation for object segmentation. ONE object in this image is outlined in bright green (the outline is only a marker — judge the object underneath, never treat green as its colour).

The outlined object is a "{obj}". The proposed referring expression is:
    "{prompt}"

Do BOTH checks, then score:
STEP 1 (COUNT) — list every "{obj}" of comparable prominence you can actually see, each with a short location tag; do NOT invent instances. This is the count of the kind.
STEP 2 (NECESSITY) — if the count is 1, the plain name already identifies it, so ANY extra descriptive/positional/functional/affordance wording is UNNECESSARY -> score AT MOST 3.
STEP 3 (UNIQUENESS) — if the count is >=2, check EACH instance and mark whether the FULL expression's distinguishing detail fits it. If TWO OR MORE fit (shared detail, e.g. "the sign showing medical info" when several are medical ads), it is AMBIGUOUS -> score AT MOST 3. If EXACTLY ONE fits -> 7-10 (4-6 if awkward/wordy).
A 7-10 requires EITHER (count 1 AND just the plain name) OR (count >=2 AND exactly one fits).

Return ONLY JSON: {{"count_of_kind": <int>, "n_fitting_full_expression": <int>, "score": <int 1-10>, "justification": "<one sentence>"}}"""


def score_single(model, img_b64, prompt, obj, api_key):
    """Single-call variant: count + necessity + uniqueness in ONE prompt (re-anchored)."""
    obj = obj or "object"
    try:
        sj = parse_json(LLM(
            model, [{"text": SCORE_SINGLE_TMPL.format(prompt=prompt, obj=obj)},
                    img_part(img_b64)], api_key, temperature=0.0))
        score = max(1, min(10, int(sj.get("score"))))
        return score, f"[{sj.get('count_of_kind')} of kind] {sj.get('justification')}"
    except Exception as e:
        return None, f"score error: {e}"


_FONT_CACHE = {}


def _load_font(size):
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            f = ImageFont.truetype(p, size)
            break
        except Exception:
            f = None
    f = f or ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def with_caption(img, prompt, sub="", score=None):
    """Append a WHITE strip under the image with BLACK text: the score + referring
    expression (+ optional smaller subtitle), word-wrapped. Returns a taller RGB image."""
    W = img.width
    pad = max(10, W // 60)
    fsize = max(15, W // 42)
    ssize = max(12, int(fsize * 0.72))
    font, sfont = _load_font(fsize), _load_font(ssize)
    meas = ImageDraw.Draw(img)

    def wrap(text, f):
        lines, cur = [], ""
        for word in (text or "").split():
            trial = (cur + " " + word).strip()
            if not cur or meas.textlength(trial, font=f) <= W - 2 * pad:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    head = f"[{score}/10]  " if score is not None else "[?/10]  "
    plines = wrap(head + (prompt or ""), font)
    slines = wrap(sub, sfont) if sub else []
    lh, slh = fsize + 6, ssize + 4
    strip_h = pad * 2 + lh * len(plines) + (pad // 2 + slh * len(slines) if slines else 0)
    out = Image.new("RGB", (W, img.height + strip_h), (255, 255, 255))  # white strip
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    y = img.height + pad
    for ln in plines:
        d.text((pad, y), ln, fill=(0, 0, 0), font=font)  # black text
        y += lh
    if slines:
        y += pad // 2
        for ln in slines:
            d.text((pad, y), ln, fill=(70, 70, 70), font=sfont)
            y += slh
    return out


def encode_mask_record(mask):
    """COCO-compatible compressed RLE + bbox + area."""
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    bbox = [round(float(v), 4) for v in mask_util.toBbox(rle).tolist()]
    area = float(mask_util.area(rle))
    return {"size": [int(v) for v in rle["size"]], "counts": counts}, bbox, round(area, 4)


def best_overlap(click_mask, text_results):
    cm = np.asarray(click_mask, dtype=bool)
    best = None
    for m, s in text_results:
        i = iou(cm, m)
        if best is None or i > best[0]:
            best = (i, np.asarray(m, dtype=bool), s)
    return best


# --------------------------------------------------------------------------------------
# SAM3 segmenter (interactive click + text/concept), thread-safe
# --------------------------------------------------------------------------------------
class Segmenter:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint
        self.device = device
        self._cuda = device.startswith("cuda")
        self._idx = int(device.split(":", 1)[1]) if ":" in device else 0
        self.model = None
        self.processor = None
        self.lock = threading.Lock()

    def _dev_ctx(self):
        import torch
        return torch.cuda.device(self._idx) if self._cuda else contextlib.nullcontext()

    def _autocast(self):
        import torch
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self._cuda else contextlib.nullcontext())

    def load(self):
        if self.model is not None:
            return
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        # The builder only relocates weights when device is exactly "cuda" (it calls
        # model.cuda(), which honours the *current* device). So build under this
        # segmenter's device context and pass the generic string; per-op _dev_ctx then
        # keeps every subsequently-created tensor on the same GPU.
        dev = "cuda" if self._cuda else "cpu"
        with self._dev_ctx():
            self.model = build_sam3_image_model(
                checkpoint_path=self.checkpoint, load_from_HF=False,
                device=dev, enable_inst_interactivity=True,
            )
            self.processor = Sam3Processor(self.model, device=dev,
                                           confidence_threshold=TEXT_THRESHOLD)

    def set_image(self, image):
        with self.lock, self._dev_ctx(), self._autocast():
            self.load()
            return self.processor.set_image(image)

    def click(self, state, points, labels):
        with self.lock, self._dev_ctx(), self._autocast():
            masks, scores, _ = self.model.predict_inst(
                state, point_coords=np.asarray(points), point_labels=np.asarray(labels),
                multimask_output=True,
            )
        i = int(np.argmax(scores))
        return (masks[i] > 0.5), float(scores[i])

    def segment_text(self, state, prompt):
        with self.lock, self._dev_ctx(), self._autocast():
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt, state)
        masks_t, scores_t = state.get("masks"), state.get("scores")
        if masks_t is None or scores_t is None or len(scores_t) == 0:
            return []
        masks = masks_t.squeeze(1).detach().cpu().bool().numpy()
        scores = scores_t.detach().float().cpu().numpy().tolist()
        order = np.argsort(scores)[::-1]
        return [(masks[i], float(scores[i])) for i in order]


def pick_candidates(seg, state, W, H, seed):
    """Auto-click distinct, clean, sensibly-sized objects. -> [(mask, score, (x,y)), ...]."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(N_CANDIDATES):
        x = int(rng.uniform(0.06, 0.94) * W)
        y = int(rng.uniform(0.06, 0.94) * H)
        mask, score = seg.click(state, [[x, y]], [1])
        area = float(mask.mean())
        if border_sides(mask) >= 2:
            continue
        if not (MIN_AREA <= area <= MAX_AREA and score >= MIN_SCORE):
            continue
        dup = next((i for i, (_, m, _) in enumerate(found) if iou(m, mask) > DEDUP_IOU), None)
        if dup is None:
            found.append((score, mask, (x, y)))
        elif score > found[dup][0]:
            found[dup] = (score, mask, (x, y))
    found.sort(key=lambda t: t[0], reverse=True)
    return [(m, s, p) for s, m, p in found[:N_SELECT]]


# --------------------------------------------------------------------------------------
# Verification gates
# --------------------------------------------------------------------------------------
_ARTICLES = {"the", "a", "an"}


def _content_words(s):
    return [t for t in re.findall(r"[a-z]+", (s or "").lower()) if t not in _ARTICLES]


def _stem(t):
    return t[:-1] if t.endswith("s") and len(t) > 3 else t


def complexity_gate(prompt, obj, simple):
    """Programmatic: the expression must add detail beyond the plain object category."""
    category = {_stem(t) for t in _content_words(obj) + _content_words(simple)}
    extra = [t for t in _content_words(prompt) if _stem(t) not in category]
    if not extra:
        return "fail", f'non-complex: prompt is just the plain object name ("{prompt}")'
    return "pass", f"adds detail: {' '.join(extra)}"


def _parse_gate(txt):
    try:
        vj = parse_json(txt)
        verdict = "pass" if str(vj.get("verdict", "")).strip().lower() == "pass" else "fail"
        return verdict, vj.get("reason"), vj
    except Exception as e:
        return "fail", f"gate parse error: {e}", {}


def _run_gate(model, prompt_text, img_b64, api_key):
    try:
        txt = LLM(model, [{"text": prompt_text}, img_part(img_b64)], api_key,
                          temperature=0.0)
    except Exception as e:
        return "fail", f"gate error: {e}", {}
    return _parse_gate(txt)


def verify(model, img_b64, prompt, obj, simple, cat, api_key):
    """1 programmatic + 4 Gemini gates. Overall PASS iff all pass."""
    gates = {}
    cj, cr = complexity_gate(prompt, obj, simple)
    gates["complexity"] = {"verdict": cj, "reason": cr}

    specs = [
        ("grounding", VERIFY_GROUNDING_TMPL.format(object=obj or "object")),
        ("distinct", VERIFY_DISTINCT_TMPL),
        ("unique", VERIFY_UNIQUE_TMPL.format(prompt=prompt)),
        ("category", VERIFY_CATEGORY_TMPL.format(prompt=prompt, cat_name=cat["name"],
                                                 cat_def=cat["definition"])),
    ]
    for name, tmpl in specs:
        v, reason, raw = _run_gate(model, tmpl, img_b64, api_key)
        gates[name] = {"verdict": v, "reason": reason}
        if name == "grounding":
            gates[name]["outlined_thing"] = raw.get("outlined_thing")

    failed = [n for n, g in gates.items() if g["verdict"] != "pass"]
    verdict = "fail" if failed else "pass"
    combined = ("; ".join(f"{n}: {gates[n]['reason']}" for n in failed)
                if failed else "all gates pass")
    return verdict, combined, gates


# Gate order for cascade mode: highest rejection power first (measured on a 100-sample
# bench: distinct fails 32%, unique 32%, category 18%, grounding 13%). Short-circuiting at
# the first fail gives verdicts IDENTICAL to verify() at ~2.6 instead of 4.0 calls.
CASCADE_ORDER = ("distinct", "unique", "category", "grounding")


def verify_cascade(model, img_b64, prompt, obj, simple, cat, api_key):
    """Same gates/prompts as verify(), but ordered by rejection power and stopped at the
    first fail. Temp-0 gates are deterministic, so kept verdicts match verify() exactly;
    only the per-gate diagnostics for gates after the first fail are omitted."""
    gates = {}
    cj, cr = complexity_gate(prompt, obj, simple)
    gates["complexity"] = {"verdict": cj, "reason": cr}
    if cj != "pass":
        return "fail", f"complexity: {cr}", gates

    tmpls = {
        "grounding": VERIFY_GROUNDING_TMPL.format(object=obj or "object"),
        "distinct": VERIFY_DISTINCT_TMPL,
        "unique": VERIFY_UNIQUE_TMPL.format(prompt=prompt),
        "category": VERIFY_CATEGORY_TMPL.format(prompt=prompt, cat_name=cat["name"],
                                                cat_def=cat["definition"]),
    }
    for name in CASCADE_ORDER:
        v, reason, raw = _run_gate(model, tmpls[name], img_b64, api_key)
        gates[name] = {"verdict": v, "reason": reason}
        if name == "grounding":
            gates[name]["outlined_thing"] = raw.get("outlined_thing")
        if v != "pass":
            return "fail", f"{name}: {reason}", gates
    return "pass", "all gates pass", gates


# --------------------------------------------------------------------------------------
# Stage: generate positives
# --------------------------------------------------------------------------------------
def gemini_select(model, numbered_img, n, cat, api_key):
    txt = LLM(model,
                      [{"text": SELECT_PROMPT_TMPL.format(
                          n=n, cat_name=cat["name"], cat_def=cat["definition"],
                          cat_examples="; ".join(cat["examples"]))},
                       img_part(png_b64(numbered_img))], api_key, temperature=0.2)
    try:
        sj = parse_json(txt)
        choice = int(sj.get("choice"))
    except (TypeError, ValueError, Exception):
        return None, None
    if not (1 <= choice <= n):
        return None, sj.get("reasoning") if isinstance(sj, dict) else None
    return choice - 1, sj.get("reasoning")


def _cat_block(cats):
    return "\n".join(
        f"- {c['id']} ({c['name']}): {c['definition']} e.g. {'; '.join(c['examples'][:3])}"
        for c in cats)


def gemini_select_auto(model, numbered_img, n, cats, api_key):
    """Let Gemini pick BOTH the best object and the best-fitting reasoning type (from `cats`)."""
    txt = LLM(model,
                      [{"text": SELECT_AUTO_PROMPT_TMPL.format(
                          n=n, n_cats=len(cats), cat_block=_cat_block(cats),
                          cat_ids=", ".join(c["id"] for c in cats))},
                       img_part(png_b64(numbered_img))], api_key, temperature=0.3)
    try:
        sj = parse_json(txt)
        choice = int(sj.get("choice"))
        cid = str(sj.get("category", "")).strip()
    except Exception:
        return None, None, None
    reason = sj.get("reasoning") if isinstance(sj, dict) else None
    if cid not in CATEGORIES_BY_ID or not (1 <= choice <= n):
        return None, None, reason
    return choice - 1, cid, reason


def process_one_auto(src, idx, total, cats_allowed, seg, model, api_key):
    """Like process_one, but Gemini itself chooses the reasoning category (from cats_allowed)
    together with the object. Returns (record, preview_image) or None."""
    tag = f"[{idx}/{total}] {src.name} <auto>"
    seed = int("".join(ch for ch in src.stem if ch.isdigit()) or "0") % (2 ** 32)
    orig = Image.open(src).convert("RGB")
    W, H = orig.size

    state = None
    try:
        try:
            state = seg.set_image(orig)
            cands = pick_candidates(seg, state, W, H, seed)
        except Exception as e:
            log(f"{tag}  SAM3 error: {e}")
            return None
        if not cands:
            log(f"{tag}  no clean object, skip")
            # record the skip so resume runs never re-grind this image (SAM3 finds no
            # candidates deterministically; without a record it re-queues forever)
            return {"file_name": src.name, "reasoning_type": "none",
                    "verdict": "skip_nocand"}, None

        # 1) let Gemini pick BOTH the object and the reasoning category
        numbered = number_image(orig, [c[0] for c in cands])
        try:
            sel_idx, cid, sel_reason = gemini_select_auto(
                model, numbered, len(cands), cats_allowed, api_key)
        except Exception as e:
            log(f"{tag}  select error: {e}")
            return None
        if cid is None:
            log(f"{tag}  no valid (object,category) selection, skip")
            return None
        cat = CATEGORIES_BY_ID[cid]
        chosen_idx = sel_idx
        mask, mscore, point = cands[chosen_idx]

        # 2) describe with the chosen category's prompt (outline-only view)
        shown = highlight(orig, mask, alpha=0.0)
        try:
            gen = parse_json(LLM(
                model,
                [{"text": DESCRIBE_PROMPT_TMPL.format(
                    cat_name=cat["name"], cat_def=cat["definition"],
                    cat_examples="; ".join(cat["examples"]))},
                 img_part(png_b64(shown))], api_key, temperature=0.5))
        except Exception as e:
            log(f"{tag}  describe error: {e}")
            return None
        if not gen.get("possible", True) or not (gen.get("prompt") or "").strip():
            log(f"{tag}  <{cid}> not possible: {gen.get('reason', '')[:60]}")
            # record so the image isn't re-attempted on resume (Gemini judged the chosen
            # object/category infeasible; a rerun would almost certainly repeat that)
            return {"file_name": src.name, "reasoning_type": cid,
                    "verdict": "skip_infeasible"}, None
        prompt = gen["prompt"].strip()
        simple = (gen.get("simple_prompt") or "").strip()
        obj = gen.get("object")
        reasoning = gen.get("reasoning")

        # 3) refine mask: concept-segment with the simple prompt, keep best overlap
        final_mask, source, ov = mask, "click", 0.0
        if simple:
            try:
                text_results = seg.segment_text(state, simple)
            except Exception as e:
                log(f"{tag}  text-seg error: {e}")
                text_results = []
            sel = best_overlap(mask, text_results)
            if sel and sel[0] >= MIN_OVERLAP_IOU:
                final_mask, source, ov = sel[1], "text", round(sel[0], 4)
    finally:
        state = None  # free encoder state before the (slower) verify phase

    # 4+5) score FIRST, then gates (cascade mode). Keep decision = gates-pass AND
    # score>=min, an AND — so ordering cannot change which samples are kept, but running
    # the cheap score first lets cascade mode skip ALL gate calls for low scorers, and the
    # ordered short-circuit cuts gate calls on the rest (bench: 7.0 -> ~4.4 calls/sample,
    # keep decisions identical).
    verify_b64 = png_b64(highlight(orig, final_mask, alpha=0.0))
    score, score_just = score_single(model, verify_b64, prompt, obj, api_key)
    if GATE_MODE == "cascade" and (score or 0) < GATE_MIN_SCORE:
        verdict, vreason, gates = "lowscore", f"score {score} < {GATE_MIN_SCORE}; gates skipped", {}
    elif GATE_MODE == "cascade":
        verdict, vreason, gates = verify_cascade(model, verify_b64, prompt, obj, simple, cat, api_key)
    else:
        verdict, vreason, gates = verify(model, verify_b64, prompt, obj, simple, cat, api_key)

    # preview: mask overlay + black-on-white caption strip (score + prompt + category)
    preview = with_caption(highlight(orig, final_mask, alpha=0.35), prompt,
                           sub=f"{cat['name']}  ·  {cid}  ·  {verdict.upper()}\n{score_just or ''}",
                           score=score)

    mask_rle, bbox, mask_area = encode_mask_record(final_mask)
    rec = {
        "file_name": src.name, "kind": "complex", "reasoning_type": cat["id"],
        "object": obj, "prompt": prompt, "simple_prompt": simple, "reasoning": reasoning,
        "verdict": verdict, "verify_reason": vreason, "verify_gates": gates,
        "score": score, "score_justification": score_just,
        "click_point": [point[0], point[1]], "click_mask_score": round(mscore, 4),
        "mask_source": source, "overlap_iou": ov, "n_candidates": len(cands),
        "chosen_idx": chosen_idx, "select_reason": sel_reason,
        "area_frac": round(float(final_mask.mean()), 4),
        "instances": [{"mask_rle": mask_rle, "bbox": bbox, "mask_area": mask_area}],
        "model": model, "auto_selected": True,
    }
    log(f"{tag}  <{cid}> src={source} ov={ov}  [{verdict.upper()}]  score={score}  | {prompt[:48]}")
    return rec, preview


def process_one(src, idx, total, cat, seg, model, api_key, out_dir):
    tag = f"[{idx}/{total}] {src.name} <{cat['id']}>"
    seed = int("".join(ch for ch in src.stem if ch.isdigit()) or "0") % (2 ** 32)
    orig = Image.open(src).convert("RGB")
    W, H = orig.size

    state = None
    try:
        try:
            state = seg.set_image(orig)
            cands = pick_candidates(seg, state, W, H, seed)
        except Exception as e:
            log(f"{tag}  SAM3 error: {e}")
            return None
        if not cands:
            log(f"{tag}  no clean object, skip")
            return None

        # 1) let Gemini pick the object best suited to this reasoning category
        sel_idx, sel_reason = None, None
        if len(cands) > 1:
            numbered = number_image(orig, [c[0] for c in cands])
            try:
                sel_idx, sel_reason = gemini_select(model, numbered, len(cands), cat, api_key)
            except Exception as e:
                log(f"{tag}  select error: {e}")
        chosen_idx = sel_idx if sel_idx is not None else 0
        mask, mscore, point = cands[chosen_idx]

        # 2) describe with the category-specific prompt (outline-only view)
        shown = highlight(orig, mask, alpha=0.0)
        try:
            gen = parse_json(LLM(
                model,
                [{"text": DESCRIBE_PROMPT_TMPL.format(
                    cat_name=cat["name"], cat_def=cat["definition"],
                    cat_examples="; ".join(cat["examples"]))},
                 img_part(png_b64(shown))], api_key, temperature=0.5))
        except Exception as e:
            log(f"{tag}  describe error: {e}")
            return None
        if not gen.get("possible", True) or not (gen.get("prompt") or "").strip():
            log(f"{tag}  category not possible: {gen.get('reason', '')[:60]}")
            return None
        prompt = gen["prompt"].strip()
        simple = (gen.get("simple_prompt") or "").strip()
        obj = gen.get("object")
        reasoning = gen.get("reasoning")

        # 3) refine mask: concept-segment with the simple prompt, keep best overlap
        final_mask, source, ov = mask, "click", 0.0
        if simple:
            try:
                text_results = seg.segment_text(state, simple)
            except Exception as e:
                log(f"{tag}  text-seg error: {e}")
                text_results = []
            sel = best_overlap(mask, text_results)
            if sel and sel[0] >= MIN_OVERLAP_IOU:
                final_mask, source, ov = sel[1], "text", round(sel[0], 4)
    finally:
        state = None  # free encoder state before the (slower) verify phase

    # 4) verify
    verify_b64 = png_b64(highlight(orig, final_mask, alpha=0.0))
    verdict, vreason, gates = verify(model, verify_b64, prompt, obj, simple, cat, api_key)

    # 5) score 1-10 (quality incl. whether the extra specification is actually needed)
    score, score_just = score_single(model, verify_b64, prompt, obj, api_key)

    # save a preview PNG with a black-on-white caption (score + prompt) under out/<cat>/<verdict>/
    render_dir = out_dir / cat["id"] / verdict
    render_dir.mkdir(parents=True, exist_ok=True)
    caption = [f"[{score if score is not None else '?'}/10]  {prompt}",
               f"{cat['name']} · {verdict.upper()}" + (f" · {score_just}" if score_just else "")]
    render_caption(highlight(orig, final_mask, alpha=0.35), caption).save(
        render_dir / (src.stem + ".png"))

    mask_rle, bbox, mask_area = encode_mask_record(final_mask)
    rec = {
        "file_name": src.name, "kind": "complex", "reasoning_type": cat["id"],
        "object": obj, "prompt": prompt, "simple_prompt": simple, "reasoning": reasoning,
        "verdict": verdict, "verify_reason": vreason, "verify_gates": gates,
        "score": score, "score_justification": score_just,
        "click_point": [point[0], point[1]], "click_mask_score": round(mscore, 4),
        "mask_source": source, "overlap_iou": ov, "n_candidates": len(cands),
        "chosen_idx": chosen_idx, "select_reason": sel_reason,
        "area_frac": round(float(final_mask.mean()), 4),
        "instances": [{"mask_rle": mask_rle, "bbox": bbox, "mask_area": mask_area}],
        "model": model,
    }
    log(f"{tag}  src={source} ov={ov}  [{verdict.upper()}]  score={score}  | {prompt[:48]}")
    return rec


def process_one_multi(src, idx, total, seg, model, api_key, out_dir):
    """PCS sample: a plain concept -> ALL its instance masks (1+ instances, multi preferred).

    One concept-proposal call, then per concept: SAM3 concept-segmentation and a single
    combined check call (correct + exhaustive + 1-10 score). Deterministic dead-ends return
    skip records so resume runs never re-grind them."""
    tag = f"[{idx}/{total}] {src.name} <multi>"
    orig = Image.open(src).convert("RGB")
    try:
        state = seg.set_image(orig)
    except Exception as e:
        log(f"{tag}  SAM3 error: {e}")
        return None                      # transient: retry on next run
    try:
        raw = parse_json(LLM(
            model, [{"text": MULTI_CONCEPT_TMPL}, img_part(png_b64(orig))], api_key,
            temperature=0.4))
        concepts = []
        for c in raw:
            name = (c.get("concept") if isinstance(c, dict) else str(c) or "").strip().lower()
            if name:
                concepts.append(name)
    except Exception:
        concepts = []
    if not concepts:
        log(f"{tag}  no usable concept, skip")
        return {"file_name": src.name, "reasoning_type": "multi_instance",
                "verdict": "skip_noconcept"}

    best_fail = None
    for concept in concepts[:3]:
        try:
            text_results = seg.segment_text(state, concept)
        except Exception as e:
            log(f"{tag}  text-seg error: {e}")
            continue
        # keep confident, non-background, deduped instances
        insts = []
        for m, s in text_results:
            if s < TEXT_THRESHOLD or border_sides(m) >= 3:
                continue
            if any(iou(m, e) > DEDUP_IOU for e in insts):
                continue
            insts.append(np.asarray(m, dtype=bool))
        if not insts:
            continue
        prev = orig
        for m in insts:
            prev = highlight(prev, m, alpha=0.3)
        # combined check: every region is the concept AND no instance missed, + 1-10 score
        try:
            vj = parse_json(LLM(
                model, [{"text": MULTI_CHECK_TMPL.format(concept=concept, n=len(insts))},
                        img_part(png_b64(prev))], api_key, temperature=0.0))
            ok = bool(vj.get("all_correct")) and bool(vj.get("exhaustive"))
            score = max(1, min(10, int(vj.get("score"))))
        except Exception:
            ok, score, vj = False, None, {}
        verdict = "pass" if ok else "fail"
        if not NO_PREVIEW:
            render_dir = out_dir / "multi_instance" / verdict
            render_dir.mkdir(parents=True, exist_ok=True)
            caption = [f"[{score if score is not None else '?'}/10]  {concept}  (x{len(insts)})",
                       f"multi-instance · {verdict.upper()} · {vj.get('reason') or ''}"]
            render_caption(prev, caption).save(render_dir / (src.stem + ".png"))
        if not ok:
            best_fail = vj.get("reason")
            continue                     # try the next concept
        instances = []
        for m in insts:
            rle, bbox, area = encode_mask_record(m)
            instances.append({"mask_rle": rle, "bbox": bbox, "mask_area": area})
        log(f"{tag}  '{concept}' x{len(insts)}  score={score}  [PASS]")
        return {
            "file_name": src.name, "kind": "multi", "reasoning_type": "multi_instance",
            "object": concept, "prompt": concept, "simple_prompt": concept,
            "reasoning": vj.get("reason"), "verdict": "pass",
            "verify_gates": {"multi_check": vj}, "score": score,
            "score_justification": vj.get("reason"), "n_instances": len(insts),
            "instances": instances, "model": model,
        }
    log(f"{tag}  no passing concept ({str(best_fail)[:50]})")
    return {"file_name": src.name, "reasoning_type": "multi_instance",
            "verdict": "skip_multi_fail"}


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, records, key=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        records = sorted(records, key=key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def stage_generate(args):
    import torch
    global NO_PREVIEW
    NO_PREVIEW = bool(getattr(args, "no_preview", False))
    api_key = get_api_key(args.api_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    results_path = out_dir / "results.jsonl"
    mode = getattr(args, "mode", "complex")
    target = getattr(args, "target_pass", None)   # stop once this many kept (score>=min)
    min_score = getattr(args, "min_score", 1)

    images = sorted(image_dir.glob("*.jpg"))
    random.Random(args.seed).shuffle(images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No .jpg images in {image_dir} (run the `download` stage first)")

    existing = load_jsonl(results_path)
    done = {(r["file_name"], r["reasoning_type"]) for r in existing}

    # Build the job list. key = (file_name, reasoning_type) so re-runs skip attempted pairs.
    #   complex: each image assigned ONE reasoning category, round-robin (balanced input)
    #   multi:   each image gets one multi-instance attempt (reasoning_type "multi_instance")
    if mode == "multi":
        jobs = [(src, None) for src in images if (src.name, "multi_instance") not in done]
        label = "multi-instance"
    else:
        cats = ([CATEGORIES_BY_ID[c] for c in args.categories] if args.categories else CATEGORIES)
        jobs = []
        for i, src in enumerate(images):
            cat = cats[i % len(cats)]
            if (src.name, cat["id"]) not in done:
                jobs.append((src, cat))
        label = "complex"
    if not jobs:
        log(f"Nothing to generate for mode={mode} (all attempted).")
        return

    # shard SAM3 across every visible GPU (as in stage_auto): more cards = more in-flight
    if device == "cuda":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cuda:0"]
    else:
        devices = ["cpu"]
    segs = [Segmenter(args.checkpoint, d) for d in devices]
    for s in segs:
        s.set_image(Image.open(jobs[0][0]).convert("RGB"))  # warm up serialized
    kept = sum(1 for r in existing if r.get("verdict") == "pass"
               and (r.get("score") or 0) >= min_score)
    log(f"generate[{label}]: {len(jobs)} jobs over {len(images)} images "
        f"[{'+'.join(devices)}, {GEMINI_MODEL}, {args.workers} workers] "
        f"kept={kept} target={target} -> {results_path}")

    records = list(existing)
    total = len(jobs)
    keyf = lambda r: (r["file_name"], r["reasoning_type"])
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while processed < len(jobs) and (target is None or kept < target):
            batch = jobs[processed: processed + args.workers * 2]
            if mode == "multi":
                futs = {ex.submit(process_one_multi, src, processed + j + 1, total,
                                  segs[j % len(segs)], GEMINI_MODEL, api_key, out_dir): src
                        for j, (src, _) in enumerate(batch)}
            else:
                futs = {ex.submit(process_one, src, processed + j + 1, total, cat,
                                  segs[j % len(segs)], GEMINI_MODEL, api_key, out_dir): src
                        for j, (src, cat) in enumerate(batch)}
            processed += len(batch)
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    log(f"FAILED {src.name}: {e}")
                    continue
                if rec:
                    records.append(rec)
                    if rec.get("verdict") == "pass" and (rec.get("score") or 0) >= min_score:
                        kept += 1
            write_jsonl(results_path, records, key=keyf)
            log(f"  progress {processed}/{total}  kept(score>={min_score})={kept}"
                + (f"/{target}" if target else ""))
    write_jsonl(results_path, records, key=keyf)
    n_pass = sum(1 for r in records if r["verdict"] == "pass")
    by_cat = collections.Counter(r["reasoning_type"] for r in records if r["verdict"] == "pass")
    log(f"generate[{label}] DONE: {len(records)} records, {n_pass} PASS, kept={kept}. "
        f"per-type PASS: {json.dumps(dict(by_cat))}")


# --------------------------------------------------------------------------------------
# Stage: auto — Gemini chooses the category; fill a per-category quota of PASS samples
# --------------------------------------------------------------------------------------
def stage_auto(args):
    """Instead of assigning a target category per image, Gemini is shown all 8 categories and
    picks the object + category that best fits each scene. Collects PASS samples until every
    category has `--per-cat` of them, saving each as a preview (mask overlay + caption strip)
    under out/samples_auto/<category>/. No negatives, no multi-instance."""
    import torch
    api_key = get_api_key(args.api_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    target = args.per_cat
    results_path = out_dir / "results_auto.jsonl"
    sample_dir = out_dir            # category folders live directly under out/

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No .jpg images in {image_dir} (run `download` or point --image-dir)")
    random.Random(args.seed).shuffle(images)  # varied scenes, deterministic order
    if args.limit:
        images = images[: args.limit]

    # resume: keep prior records, seed per-category PASS counts, skip already-attempted images
    records = load_jsonl(results_path)
    min_score = args.min_score
    global GATE_MODE, GATE_MIN_SCORE
    GATE_MODE = args.gate_mode
    GATE_MIN_SCORE = min_score
    counts = collections.Counter(r["reasoning_type"] for r in records
                                 if r.get("verdict") == "pass"
                                 and (r.get("score") or 0) >= min_score)
    done_files = {r["file_name"] for r in records}
    images = [im for im in images if im.name not in done_files]

    keyf = lambda r: (r["file_name"], r["reasoning_type"])
    cat_line = lambda: json.dumps({c["id"]: counts[c["id"]] for c in CATEGORIES})

    # Shard SAM3 across every visible GPU: each in-flight image pins ~0.6 GB of encoder
    # state on its GPU for the duration of its (slow, network-bound) Gemini calls, so more
    # cards = more images can be in flight at once. Restrict cards with CUDA_VISIBLE_DEVICES.
    if device == "cuda":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cuda:0"]
    else:
        devices = ["cpu"]
    segs = [Segmenter(args.checkpoint, d) for d in devices]
    if images:
        for s in segs:
            s.set_image(Image.open(images[0]).convert("RGB"))  # warm up + load each GPU
    log(f"auto: target {target}/cat ({target * len(CATEGORIES)} total) over up to "
        f"{len(images)} images [{'+'.join(devices)}, {GEMINI_MODEL}, {args.workers} workers] "
        f"-> {sample_dir}")

    lock = threading.Lock()
    total = len(images)
    processed = 0
    t_start = time.time()
    first_hit = {}          # cat -> seconds until its first kept (score>=min) sample
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while processed < len(images):
            needed = [c for c in CATEGORIES if counts[c["id"]] < target]
            if not needed:
                break
            batch = images[processed: processed + args.workers * 2]
            futs = {}
            for j, src in enumerate(batch):
                seg = segs[j % len(segs)]  # round-robin across GPUs to spread state memory
                futs[ex.submit(process_one_auto, src, processed + j + 1, total, needed,
                               seg, GEMINI_MODEL, api_key)] = src
            processed += len(batch)
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    log(f"FAILED {src.name}: {e}")
                    continue
                if not res:
                    continue
                rec, preview = res
                records.append(rec)
                if rec["verdict"] == "pass" and (rec.get("score") or 0) >= min_score:
                    cid = rec["reasoning_type"]
                    keep = False
                    with lock:
                        if counts[cid] < target:
                            counts[cid] += 1
                            keep = True
                            first_hit.setdefault(cid, round(time.time() - t_start, 1))
                    if keep and not args.no_preview:
                        d = sample_dir / cid
                        d.mkdir(parents=True, exist_ok=True)
                        preview.save(d / (Path(rec["file_name"]).stem + ".png"))
            write_jsonl(results_path, records, key=keyf)
            log(f"  progress {processed}/{total}  PASS/cat: {cat_line()}")

    write_jsonl(results_path, records, key=keyf)
    filled = [c["id"] for c in CATEGORIES if counts[c["id"]] >= target]
    log(f"auto DONE: {len(filled)}/{len(CATEGORIES)} categories filled to {target} "
        f"(score>={min_score}). kept/cat: {cat_line()}  -> {sample_dir}")
    # --- timing report ---
    log(f"[timing] seconds to FIRST score>={min_score} sample per cat: {json.dumps(first_hit)}")
    log(f"[timing] total wall: {time.time()-t_start:.0f}s")


# --------------------------------------------------------------------------------------
# Stage: negatives
# --------------------------------------------------------------------------------------
def propose_negatives(model, img, present, k, api_key):
    txt = LLM(model,
                      [{"text": NEG_PROPOSE_TMPL.format(
                          k=k, present="\n".join(f"- {p}" for p in sorted(present)) or "- (none)")},
                       img_part(png_b64(img))], api_key, temperature=0.7)
    try:
        arr = parse_json(txt)
        return [str(p).strip().lower() for p in arr if str(p).strip()]
    except Exception:
        return []


def verify_absent(model, img, cands, api_key):
    if not cands:
        return {}
    txt = LLM(model,
                      [{"text": NEG_VERIFY_TMPL.format(cands="\n".join(f"- {c}" for c in cands))},
                       img_part(png_b64(img))], api_key, temperature=0.0)
    out = {}
    try:
        obj = parse_json(txt)
        for kk, vv in obj.items():
            out[str(kk).strip().lower()] = (bool(vv) is True)
    except Exception:
        pass
    return {c: out.get(c, False) for c in cands}  # unparsed -> not absent (conservative)


def kept_positives(out_dir, min_score=7):
    """All kept positives across BOTH tracks: complex (results_auto.jsonl, the `auto` stage)
    and multi-instance PCS (results.jsonl, the `generate --mode multi` stage)."""
    rows = []
    for fname in ("results_auto.jsonl", "results.jsonl"):
        for r in load_jsonl(Path(out_dir) / fname):
            if r.get("verdict") == "pass" and (r.get("score") or 0) >= min_score \
                    and r.get("instances"):
                rows.append(r)
    return rows


def stage_negatives(args):
    api_key = get_api_key(args.api_key)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    results = kept_positives(out_dir, args.min_score)
    if not results:
        raise SystemExit("No kept positives found; run generation first.")

    # positives present per image, from kept records (avoid making a positive a negative)
    present = collections.defaultdict(set)
    for r in results:
        for p in (r.get("simple_prompt"), r.get("object")):
            if p:
                present[r["file_name"]].add(str(p).strip().lower())

    files = sorted({r["file_name"] for r in results})
    if args.limit:
        files = files[: args.limit]

    sidecar_path = out_dir / "image_negatives.json"
    sidecar = json.load(open(sidecar_path)) if sidecar_path.exists() else {}
    # seed with the old sa1b_pcs 30-deep verified pools where available (free, pre-verified)
    old_pool = Path("/mnt/data0/ameen/train_data_sa1b_pcs/image_negatives.json")
    if old_pool.exists():
        for fn, negs in json.load(open(old_pool)).items():
            fn = os.path.basename(fn)
            if fn not in sidecar and negs:
                sidecar[fn] = [str(p).strip().lower() for p in negs]
    todo = [f for f in files if f not in sidecar]
    log(f"negatives: mining {len(todo)} images (k={args.neg_k}, cap={args.neg_cap}) "
        f"-> {sidecar_path}")

    def one(fn):
        try:
            img = Image.open(image_dir / fn).convert("RGB")
        except Exception as e:
            return fn, None
        cands = propose_negatives(GEMINI_MODEL, img, present.get(fn, set()), args.neg_k, api_key)
        # drop anything colliding with a positive on this image
        cands = [c for c in dict.fromkeys(cands) if c not in present.get(fn, set())]
        verdicts = verify_absent(GEMINI_MODEL, img, cands, api_key)
        kept = [c for c in cands if verdicts.get(c)]
        random.Random(args.seed).shuffle(kept)
        return fn, kept[: args.neg_cap]

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fn, kept in ex.map(one, todo):
            if kept is None:
                continue
            sidecar[fn] = kept
            n += 1
            if n % 25 == 0:
                json.dump(sidecar, open(sidecar_path, "w"))
                log(f"negatives: {n}/{len(todo)}")
    json.dump(sidecar, open(sidecar_path, "w"))
    counts = [len(v) for v in sidecar.values()]
    log(f"negatives DONE: {len(sidecar)} images, mean={np.mean(counts):.2f} negs/img "
        f"-> {sidecar_path}")


# --------------------------------------------------------------------------------------
# Stage: export COCO
# --------------------------------------------------------------------------------------
def image_size(path):
    with Image.open(path) as im:
        return im.size


def make_coco(rows, image_dir):
    """One category per record (name = prompt/concept, +reasoning_type +kind); one
    annotation per instance (complex = 1 instance, multi = N). Disjoint id namespaces."""
    images, annotations, categories = [], [], []
    img_id = cat_id = ann_id = 0
    for rec in rows:
        w, h = image_size(Path(image_dir) / rec["file_name"])
        img_id += 1
        images.append({"id": img_id, "file_name": rec["file_name"], "width": w, "height": h})
        cat_id += 1
        categories.append({
            "id": cat_id, "name": rec["prompt"],
            "supercategory": rec.get("simple_prompt") or rec.get("object") or "object",
            "reasoning_type": rec["reasoning_type"], "kind": rec.get("kind", "complex"),
        })
        for inst in rec["instances"]:
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id,
                "bbox": inst["bbox"], "area": inst["mask_area"],
                "segmentation": inst["mask_rle"], "iscrowd": 0,
            })
    return {"images": images, "annotations": annotations, "categories": categories}


def stratified_split(rows, test_frac, seed):
    """Split per (kind, reasoning_type) so both tracks + all categories are represented."""
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r.get("kind", "complex"), r["reasoning_type"])].append(r)
    train, test = [], []
    for recs in buckets.values():
        rng.shuffle(recs)
        n_test = max(1, int(round(len(recs) * test_frac))) if len(recs) > 1 else 0
        test += recs[:n_test]
        train += recs[n_test:]
    return train, test


def _usable_positives(out_dir, image_dir, min_score=7):
    """Kept positives from BOTH tracks + the reusable legacy sa1b_pcs PCS set."""
    rows = [r for r in kept_positives(out_dir, min_score)
            if (image_dir / r["file_name"]).exists()]
    rows += load_sa1b_pcs(image_dir)
    return rows


def load_sa1b_pcs(image_dir,
                  path="/mnt/data0/ameen/train_data_sa1b_pcs/train_annotations.coco.json"):
    """Convert the legacy multi-instance PCS COCO (simple concept -> all instances, human-era
    pipeline output) into mix records. Format matches ours: RLE dict with str counts."""
    p = Path(path)
    if not p.exists():
        return []
    d = json.load(open(p))
    id2im = {im["id"]: im for im in d["images"]}
    id2cat = {c["id"]: c["name"] for c in d["categories"]}
    groups = collections.defaultdict(list)
    for a in d["annotations"]:
        groups[(a["image_id"], a["category_id"])].append(a)
    rows = []
    for (iid, cid), anns in groups.items():
        fn = os.path.basename(id2im[iid]["file_name"])
        if not (Path(image_dir) / fn).exists():
            continue
        rows.append({
            "file_name": fn, "kind": "multi", "reasoning_type": "multi_instance",
            "object": id2cat[cid], "prompt": id2cat[cid], "simple_prompt": id2cat[cid],
            "verdict": "pass", "score": None, "source": "sa1b_pcs",
            "instances": [{"mask_rle": a["segmentation"], "bbox": a["bbox"],
                           "mask_area": a.get("area", 0)} for a in anns],
        })
    return rows


def image_split(rows, test_frac, seed):
    """IMAGE-level train/test split: every record of an image lands in the same side, so no
    pixel leakage between splits even when tracks share images."""
    import hashlib
    def bucket(fn):
        h = hashlib.md5(f"{seed}:{fn}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF
    train = [r for r in rows if bucket(r["file_name"]) >= test_frac]
    test = [r for r in rows if bucket(r["file_name"]) < test_frac]
    return train, test


def stage_export(args):
    """Single combined export of ALL positives (both tracks), no ratio adjustment.
    Use `mix` if you want to enforce multi:complex and negative ratios."""
    out_dir = Path(args.out_dir)
    image_dir = Path(args.image_dir)
    rows = _usable_positives(out_dir, image_dir)
    if not rows:
        raise SystemExit("No usable PASS records to export.")
    train, test = stratified_split(rows, args.test_frac, args.seed)

    dataset_dir = out_dir / "dataset"
    for split, recs in [("train", train), ("test", test)]:
        p = dataset_dir / split / "_annotations.coco.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(make_coco(recs, image_dir), open(p, "w"), separators=(",", ":"))

    sidecar_path = out_dir / "image_negatives.json"
    if sidecar_path.exists():
        allfiles = {r["file_name"] for r in rows}
        full = json.load(open(sidecar_path))
        json.dump({k: v for k, v in full.items() if k in allfiles},
                  open(dataset_dir / "image_negatives.json", "w"))

    print(json.dumps({
        "train": len(train), "test": len(test), "total": len(rows),
        "per_type": dict(collections.Counter(r["reasoning_type"] for r in rows)),
        "out": str(dataset_dir),
    }, indent=2))


def stage_mix(args):
    """Build the FINAL mixed dataset with explicit ratios:

      --multi-frac         fraction of POSITIVES that are multi-instance (rest = complex)
      --neg-frac-complex   negative phrases added per complex positive
      --neg-frac-multi     negative phrases added per multi positive

    Complex + multi positives are merged under disjoint id namespaces (like the original
    merge_mixed). Negatives come from the verified-absent pool (image_negatives.json) and
    are written to a sidecar sized to the requested per-track budgets; the training loader
    turns each sidecar phrase into an empty-target datapoint.
    """
    out_dir = Path(args.out_dir)
    image_dir = Path(args.image_dir)
    rng = random.Random(args.seed)

    rows = _usable_positives(out_dir, image_dir, getattr(args, "min_score", 7))
    complex_rows = [r for r in rows if r.get("kind", "complex") == "complex"]
    multi_rows = [r for r in rows if r.get("kind") == "multi"]
    if not complex_rows and not multi_rows:
        raise SystemExit("No positives; run `generate` (both modes) first.")
    rng.shuffle(complex_rows)
    rng.shuffle(multi_rows)

    # --- enforce multi:complex ratio by subsampling the more-abundant track ---
    mf = args.multi_frac
    nc, nm = len(complex_rows), len(multi_rows)
    if mf <= 0:
        use_c, use_m = complex_rows, []
    elif mf >= 1:
        use_c, use_m = [], multi_rows
    else:
        need_m = int(round(mf / (1 - mf) * nc))       # multis needed to pair all complex
        if need_m <= nm:
            use_c, use_m = complex_rows, multi_rows[:need_m]
        else:                                          # not enough multi -> trim complex
            need_c = int(round((1 - mf) / mf * nm))
            use_c, use_m = complex_rows[:need_c], multi_rows
    positives = use_c + use_m
    if not positives:
        raise SystemExit("Ratio produced 0 positives; check --multi-frac and your data.")

    # --- IMAGE-level split (no leakage across tracks) + write merged COCO ---
    train, test = image_split(positives, args.test_frac, args.seed)
    dataset_dir = out_dir / "dataset_mixed"
    for split, recs in [("train", train), ("test", test)]:
        p = dataset_dir / split / "_annotations.coco.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(make_coco(recs, image_dir), open(p, "w"), separators=(",", ":"))

    # --- negatives: budget per track, drawn from the verified-absent pool ---
    sidecar_path = out_dir / "image_negatives.json"
    full = json.load(open(sidecar_path)) if sidecar_path.exists() else {}
    kept_files = {r["file_name"] for r in positives}
    pos_phrases = collections.defaultdict(set)
    for r in positives:
        for p in (r.get("simple_prompt"), r.get("object")):
            if p:
                pos_phrases[r["file_name"]].add(str(p).strip().lower())
    # pool of (file, phrase), restricted to images in the export, minus any positive phrase
    pool = [(fn, ph) for fn, phs in full.items() if fn in kept_files
            for ph in phs if ph not in pos_phrases[fn]]
    rng.shuffle(pool)

    want_c = int(round(args.neg_frac_complex * len(use_c)))
    want_m = int(round(args.neg_frac_multi * len(use_m)))
    want = min(len(pool), want_c + want_m)
    chosen = pool[:want]
    neg_sidecar = collections.defaultdict(list)
    for fn, ph in chosen:
        neg_sidecar[fn].append(ph)
    json.dump(neg_sidecar, open(dataset_dir / "image_negatives.json", "w"))

    realized_mf = round(len(use_m) / max(1, len(positives)), 3)
    composition = {
        "positives": {"complex": len(use_c), "multi": len(use_m), "total": len(positives)},
        "multi_frac": {"requested": mf, "realized": realized_mf},
        "negatives": {
            "requested_complex": want_c, "requested_multi": want_m,
            "budget": want_c + want_m, "written": len(chosen),
            "pool_available": len(pool),
            "neg_frac_complex": args.neg_frac_complex, "neg_frac_multi": args.neg_frac_multi,
        },
        "split": {"train": len(train), "test": len(test)},
        "per_type": dict(collections.Counter(r["reasoning_type"] for r in positives)),
        "note": ("negatives are image-keyed (not per-track in the sidecar); the loader's "
                 "max_negatives_per_datapoint still bounds per-step negatives at train time."),
        "out": str(dataset_dir),
    }
    json.dump(composition, open(dataset_dir / "composition.json", "w"), indent=2)
    print(json.dumps(composition, indent=2))


# --------------------------------------------------------------------------------------
# Stage: download SA-1B tar
# --------------------------------------------------------------------------------------
def stage_download(args):
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    tar_path = Path(args.out_dir) / "sa1b_part.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists() or args.force:
        log(f"downloading tar -> {tar_path}")
        urllib.request.urlretrieve(args.tar_url, tar_path)
    log(f"extracting .jpg from {tar_path} -> {image_dir}")
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.lower().endswith(".jpg"):
                m.name = os.path.basename(m.name)
                tf.extract(m, image_dir)
                n += 1
    log(f"download DONE: {n} images in {image_dir}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def stage_rescore(args):
    """Backfill scores onto already-generated `auto` samples: for each PASS record with no
    score, re-score with Gemini and re-render its samples_auto/<cat>/<stem>.png (black-on-white
    caption with the score). Masks come from results_auto.jsonl, so no SAM3 rerun is needed."""
    from pycocotools import mask as mask_util
    api_key = get_api_key(args.api_key)
    out_dir = Path(args.out_dir)
    image_dir = Path(args.image_dir)
    results_path = out_dir / "results_auto.jsonl"
    sample_dir = out_dir            # category folders live directly under out/
    records = load_jsonl(results_path)
    # ONLY the samples that were actually saved as previews (the <=per-cat kept per category);
    # never score/render PASS records that never made it to an image.
    def _preview_path(r):
        return sample_dir / r["reasoning_type"] / (Path(r["file_name"]).stem + ".png")
    todo = [r for r in records if r.get("verdict") == "pass"
            and (args.force or r.get("score") is None)
            and _preview_path(r).exists()]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        log("rescore: nothing to do (all PASS records already scored).")
        return
    log(f"rescore: scoring + re-rendering {len(todo)} PASS samples -> {sample_dir}")

    def _decode(inst):
        rle = inst["mask_rle"]
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        return mask_util.decode({"size": rle["size"], "counts": counts}).astype(bool)

    def one(r):
        try:
            img = Image.open(image_dir / r["file_name"]).convert("RGB")
        except Exception as e:
            log(f"rescore: open fail {r['file_name']}: {e}")
            return None
        m = _decode(r["instances"][0])
        score, just = score_single(GEMINI_MODEL, png_b64(highlight(img, m, alpha=0.0)),
                                    r["prompt"], r.get("object"), api_key)
        cat = CATEGORIES_BY_ID[r["reasoning_type"]]
        preview = with_caption(highlight(img, m, alpha=0.35), r["prompt"],
                               sub=f"{cat['name']}  ·  {r['reasoning_type']}  ·  PASS\n{just or ''}",
                               score=score)
        return r, score, just, preview

    keyf = lambda x: (x["file_name"], x["reasoning_type"])
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(one, todo):
            if not res:
                continue
            r, score, just, preview = res
            r["score"], r["score_justification"] = score, just
            d = sample_dir / r["reasoning_type"]
            d.mkdir(parents=True, exist_ok=True)
            preview.save(d / (Path(r["file_name"]).stem + ".png"))
            n += 1
            if n % 25 == 0:
                write_jsonl(results_path, records, key=keyf)
                log(f"  rescored {n}/{len(todo)}")
    write_jsonl(results_path, records, key=keyf)
    scored = [r["score"] for r in records
              if r.get("verdict") == "pass" and r.get("score") is not None]
    log(f"rescore DONE: {n} samples; mean score={np.mean(scored):.2f} over {len(scored)} PASS "
        f"-> {sample_dir}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage",
                    choices=["download", "generate", "auto", "rescore", "negatives",
                             "export", "mix", "all"])
    ap.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--api-key", default=None, help="Gemini key (else $GEMINI_API_KEY)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap #images for a stage")
    ap.add_argument("--target-pass", type=int, default=None,
                    help="generate: stop once this many kept (pass & score>=min-score) samples exist")
    ap.add_argument("--mode", choices=["complex", "multi"], default="complex",
                    help="generate: complex referring samples (8 categories) or multi-instance")
    ap.add_argument("--categories", nargs="+", choices=list(CATEGORIES_BY_ID),
                    help="restrict complex generation to these reasoning categories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cat", type=int, default=20,
                    help="auto: samples to collect per reasoning category")
    ap.add_argument("--min-score", type=int, default=1,
                    help="auto: only count samples with score >= this toward the per-cat quota")
    ap.add_argument("--no-preview", action="store_true",
                    help="auto: skip the caption PNG previews; store everything in results_auto.jsonl only")
    ap.add_argument("--gate-mode", choices=["cascade", "full"], default="cascade",
                    help="cascade (default): score first, skip gates for low scorers, run "
                         "gates in rejection-power order stopping at first fail — keep "
                         "decisions identical to full at ~37%% fewer LLM calls. "
                         "full: original always-all-gates order (complete diagnostics).")
    # negatives
    ap.add_argument("--neg-k", type=int, default=20, help="candidate negatives proposed/image")
    ap.add_argument("--neg-cap", type=int, default=15, help="verified negatives kept/image")
    # export / mix ratios
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--multi-frac", type=float, default=0.5,
                    help="mix: fraction of positives that are multi-instance (rest complex)")
    ap.add_argument("--neg-frac-complex", type=float, default=10.0,
                    help="mix: negative phrases added per complex positive")
    ap.add_argument("--neg-frac-multi", type=float, default=10.0,
                    help="mix: negative phrases added per multi positive")
    # download
    ap.add_argument("--tar-url", default=DEFAULT_TAR_URL)
    ap.add_argument("--force", action="store_true", help="re-download tar even if present")
    return ap


def main():
    args = build_parser().parse_args()
    if args.stage in ("download", "all"):
        stage_download(args)
    if args.stage == "generate":
        stage_generate(args)
    if args.stage == "auto":
        stage_auto(args)
    if args.stage == "rescore":
        stage_rescore(args)
    if args.stage == "all":                 # both tracks
        args.mode = "complex"; stage_generate(args)
        args.mode = "multi"; stage_generate(args)
    if args.stage in ("negatives", "all"):
        stage_negatives(args)
    if args.stage == "export":
        stage_export(args)
    if args.stage in ("mix", "all"):
        stage_mix(args)


if __name__ == "__main__":
    main()
