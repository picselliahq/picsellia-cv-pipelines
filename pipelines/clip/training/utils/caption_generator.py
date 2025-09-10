import json, os, math, random
from collections import defaultdict, Counter
from PIL import Image
from pathlib import Path

def coco_to_rich_captions(
    coco_json_path: str,
    images_root: str,
    max_captions_per_image: int = 5,
    include_relations: bool = True,
    include_positions: bool = True,
    include_sizes: bool = True,
    seed: int = 42,
):
    """
    Build diverse captions per image from a COCO instances JSON.

    Returns:
        List[Dict]: [{"image": <abs_or_rel_image_path>, "captions": [<str>, ...]}, ...]
    """
    random.seed(seed)

    # ---- load coco ----
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        anns_by_img[a["image_id"]].append(a)

    # ---- helpers ----
    IRREG = {
        "person": "people", "man": "men", "woman": "women", "child": "children",
        "mouse": "mice", "goose": "geese", "tooth": "teeth", "foot": "feet"
    }

    def pluralize(noun, n):
        if n == 1:
            return noun
        if noun in IRREG:
            return IRREG[noun]
        if noun.endswith("y") and noun[-2] not in "aeiou":
            return noun[:-1] + "ies"
        if noun.endswith(("s","x","z","ch","sh")):
            return noun + "es"
        return noun + "s"

    def join_counts(counts: Counter):
        items = []
        for k, v in counts.items():
            items.append(f"{v} {pluralize(k, v)}")
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    def size_bucket(area_ratio):
        if area_ratio < 0.02: return "small"
        if area_ratio < 0.12: return "medium"
        return "large"

    def pos_words(cx, cy, W, H):
        horiz = "left" if cx < W/3 else ("right" if cx > 2*W/3 else "center")
        vert  = "top" if cy < H/3 else ("bottom" if cy > 2*H/3 else "middle")
        bits = []
        bits.append("on the left" if horiz=="left" else "on the right" if horiz=="right" else "in the center")
        if vert == "top": bits.append("at the top")
        elif vert == "bottom": bits.append("at the bottom")
        return ", ".join(bits)

    def relation(b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        c1x = x1 + w1/2; c2x = x2 + w2/2
        # overlap (any) -> "next to" (cheap but effective for variety)
        xi1, yi1 = max(x1,x2), max(y1,y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        if inter > 0:
            return "next to"
        return "to the left of" if c1x < c2x else "to the right of"

    # sentence templates
    GLOBAL_TEMPLATES = [
        "a photo of {main}.",
        "an image showing {main}.",
        "a scene with {main}.",
        "a photograph featuring {main}.",
        "a picture of {main}."
    ]
    ATTR_TEMPLATES = [
        "a {sz} {cat} {pos}.",
        "a {sz} {cat}.",
        "a {cat} {pos}.",
        "a photo of a {sz} {cat}.",
    ]
    REL_TEMPLATES = [
        "{a1} {c1} {rel} {a2} {c2}.",
        "you can see {a1} {c1} {rel} {a2} {c2}.",
        "a scene with {a1} {c1} {rel} {a2} {c2}.",
    ]

    results = []

    for img_id, meta in img_meta.items():
        file_name = meta["file_name"]
        img_path = str(Path(images_root) / file_name)

        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        # get width/height (prefer metadata; fall back to opening image)
        W = meta.get("width"); H = meta.get("height")
        if (W is None or H is None) or (W == 0 or H == 0):
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                # can't compute sizes/positions; degrade gracefully
                W = H = None

        captions = set()

        # --- global count-based caption
        counts = Counter(cat_name[a["category_id"]] for a in anns)
        # Trim to top few to avoid run-on sentences
        counts = Counter(dict(counts.most_common(6)))
        main = join_counts(counts)
        if main:
            # sample up to 2–3 variations
            k = min(3, len(GLOBAL_TEMPLATES))
            for t in random.sample(GLOBAL_TEMPLATES, k=k):
                captions.add(t.format(main=main))

        # --- attribute captions for prominent objects (size + position)
        if (include_positions or include_sizes) and (W and H):
            # sort by bbox area, largest first
            ann_sorted = sorted(
                anns, key=lambda a: a["bbox"][2]*a["bbox"][3], reverse=True
            )[:4]  # top few
            for a in ann_sorted:
                x,y,w,h = a["bbox"]
                cat = cat_name[a["category_id"]]
                pieces = {}
                if include_sizes:
                    ar = (w*h)/(W*H + 1e-6)
                    pieces["sz"] = size_bucket(ar)
                else:
                    pieces["sz"] = ""
                if include_positions:
                    pieces["pos"] = pos_words(x+w/2, y+h/2, W, H)
                else:
                    pieces["pos"] = ""
                pieces["cat"] = cat
                # choose 1–2 templates
                tks = min(2, len(ATTR_TEMPLATES))
                for t in random.sample(ATTR_TEMPLATES, k=tks):
                    # clean stray spaces when sz/pos disabled
                    sent = t.format(**pieces).replace("  ", " ").replace(" ,", ",").strip()
                    captions.add(sent)

        # --- relation captions (left/right/next to)
        if include_relations and len(anns) >= 2 and (W and H):
            # sample up to 3 distinct pairs
            sample = random.sample(anns, k=min(4, len(anns)))
            pairs = []
            for i in range(len(sample)):
                for j in range(i+1, len(sample)):
                    pairs.append((sample[i], sample[j]))
            random.shuffle(pairs)
            for a1, a2 in pairs[:3]:
                x1,y1,w1,h1 = a1["bbox"]; x2,y2,w2,h2 = a2["bbox"]
                rel = relation((x1,y1,w1,h1), (x2,y2,w2,h2))
                # sizes
                if include_sizes:
                    ar1 = (w1*h1)/(W*H + 1e-6); ar2 = (w2*h2)/(W*H + 1e-6)
                    s1 = size_bucket(ar1); s2 = size_bucket(ar2)
                else:
                    s1 = s2 = ""
                c1 = cat_name[a1["category_id"]]
                c2 = cat_name[a2["category_id"]]
                t = random.choice(REL_TEMPLATES)
                sent = t.format(a1=("a " + s1).strip(), c1=c1, rel=rel, a2=("a " + s2).strip(), c2=c2)
                captions.add(sent.replace("  ", " ").strip())

        # finalize: keep up to max_captions_per_image (stable but diverse)
        caps = list(captions)
        random.shuffle(caps)
        caps = caps[:max_captions_per_image]

        if caps:
            results.append({"image": img_path, "captions": caps, **meta})

    return results