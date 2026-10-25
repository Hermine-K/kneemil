"""Turning free-text radiology reports into soft training targets with an LLM.

The LLM (an open-weights model run locally, since the competition data may not
be sent to a third party) classifies each finding into one of five certainty
levels. This module holds everything except the model itself: the prompt, a
tolerant parser for its answers, the mapping to soft targets and a
memory-safe batching rule.
"""

import json
import re

import numpy as np

from .constants import TARGETS

#: JSON keys used in the prompt (no apostrophe, easier for the model).
KEYS = [t.replace("Baker's", "Bakers") for t in TARGETS]

#: The five certainty levels the model must choose from.
CATEGORIES = ["definite", "probable", "possible", "negative", "not_mentioned"]

#: Mapping from certainty level to soft target.
SOFT = {"definite": 0.95, "probable": 0.80, "possible": 0.40, "negative": 0.03}

#: Soft target for a finding the report does not mention: absence of mention
#: is not a confirmed negative.
NOT_MENTIONED = 0.07

BASE_PROMPT = """You read knee MRI radiology reports. Reports may be in any language (English, Spanish, Turkish, German, French, Portuguese, etc.).
For each of the 12 findings below, classify what the report says about THIS knee into exactly one category:
"definite"  : clearly stated as present
"probable"  : likely, suspected, suggestive of, compatible with
"possible"  : cannot be excluded, questionable, minimal/borderline, to correlate clinically
"negative"  : explicitly stated as absent, normal or intact
"not_mentioned" : the report says nothing about it

Findings:
ACL: tear (partial or complete) or injury of the anterior cruciate ligament or of an ACL graft. Intact graft = negative.
MCL: sprain, tear or injury of the medial collateral ligament.
Medial Meniscus: TEAR of the medial meniscus. Grade I or II intrasubstance degeneration without surface extension is NOT a tear (negative).
Lateral Meniscus: TEAR of the lateral meniscus. Same rule as above.
Medial OA: osteoarthritis of the medial tibiofemoral compartment (cartilage loss, osteophytes, joint space narrowing, subchondral changes). Meniscal degeneration alone is NOT osteoarthritis.
Lateral OA: osteoarthritis of the lateral tibiofemoral compartment. Same rule.
PF OA: patellofemoral osteoarthritis (patellar or trochlear cartilage loss, chondromalacia, osteophytes).
Effusion: joint effusion. Physiological or trace fluid = possible.
Synovitis: synovitis, synovial thickening or inflammation.
Bakers: Baker's cyst or popliteal cyst.
Contusion: bone contusion, bone bruise, traumatic bone marrow edema.
Fracture: any fracture (including stress, insufficiency, avulsion, osteochondral).

Be careful with laterality: medial = internal, lateral = external.
Answer ONLY with a JSON object with exactly these 12 keys:
{"ACL":"...","MCL":"...","Medial Meniscus":"...","Lateral Meniscus":"...","Medial OA":"...","Lateral OA":"...","PF OA":"...","Effusion":"...","Synovitis":"...","Bakers":"...","Contusion":"...","Fracture":"..."}"""

GLOSSARY = """
Multilingual abbreviations and terms:
LCA (Spanish, French, Italian, Portuguese) = ACL. LCP = PCL, which is NOT the ACL.
LCM, LLI, LCI = MCL. LCL, LLE, FCL = lateral collateral ligament, which is NOT the MCL.
menisco interno, ménisque interne, Innenmeniskus, iç menisküs = medial meniscus. externo, externe, Außenmeniskus, dış menisküs = lateral meniscus.
rotura, rupture, déchirure, Riss, yırtık = tear. ön çapraz bağ = ACL. arka çapraz bağ = PCL.
derrame, épanchement, Erguss, efüzyon, sıvı artışı = effusion. edema óseo, œdème osseux, Knochenmarködem, kemik iliği ödemi, contusión = contusion.
fractura, fracture, Fraktur, kırık = fracture. quiste de Baker, kyste poplité, Baker kisti = Baker's cyst.
If the report has an impression or conclusion section, rely on it when the findings are ambiguous."""

RULE = ('\nEach value MUST be exactly one of: "definite", "probable", "possible", "negative", '
        '"not_mentioned". Never write the name of the finding as a value.')

#: Final system prompt (version 3, gold macro AUC 0.839).
SYSTEM_PROMPT = BASE_PROMPT + "\n" + GLOSSARY + RULE

SYNONYMS = {
    "definite": "definite", "present": "definite", "yes": "definite", "positive": "definite", "confirmed": "definite",
    "probable": "probable", "likely": "probable", "suspected": "probable", "suggestive": "probable",
    "possible": "possible", "questionable": "possible", "equivocal": "possible", "uncertain": "possible",
    "indeterminate": "possible",
    "negative": "negative", "absent": "negative", "no": "negative", "normal": "negative", "intact": "negative",
    "none": "negative",
    "not_mentioned": "not_mentioned", "notmentioned": "not_mentioned", "unknown": "not_mentioned",
    "na": "not_mentioned", "n/a": "not_mentioned",
}
NEGATION_WORDS = ("no ", "not ", "without", "intact", "normal", "absent", "none", "negative", "unremarkable")


def normalize_category(value):
    """Map a free-form answer to one of :data:`CATEGORIES`.

    Known synonyms are mapped directly. Any other non-empty value is read as
    present (the model sometimes writes the finding itself, such as "tear"),
    unless it contains a negation word.
    """
    v = str(value).strip().lower()
    if not v:
        return None
    for cand in (v, v.replace(" ", "_"), v.replace("-", "_"), v.split()[0].strip(",.;:")):
        if cand in SYNONYMS:
            return SYNONYMS[cand]
    if any(w in v + " " for w in NEGATION_WORDS):
        return "negative"
    return "definite"


def parse_answer(text, min_hits=6):
    """Parse the model answer into 12 categories, or ``None`` if unusable.

    Accepts JSON inside code fences, truncated JSON and unexpected wording.
    Missing keys become ``not_mentioned``; at least ``min_hits`` keys must be found.
    """
    if "{" not in text:
        return None
    blob = text[text.index("{"):]
    blob = blob[:blob.rindex("}") + 1] if "}" in blob else blob + "}"
    try:
        d = json.loads(blob)
    except Exception:
        d = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', blob))
    if not d:
        return None
    norm = {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in d.items()}
    out, hits = [], 0
    for k in KEYS:
        v = norm.get(re.sub(r"[^a-z]", "", k.lower()))
        c = normalize_category(v) if v is not None else None
        hits += c is not None
        out.append(c or "not_mentioned")
    return out if hits >= min_hits else None


def to_soft(categories):
    """Soft targets (floats) from a list of 12 categories (``None`` or NaN means not mentioned)."""
    return np.array([SOFT.get(c, NOT_MENTIONED) if isinstance(c, str) else NOT_MENTIONED
                     for c in categories], dtype=float)


def make_batches(lengths, token_budget=14000, max_batch=16):
    """Group prompts so that ``batch size x longest prompt`` stays under ``token_budget``.

    Parameters
    ----------
    lengths : list of (index, n_tokens)
        Prompt lengths; they are sorted by length first.

    Returns
    -------
    list of list of int
        Prompt indices per batch.
    """
    batches, cur, cur_max = [], [], 0
    for i, L in sorted(lengths, key=lambda x: x[1]):
        new_max = max(cur_max, L)
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * new_max > token_budget):
            batches.append(cur)
            cur, new_max = [], L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def generate_safe(generate, prompts, **kwargs):
    """Call ``generate(prompts)`` and split the batch in two on out-of-memory.

    ``generate`` is any function returning one string per prompt. A single
    prompt that still does not fit returns an empty answer instead of crashing.
    """
    try:
        return generate(prompts, **kwargs)
    except Exception as e:  # torch.cuda.OutOfMemoryError, without importing torch here
        if "out of memory" not in str(e).lower() and type(e).__name__ != "OutOfMemoryError":
            raise
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        if len(prompts) == 1:
            return [""]
        h = len(prompts) // 2
        return generate_safe(generate, prompts[:h], **kwargs) + generate_safe(generate, prompts[h:], **kwargs)
