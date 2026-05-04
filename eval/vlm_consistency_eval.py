"""
VLM Consistency & Uncertainty Evaluation
==========================================
Run this AFTER vlm_eval_tasks.py has finished.

Reads existing predictions.json produced by vlm_eval_tasks.py, reloads the
same model, and performs N stochastic forward passes per image (temperature
sampling).  Computes consistency and uncertainty metrics suitable for an
ICML/AAAI paper.

Per-image outputs
-----------------
  all_runs           list of N raw parsed results (one per stochastic pass)
  majority_vote      the most common binary prediction across N runs
  consistency_score  fraction of runs agreeing with majority_vote  (0→1)
  prediction_entropy Shannon entropy over the binary vote distribution  (bits)
  confidence_mean    mean model-reported confidence across N successful runs
  confidence_std     std  model-reported confidence across N successful runs
  single_run_*       the fields from the original single-run predictions.json
                     (anomaly_present, confidence, reasoning) — carried over for
                     cross-analysis without reloading vlm_eval_tasks output

For multiclass task, additionally:
  class_votes        {class_name: count} histogram over N runs
  class_entropy      entropy over 11-class distribution  (bits)

Aggregate outputs (per model × task)
-------------------------------------
  consistency_predictions.json    full per-image consistency records
  consistency_metrics.json        aggregate stats (mean consistency, ECE, …)
  figures/reliability_diagram.*   calibration reliability diagram (PDF+PNG)
  figures/consistency_hist.*      histogram of consistency scores
  figures/confidence_vs_correct.* scatter: mean confidence vs correctness
  figures/ece_comparison.*        ECE bar chart across models (if --model all)
  figures/per_class_consistency.* heatmap: consistency score per anomaly class

Usage
-----
  # Consistency for binary task, single model, 5 runs at temperature=0.7
  python vlm_consistency_eval.py \\
      --task        binary \\
      --model       qwen25vl_7b \\
      --images-dir  ./Data/images \\
      --dataset-json ./Data/dataset.json \\
      --eval-dir    ./vlm_eval_outputs \\
      --n-runs      5 \\
      --temperature 0.7

  # Run all models
  python vlm_consistency_eval.py --task binary --model all \\
      --images-dir ./Data/images --dataset-json ./Data/dataset.json \\
      --eval-dir ./vlm_eval_outputs --n-runs 5

  # Only regenerate figures from saved consistency_predictions.json
  python vlm_consistency_eval.py --task binary --model qwen25vl_7b \\
      --eval-dir ./vlm_eval_outputs --figures-only

Requirements:
    Same environment as vlm_eval_tasks.py (transformers, torch, Pillow, tqdm,
    scikit-learn, matplotlib, seaborn, numpy)
"""

import argparse, gc, json, math, os, re, sys, time, warnings
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install torch Pillow tqdm")

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install numpy matplotlib seaborn scikit-learn")


# ── Publication-quality matplotlib defaults ────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# =============================================================================
# TAXONOMY (must match vlm_eval_tasks.py exactly)
# =============================================================================

NORMAL_CLASS = "normal"

ANOMALY_CLASSES = [
    "animal_on_road",
    "extreme_weather",
    "road_surface_hazard",
    "fallen_debris_or_vegetation",
    "strange_object_on_road",
    "vehicle_incident",
    "infrastructure_failure",
    "human_presence_anomaly",
    "adverse_lighting",
    "oversized_or_unusual_vehicle",
    "multi_hazard_compound",
]

ALL_CLASSES = [NORMAL_CLASS] + ANOMALY_CLASSES


# =============================================================================
# MODEL REGISTRY  (must match vlm_eval_tasks.py exactly)
# =============================================================================

MODEL_REGISTRY = {
    "internvl3_1b":   {"hf_id": "OpenGVLab/InternVL3-1B-hf",                "loader": "internvl3",      "display_name": "InternVL3-1B"},
    "internvl3_2b":   {"hf_id": "OpenGVLab/InternVL3-2B-hf",                "loader": "internvl3",      "display_name": "InternVL3-2B"},
    "internvl3_8b":   {"hf_id": "OpenGVLab/InternVL3-8B-hf",                "loader": "internvl3",      "display_name": "InternVL3-8B"},
    "qwen2vl_2b":     {"hf_id": "Qwen/Qwen2-VL-2B-Instruct",                "loader": "qwen2vl",        "display_name": "Qwen2-VL-2B"},
    "qwen25vl_3b":    {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",              "loader": "qwen25vl",       "display_name": "Qwen2.5-VL-3B"},
    "qwen25vl_7b":    {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",              "loader": "qwen25vl",       "display_name": "Qwen2.5-VL-7B"},
    "llava_ov_7b":    {"hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",  "loader": "llava_onevision","display_name": "LLaVA-OV-7B"},
    "llava_13b":      {"hf_id": "llava-hf/llava-v1.6-vicuna-13b-hf",        "loader": "llava_next",     "display_name": "LLaVA-1.6-13B"},
    "llama32_11b":    {"hf_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",  "loader": "llama32",        "display_name": "Llama-3.2-11B"},
}


# =============================================================================
# PROMPTS (must match vlm_eval_tasks.py)
# =============================================================================

SYSTEM_PROMPT = """\
You are an autonomous driving safety analyst examining dashcam images.
Your task is to determine whether a driving scene contains a safety anomaly.

═══════════════════════════════════════════════════════════════
  ANOMALY TAXONOMY  (memorise all 10 classes exactly)
═══════════════════════════════════════════════════════════════

1. animal_on_road
   Any animal — deer, dog, cow, horse, wild boar, bird flock —
   that is physically ON the road surface or actively crossing
   in front of the ego vehicle. Animals on the pavement/sidewalk
   do NOT count.

2. extreme_weather
   Severe weather that makes the road surface completely invisible
   or impassable: dense fog with near-zero visibility, active flash
   flooding covering the road, heavy blizzard obscuring lane markings,
   black ice (road appears wet but is frozen). Light rain, overcast
   skies, and normal night driving are NOT extreme weather.

3. road_surface_hazard
   Physical damage to or blockage embedded IN the road surface:
   large sinkholes, collapsed pavement, rockslides/mudslides across
   the road, a flooded underpass where the road is submerged. The
   hazard must be on or part of the road itself, not beside it.

4. fallen_debris_or_vegetation
   Objects that have FALLEN ONTO the road and block a driving lane:
   fallen trees, large branches, boulders, spilled cargo from trucks,
   scattered furniture or construction material. Small leaves or
   surface dirt do NOT qualify.

5. vehicle_incident
   A vehicle in an abnormal state that blocks or endangers traffic:
   overturned car, crashed/crumpled vehicle in the lane, vehicle on
   fire, car stopped sideways across multiple lanes, wrong-way driver
   approaching head-on. A vehicle parked normally at the roadside
   does NOT count.

6. infrastructure_failure
   A road infrastructure element that has physically collapsed or
   fallen onto the road: fallen traffic-light pole, downed overhead
   power line across the lane, collapsed road barrier/guardrail lying
   in the road, a crumpled road sign blocking a lane. Signs that are
   merely tilted or bent do NOT count.

7. human_presence_anomaly
   A person who is IN the road in an unexpected or dangerous position:
   standing or walking in a driving lane, lying on the road, running
   across lanes outside a marked crossing. Pedestrians on sidewalks
   or at a marked crossing are NORMAL.

8. adverse_lighting
   Extreme lighting that makes safe driving impossible (beyond normal
   night-time or daytime glare): direct blinding sun glare that makes
   the road completely invisible, a tunnel with total darkness and no
   lights, blinding oncoming high-beams that wash out the entire scene.
   Standard night driving with streetlights is NOT adverse lighting.

9. oversized_or_unusual_vehicle
   A vehicle that is either (a) physically oversized and dominates the
   camera frame — cement mixer with rotating drum, dump truck with
   raised tipper body, mobile crane with extended boom, heavy machinery
   transporter — or (b) wrong for this road type — agricultural tractor
   or harvester on a public highway, military vehicle/tank, horse-drawn
   carriage on a motorway.
   EXPLICITLY NOT this class: standard semi-trucks/lorries driving
   normally in a highway lane, city buses on city streets, delivery
   vans, box trucks.

10. multi_hazard_compound
    Two or more of the above nine anomaly classes are SIMULTANEOUSLY
    visible in the same image (e.g. a vehicle incident AND fallen debris).

═══════════════════════════════════════════════════════════════
  ALWAYS NORMAL — never classify these as anomalous
═══════════════════════════════════════════════════════════════
• Cars, SUVs, motorcycles, vans driving in their lanes
• Standard semi-trucks or lorries on highways
• City buses or coaches on city streets
• Delivery trucks or box trucks
• Pedestrians on sidewalks or at marked crossings
• Cyclists in a bike lane
• Construction zones with cones and signs (no lane-blocking debris)
• Overcast sky, light rain, dawn/dusk lighting, normal night driving
• Traffic jams — multiple vehicles stopped normally

═══════════════════════════════════════════════════════════════
  CRITICAL RULES
═══════════════════════════════════════════════════════════════
• If you see a cement mixer drum, rotating bowl, crane arm,
  tipper body or heavy construction equipment on a vehicle →
  classify as oversized_or_unusual_vehicle, even if it is
  driving normally in a lane.
• If an animal is near the road but NOT on the paved surface →
  classify as normal.
• Do NOT invent anomalies. If the scene is ambiguous → normal.
• Confidence must reflect genuine uncertainty, not politeness.
"""

BINARY_PROMPT = """\
Examine the dashcam image carefully.

Determine whether this driving scene contains a safety anomaly as defined
in the taxonomy above.

Return ONLY the following JSON — no extra text, no markdown fences:
{
  "anomaly_present": true | false,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one concise sentence stating the primary visual evidence>"
}"""

MULTICLASS_PROMPT = """\
Examine the dashcam image carefully.

Classify this scene into EXACTLY ONE of these 11 classes:
  normal, animal_on_road, extreme_weather, road_surface_hazard,
  fallen_debris_or_vegetation, vehicle_incident, infrastructure_failure,
  human_presence_anomaly, adverse_lighting, oversized_or_unusual_vehicle,
  multi_hazard_compound

Return ONLY the following JSON — no extra text, no markdown fences:
{
  "anomaly_present": true | false,
  "scene_class": "<one of the 11 class names above>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one concise sentence>"
}"""


# =============================================================================
# JSON PARSING  (must match vlm_eval_tasks.py)
# =============================================================================

def parse_json_response(raw: str) -> dict | None:
    """Robust JSON extraction with fallbacks."""
    if not raw:
        return None
    text = raw.strip()
    # Fix Python booleans / None that some models emit
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b',  'null',  text)
    # Try code fences
    fence_m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_m:
        try:
            return json.loads(fence_m.group(1))
        except json.JSONDecodeError:
            pass
    # Greedily find last {...}
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    return None


def _text_fallback_binary(raw: str) -> dict | None:
    t = raw.lower()
    m = re.search(r'"anomaly_present"\s*:\s*(true|false)', t)
    if m:
        val = m.group(1) == "true"
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
        conf   = float(conf_m.group(1)) if conf_m else 0.5
        reason = raw.strip()[:120]
        return {"anomaly_present": val, "confidence": conf, "reasoning": reason}
    strong_neg = ["no anomaly", "normal scene", "no safety", "not anomal",
                  "everything appears normal", "scene is normal"]
    if any(p in t for p in strong_neg):
        return {"anomaly_present": False, "confidence": 0.5, "reasoning": raw.strip()[:120]}
    anomaly_kws = ["animal on road", "road hazard", "debris", "vehicle incident",
                   "infrastructure failure", "human presence", "adverse lighting",
                   "oversized vehicle", "extreme weather", "anomal"]
    if any(p in t for p in anomaly_kws):
        return {"anomaly_present": True, "confidence": 0.5, "reasoning": raw.strip()[:120]}
    return None


def _text_fallback_multiclass(raw: str) -> dict | None:
    t = raw.lower()
    binary = _text_fallback_binary(raw)
    if binary is None:
        return None
    for cls in ALL_CLASSES:
        if cls.replace("_", " ") in t or cls in t:
            return {**binary, "scene_class": cls, "confidence": binary["confidence"],
                    "reasoning": binary["reasoning"]}
    return {**binary, "scene_class": "normal" if not binary["anomaly_present"] else "unknown",
            "confidence": binary["confidence"], "reasoning": binary["reasoning"]}


# =============================================================================
# MODEL LOADERS (adapted from vlm_eval_tasks.py — same interface, different
# generate call that accepts temperature)
# =============================================================================

def _make_stochastic_infer(base_infer_fn, temperature: float):
    """
    Wraps a deterministic `infer(model, proc, image, system, user)` function to
    use temperature sampling.  Patches model.generate temporarily via a closure
    that passes do_sample=True and temperature into every model.generate call.

    Because each model loader bakes `model.generate` calls inside a closure, we
    intercept by monkey-patching `model.generate` before calling `infer`, then
    restoring it afterwards.
    """
    import functools

    def stochastic_infer(model, processor, image, system, user):
        original_generate = model.generate

        @functools.wraps(original_generate)
        def patched_generate(*args, **kwargs):
            kwargs["do_sample"]   = True
            kwargs["temperature"] = temperature
            kwargs.pop("greedy",  None)
            return original_generate(*args, **kwargs)

        model.generate = patched_generate
        try:
            result = base_infer_fn(model, processor, image, system, user)
        finally:
            model.generate = original_generate
        return result

    return stochastic_infer


# ── Re-export the same loaders from vlm_eval_tasks.py ────────────────────────
# We import the loader functions directly to avoid code duplication.

def _import_loaders():
    """Dynamically import LOADERS dict from vlm_eval_tasks.py."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "vlm_eval_tasks",
        Path(__file__).resolve().parent / "vlm_eval_tasks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["vlm_eval_tasks"] = mod
    spec.loader.exec_module(mod)
    return mod.LOADERS


# =============================================================================
# GROUND TRUTH LOADER
# =============================================================================

def load_ground_truth(dataset_json: Path) -> dict:
    """Returns {filename: {"anomaly_present": bool, "scene_class": str, "description": str}}"""
    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data)
    gt: dict = {}
    for key, record in samples.items():
        if key == "metadata":
            continue
        if isinstance(record, dict) and "anomaly_present" in record:
            gt[key] = record
    return gt


def find_images(images_dir: Path) -> list[Path]:
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    return paths


def load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# =============================================================================
# SINGLE-IMAGE STOCHASTIC INFERENCE
# =============================================================================

def _run_one_pass_binary(model, processor, stoch_infer, image_path: Path, model_key: str) -> dict:
    try:
        image = load_image_pil(image_path)
        raw   = stoch_infer(model, processor, image, SYSTEM_PROMPT, BINARY_PROMPT)
        parsed = parse_json_response(raw)
        if parsed is None:
            parsed = _text_fallback_binary(raw)
        if parsed is None:
            return {"status": "failed", "raw": raw[:200]}
        return {
            "status":          "success",
            "anomaly_present": bool(parsed.get("anomaly_present", False)),
            "confidence":      float(parsed.get("confidence", 0.5)),
            "reasoning":       str(parsed.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _run_one_pass_multiclass(model, processor, stoch_infer, image_path: Path, model_key: str) -> dict:
    try:
        image = load_image_pil(image_path)
        raw   = stoch_infer(model, processor, image, SYSTEM_PROMPT, MULTICLASS_PROMPT)
        parsed = parse_json_response(raw)
        if parsed is None:
            parsed = _text_fallback_multiclass(raw)
        if parsed is None:
            return {"status": "failed", "raw": raw[:200]}
        raw_class = str(parsed.get("scene_class", "")).strip().lower().replace(" ", "_")
        if raw_class not in ALL_CLASSES:
            raw_class = "normal" if not parsed.get("anomaly_present", True) else "unknown"
        return {
            "status":          "success",
            "anomaly_present": bool(parsed.get("anomaly_present", False)),
            "scene_class":     raw_class,
            "confidence":      float(parsed.get("confidence", 0.5)),
            "reasoning":       str(parsed.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# =============================================================================
# CONSISTENCY COMPUTATION
# =============================================================================

def compute_binary_consistency(runs: list[dict]) -> dict:
    """
    Given N run results (dicts with status/anomaly_present/confidence),
    return consistency statistics.
    """
    successful = [r for r in runs if r.get("status") == "success"]
    n_ok = len(successful)
    if n_ok == 0:
        return {
            "majority_vote":      None,
            "consistency_score":  0.0,
            "prediction_entropy": math.log2(2),   # max uncertainty
            "confidence_mean":    None,
            "confidence_std":     None,
            "n_successful_runs":  0,
        }

    votes = [r["anomaly_present"] for r in successful]
    n_true  = sum(votes)
    n_false = n_ok - n_true

    majority_vote     = n_true >= n_false
    consistency_score = max(n_true, n_false) / n_ok

    # Entropy over the binary distribution
    p_anom = n_true / n_ok
    p_norm = n_false / n_ok
    def _h(p): return 0.0 if p == 0 else -p * math.log2(p)
    prediction_entropy = _h(p_anom) + _h(p_norm)

    confs = [r["confidence"] for r in successful]
    return {
        "majority_vote":      majority_vote,
        "consistency_score":  round(consistency_score, 4),
        "prediction_entropy": round(prediction_entropy, 4),
        "confidence_mean":    round(float(np.mean(confs)), 4),
        "confidence_std":     round(float(np.std(confs)),  4),
        "n_successful_runs":  n_ok,
    }


def compute_multiclass_consistency(runs: list[dict]) -> dict:
    """Extends binary consistency with per-class vote histogram and class entropy."""
    base = compute_binary_consistency(runs)
    successful = [r for r in runs if r.get("status") == "success"]
    if not successful:
        base["class_votes"]  = {}
        base["class_entropy"] = math.log2(len(ALL_CLASSES))
        return base

    class_counts = Counter(r.get("scene_class", "unknown") for r in successful)
    n_ok = len(successful)

    # Entropy over 11-class distribution
    probs = [class_counts.get(c, 0) / n_ok for c in ALL_CLASSES]
    def _h(p): return 0.0 if p == 0 else -p * math.log2(p)
    class_entropy = sum(_h(p) for p in probs)

    base["class_votes"]   = dict(class_counts)
    base["class_entropy"] = round(class_entropy, 4)
    return base


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

def compute_aggregate_binary(consistency_records: list[dict], gt_map: dict) -> dict:
    """
    Computes aggregate metrics over all images:
    - mean/std consistency, entropy, confidence
    - ECE (Expected Calibration Error) using majority-vote confidence
    - Accuracy of majority vote vs. ground truth
    """
    valid = [r for r in consistency_records
             if r.get("majority_vote") is not None
             and r.get("filename") in gt_map]
    if not valid:
        return {}

    consistencies  = [r["consistency_score"]  for r in valid]
    entropies      = [r["prediction_entropy"]  for r in valid]
    conf_means     = [r["confidence_mean"]     for r in valid if r["confidence_mean"] is not None]

    # Majority-vote accuracy
    y_true = [int(gt_map[r["filename"]].get("anomaly_present", False)) for r in valid]
    y_pred = [int(r["majority_vote"]) for r in valid]
    acc    = accuracy_score(y_true, y_pred)
    bal    = balanced_accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    # ECE: use consistency_score as the "calibrated confidence"
    # (higher consistency → model is more certain)
    n_bins = 10
    ece    = _compute_ece(
        y_true_bin=[bool(gt) for gt in y_true],
        y_pred_bin=[r["majority_vote"] for r in valid],
        confidences=[r["consistency_score"] for r in valid],
        n_bins=n_bins,
    )

    return {
        "n_images":            len(valid),
        "majority_vote_accuracy":   round(acc, 4),
        "majority_vote_balanced_acc": round(bal, 4),
        "majority_vote_f1":    round(f1, 4),
        "mean_consistency":    round(float(np.mean(consistencies)), 4),
        "std_consistency":     round(float(np.std(consistencies)),  4),
        "mean_entropy":        round(float(np.mean(entropies)), 4),
        "std_entropy":         round(float(np.std(entropies)),  4),
        "mean_confidence":     round(float(np.mean(conf_means)), 4) if conf_means else None,
        "std_confidence":      round(float(np.std(conf_means)),  4) if conf_means else None,
        "ece":                 round(ece, 4),
    }


def _compute_ece(y_true_bin: list[bool], y_pred_bin: list[bool],
                 confidences: list[float], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Bins predictions by confidence (consistency_score), computes
    |fraction correct - mean confidence| per bin, weighted by bin size.
    """
    n = len(y_true_bin)
    if n == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [lo <= c <= hi for c in confidences]
        if not any(mask):
            continue
        idxs    = [i for i, m in enumerate(mask) if m]
        bin_acc = np.mean([y_true_bin[i] == y_pred_bin[i] for i in idxs])
        bin_conf= np.mean([confidences[i]                 for i in idxs])
        ece    += len(idxs) / n * abs(bin_acc - bin_conf)
    return float(ece)


# =============================================================================
# FIGURES
# =============================================================================

def _save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def plot_consistency_histogram(consistency_records: list[dict], display: str, out_dir: Path):
    scores = [r["consistency_score"] for r in consistency_records
              if r.get("majority_vote") is not None]
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(scores, bins=20, range=(0.0, 1.0), color=PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {np.mean(scores):.2f}")
    ax.set_xlabel("Consistency Score")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Prediction Consistency — {display}")
    ax.legend()
    _save_fig(fig, out_dir, "consistency_histogram")


def plot_reliability_diagram(consistency_records: list[dict], gt_map: dict,
                              display: str, out_dir: Path, n_bins: int = 10):
    """
    Reliability diagram (calibration plot).
    X-axis: consistency_score (used as confidence)
    Y-axis: fraction of correct majority-vote predictions in that bin
    """
    valid = [r for r in consistency_records
             if r.get("majority_vote") is not None and r.get("filename") in gt_map]
    if not valid:
        return

    confs   = [r["consistency_score"]  for r in valid]
    correct = [r["majority_vote"] == gt_map[r["filename"]].get("anomaly_present", False)
               for r in valid]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc  = []
    bin_conf = []
    bin_cnt  = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        idxs = [i for i, c in enumerate(confs) if lo <= c <= hi]
        if not idxs:
            continue
        bin_acc.append(np.mean([correct[i] for i in idxs]))
        bin_conf.append(np.mean([confs[i]  for i in idxs]))
        bin_cnt.append(len(idxs))

    if not bin_acc:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    bar_w = 1.0 / n_bins * 0.8
    ax.bar(bin_conf, bin_acc, width=bar_w, alpha=0.6, color=PALETTE[0], label="Model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Consistency Score (confidence proxy)")
    ax.set_ylabel("Fraction Correct")
    ax.set_title(f"Reliability Diagram — {display}")
    ax.legend()

    ece = _compute_ece(
        [gt_map[r["filename"]].get("anomaly_present", False) for r in valid],
        [r["majority_vote"] for r in valid],
        confs,
        n_bins=n_bins,
    )
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=9, verticalalignment="top")
    _save_fig(fig, out_dir, "reliability_diagram")


def plot_confidence_vs_correctness(consistency_records: list[dict], gt_map: dict,
                                    display: str, out_dir: Path):
    """Scatter: mean confidence (y) vs consistency_score (x), coloured by correctness."""
    valid = [r for r in consistency_records
             if r.get("majority_vote") is not None
             and r.get("confidence_mean") is not None
             and r.get("filename") in gt_map]
    if not valid:
        return

    x = [r["consistency_score"] for r in valid]
    y = [r["confidence_mean"]   for r in valid]
    c = [r["majority_vote"] == gt_map[r["filename"]].get("anomaly_present", False)
         for r in valid]
    colors = [PALETTE[2] if ok else PALETTE[3] for ok in c]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, c=colors, alpha=0.35, s=10, linewidths=0)
    ax.set_xlabel("Consistency Score")
    ax.set_ylabel("Mean Confidence (across runs)")
    ax.set_title(f"Confidence vs. Consistency — {display}")
    handles = [
        mpatches.Patch(color=PALETTE[2], label="Correct (majority vote)"),
        mpatches.Patch(color=PALETTE[3], label="Incorrect"),
    ]
    ax.legend(handles=handles)
    _save_fig(fig, out_dir, "confidence_vs_consistency")


def plot_per_class_consistency(consistency_records: list[dict], gt_map: dict,
                                display: str, out_dir: Path):
    """
    Heatmap: rows = ground-truth anomaly class, columns = [mean_consistency, mean_entropy].
    Only for images whose GT class is known.
    """
    from collections import defaultdict

    class_data: dict[str, list] = defaultdict(list)
    for r in consistency_records:
        if r.get("majority_vote") is None:
            continue
        fname = r.get("filename", "")
        if fname not in gt_map:
            continue
        gt_rec = gt_map[fname]
        # Try to get GT class — fall back to anomaly_present flag
        gt_cls = gt_rec.get("scene_class", "")
        if not gt_cls:
            gt_cls = "anomalous" if gt_rec.get("anomaly_present") else "normal"
        class_data[gt_cls].append({
            "consistency": r["consistency_score"],
            "entropy":     r["prediction_entropy"],
        })

    if not class_data:
        return

    classes = sorted(class_data.keys())
    data_mat = np.array([
        [np.mean([d["consistency"] for d in class_data[c]]),
         np.mean([d["entropy"]     for d in class_data[c]])]
        for c in classes
    ])

    fig, ax = plt.subplots(figsize=(5, max(3, 0.45 * len(classes))))
    im = ax.imshow(data_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mean Consistency", "Mean Entropy"], fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=8)
    ax.set_title(f"Per-Class Consistency — {display}")
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Annotate cells
    for i in range(len(classes)):
        for j in range(2):
            ax.text(j, i, f"{data_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    _save_fig(fig, out_dir, "per_class_consistency")


def plot_ece_comparison(model_ece: dict[str, float], out_dir: Path):
    """Bar chart of ECE across models (for aggregate view)."""
    if not model_ece:
        return
    names = list(model_ece.keys())
    vals  = [model_ece[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(names)), 4))
    bars = ax.bar(range(len(names)), vals, color=PALETTE[:len(names)], edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Comparison Across Models")
    ax.set_ylim(0, max(vals) * 1.25 + 0.01)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out_dir, "ece_comparison")


def plot_consistency_comparison(model_mean_consistency: dict[str, float], out_dir: Path):
    """Bar chart of mean consistency across models."""
    if not model_mean_consistency:
        return
    names = list(model_mean_consistency.keys())
    vals  = [model_mean_consistency[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(names)), 4))
    bars = ax.bar(range(len(names)), vals, color=PALETTE[:len(names)], edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Consistency Score")
    ax.set_title("Prediction Consistency Across Models")
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out_dir, "consistency_comparison")


# =============================================================================
# SAVE HELPERS
# =============================================================================

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# PER-IMAGE CLI DISPLAY
# =============================================================================

def _print_consistency_row(task: str, image_id: str, result: dict,
                            gt_map: dict, idx: int, total: int):
    filename = result.get("filename", image_id)
    gt_rec   = gt_map.get(filename, {})
    gt_anom  = gt_rec.get("anomaly_present", "?")
    mv       = result.get("majority_vote")
    cons     = result.get("consistency_score", 0.0)
    ent      = result.get("prediction_entropy", 0.0)
    n_ok     = result.get("n_successful_runs", 0)

    gt_tag   = "ANOMALY" if gt_anom is True  else ("NORMAL" if gt_anom is False else "?")
    mv_tag   = "ANOMALY" if mv is True        else ("NORMAL" if mv is False       else "?")
    match    = "✓" if (mv is not None and mv == gt_anom) else "✗"

    if task == "multiclass":
        cv = result.get("class_votes", {})
        top = max(cv, key=cv.get, default="?") if cv else "?"
        print(f"  [{idx:>6}/{total}] {match}  {image_id:<18}  "
              f"GT={gt_tag:<8} MV={mv_tag:<8} cons={cons:.2f} ent={ent:.2f}  "
              f"top_class={top}  runs={n_ok}")
    else:
        print(f"  [{idx:>6}/{total}] {match}  {image_id:<18}  "
              f"GT={gt_tag:<8} MV={mv_tag:<8} cons={cons:.2f} ent={ent:.2f}  "
              f"runs={n_ok}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_consistency(
    task:         str,
    model_key:    str,
    images_dir:   Path,
    dataset_json: Path,
    eval_dir:     Path,
    n_runs:       int,
    temperature:  float,
    use_4bit:     bool,
    delay:        float,
    preview:      int,
    figures_only: bool,
    hf_token:     str | None,
):
    cfg     = MODEL_REGISTRY[model_key]
    display = cfg["display_name"]

    task_dir   = eval_dir / task / model_key
    out_dir    = task_dir / "consistency"
    fig_dir    = out_dir  / "figures"
    pred_path  = out_dir  / "consistency_predictions.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Consistency Eval")
    print(f"  Task        : {task.upper()}")
    print(f"  Model       : {display}")
    print(f"  N runs      : {n_runs}  |  Temperature: {temperature}")
    print(f"  Eval dir    : {task_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*65}")

    # ── Load ground truth ─────────────────────────────────────────────────────
    gt_map = load_ground_truth(dataset_json)
    print(f"[INFO] GT records: {len(gt_map):,}")

    # ── Load single-run predictions from vlm_eval_tasks.py ───────────────────
    single_run_path = task_dir / "predictions.json"
    single_run_preds: dict[str, dict] = {}
    if single_run_path.exists():
        with open(single_run_path) as f:
            sr_data = json.load(f)
        for rec in sr_data.get("predictions", []):
            single_run_preds[rec.get("filename", "")] = rec
        print(f"[INFO] Single-run predictions loaded: {len(single_run_preds):,} images")
    else:
        print(f"[WARN] {single_run_path} not found — single-run fields will be omitted. "
              f"Run vlm_eval_tasks.py first.")

    # ── Collect images ────────────────────────────────────────────────────────
    images = find_images(images_dir)
    if task in ("description", "multiclass"):
        images = [p for p in images if gt_map.get(p.name, {}).get("anomaly_present")]
    if preview > 0:
        images = images[:preview]
    images_to_run = images

    if figures_only:
        images_to_run = []

    # ── Resume support ────────────────────────────────────────────────────────
    consistency_results: list[dict] = []
    processed_ids: set[str] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            existing = json.load(f)
        consistency_results = existing.get("results", [])
        processed_ids = {r.get("image_id") for r in consistency_results}
        if processed_ids:
            print(f"[INFO] Resuming: {len(consistency_results):,} already done.")

    images_to_run = [p for p in images_to_run if p.stem not in processed_ids]

    if images_to_run:
        # ── Load model ────────────────────────────────────────────────────────
        try:
            LOADERS = _import_loaders()
        except Exception as e:
            sys.exit(f"[ERROR] Could not import loaders from vlm_eval_tasks.py: {e}")

        print(f"[INFO] Loading {display} ...")
        model, processor, base_infer = LOADERS[cfg["loader"]](cfg["hf_id"], use_4bit)
        stoch_infer = _make_stochastic_infer(base_infer, temperature)

        if task == "binary":
            run_one_fn      = _run_one_pass_binary
            consistency_fn  = compute_binary_consistency
        else:  # multiclass
            run_one_fn      = _run_one_pass_multiclass
            consistency_fn  = compute_multiclass_consistency

        total = len(images_to_run)
        print(f"[INFO] Running {n_runs} stochastic passes on {total:,} images ...")

        for idx, image_path in enumerate(tqdm(images_to_run,
                                               desc=f"{display}/consistency",
                                               unit="img"), start=1):
            image_id = image_path.stem
            filename = image_path.name

            # N stochastic passes
            all_runs = []
            for _ in range(n_runs):
                run_result = run_one_fn(model, processor, stoch_infer, image_path, model_key)
                all_runs.append(run_result)

            # Compute consistency stats
            stats = consistency_fn(all_runs)

            # Carry over single-run fields for cross-analysis
            sr = single_run_preds.get(filename, {})
            record = {
                "image_id":                image_id,
                "filename":                filename,
                # Consistency fields
                **stats,
                "all_runs":                all_runs,
                # Single-run fields (from vlm_eval_tasks.py output)
                "single_run_status":       sr.get("status"),
                "single_run_anomaly":      sr.get("anomaly_present"),
                "single_run_confidence":   sr.get("confidence"),
                "single_run_reasoning":    sr.get("reasoning", "")[:200],
            }
            if task == "multiclass" and "scene_class" in sr:
                record["single_run_scene_class"] = sr["scene_class"]

            consistency_results.append(record)

            _print_consistency_row(task, image_id, record, gt_map,
                                   len(consistency_results), len(images))

            # Incremental save (crash-safe)
            save_json(pred_path, {
                "model_key":   model_key,
                "display":     display,
                "task":        task,
                "n_runs":      n_runs,
                "temperature": temperature,
                "results":     consistency_results,
            })
            time.sleep(delay)

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] VRAM freed.")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print(f"[INFO] Computing aggregate consistency metrics ...")
    agg = compute_aggregate_binary(consistency_results, gt_map)
    save_json(out_dir / "consistency_metrics.json", {
        "model_key":   model_key,
        "display":     display,
        "task":        task,
        "n_runs":      n_runs,
        "temperature": temperature,
        **agg,
    })
    if agg:
        print(f"\n  ── Consistency Metrics — {display} ──")
        for k in ["n_images", "majority_vote_accuracy", "majority_vote_balanced_acc",
                  "majority_vote_f1", "mean_consistency", "std_consistency",
                  "mean_entropy", "ece"]:
            print(f"    {k:<35}: {agg.get(k, '—')}")
        print()

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"[INFO] Generating figures → {fig_dir} ...")
    plot_consistency_histogram(consistency_results, display, fig_dir)
    plot_reliability_diagram(consistency_results, gt_map, display, fig_dir)
    plot_confidence_vs_correctness(consistency_results, gt_map, display, fig_dir)
    if task == "multiclass":
        plot_per_class_consistency(consistency_results, gt_map, display, fig_dir)
    print(f"[INFO] Figures saved → {fig_dir}/")

    return agg


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            "VLM consistency & uncertainty evaluation — run AFTER vlm_eval_tasks.py.\n"
            "Performs N stochastic passes per image and computes consistency metrics."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--task", required=True,
                   choices=["binary", "multiclass"],
                   help="binary or multiclass (description task not applicable)")
    p.add_argument("--model", default="all",
                   choices=list(MODEL_REGISTRY.keys()) + ["all"],
                   help="Model key or 'all' to run every model sequentially.")
    p.add_argument("--images-dir",   "-i", required=False, default=None,
                   help="Path to Data/images/ (required unless --figures-only).")
    p.add_argument("--dataset-json", "-j", required=True,
                   help="Path to Data/dataset.json (ground truth).")
    p.add_argument("--eval-dir",     "-e", default="./vlm_eval_outputs",
                   help="Root of vlm_eval_tasks.py outputs (default: ./vlm_eval_outputs).")
    p.add_argument("--n-runs",       type=int,   default=5,
                   help="Number of stochastic passes per image (default: 5).")
    p.add_argument("--temperature",  type=float, default=0.7,
                   help="Sampling temperature for stochastic runs (default: 0.7).")
    p.add_argument("--use-4bit",     action="store_true",
                   help="Load model in 4-bit quantisation.")
    p.add_argument("--delay",        type=float, default=0.05,
                   help="Seconds to sleep between images (default: 0.05).")
    p.add_argument("--preview",      type=int,   default=0,
                   help="Run on first N images only (0 = all).")
    p.add_argument("--figures-only", action="store_true",
                   help="Skip inference — regenerate figures from saved consistency_predictions.json.")
    p.add_argument("--hf-token",     type=str,   default=None,
                   help="HuggingFace token for gated models (e.g. Llama-3.2).")
    p.add_argument("--sanity-check", action="store_true",
                   help=(
                       "Run 3-image sanity check: verify predictions.json structure,\n"
                       "check consistency_predictions.json saves correctly, print file sizes."
                   ))
    args = p.parse_args()

    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("[INFO] HuggingFace auth OK.")
        except ImportError:
            print("[WARN] huggingface_hub not installed.")

    if not args.figures_only and args.images_dir is None:
        p.error("--images-dir is required unless --figures-only is set.")

    if args.sanity_check:
        args.preview = 3
        print("[SANITY CHECK] Running on 3 images only.")

    eval_dir   = Path(args.eval_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    all_agg: dict[str, dict] = {}

    for model_key in models_to_run:
        agg = run_consistency(
            task         = args.task,
            model_key    = model_key,
            images_dir   = images_dir,
            dataset_json = Path(args.dataset_json),
            eval_dir     = eval_dir,
            n_runs       = args.n_runs,
            temperature  = args.temperature,
            use_4bit     = args.use_4bit,
            delay        = args.delay,
            preview      = args.preview,
            figures_only = args.figures_only,
            hf_token     = args.hf_token,
        )
        if agg:
            all_agg[MODEL_REGISTRY[model_key]["display_name"]] = agg

    # ── Aggregate comparison figures (multi-model) ────────────────────────────
    if len(models_to_run) > 1 and all_agg:
        agg_fig_dir = eval_dir / args.task / "aggregate_consistency" / "figures"
        agg_fig_dir.mkdir(parents=True, exist_ok=True)

        ece_map   = {name: v["ece"]              for name, v in all_agg.items() if "ece"              in v}
        cons_map  = {name: v["mean_consistency"] for name, v in all_agg.items() if "mean_consistency" in v}

        plot_ece_comparison(ece_map, agg_fig_dir)
        plot_consistency_comparison(cons_map, agg_fig_dir)

        save_json(eval_dir / args.task / "aggregate_consistency" / "all_consistency_metrics.json",
                  all_agg)
        print(f"[INFO] Aggregate figures → {agg_fig_dir}/")

    if args.sanity_check:
        print("\n[SANITY CHECK] Results:")
        for model_key in models_to_run:
            out_dir  = eval_dir / args.task / model_key / "consistency"
            pred_p   = out_dir / "consistency_predictions.json"
            metric_p = out_dir / "consistency_metrics.json"
            fig_dir  = out_dir / "figures"
            print(f"\n  Model: {MODEL_REGISTRY[model_key]['display_name']}")
            if pred_p.exists():
                size = pred_p.stat().st_size / 1024
                with open(pred_p) as f:
                    data = json.load(f)
                n = len(data.get("results", []))
                print(f"    consistency_predictions.json : {size:.1f} KB  ({n} records)")
                if data.get("results"):
                    ex = data["results"][0]
                    print(f"    Example record keys : {list(ex.keys())}")
                    print(f"    n_successful_runs   : {ex.get('n_successful_runs')}")
                    print(f"    consistency_score   : {ex.get('consistency_score')}")
                    print(f"    majority_vote       : {ex.get('majority_vote')}")
                    n_all_runs = len(ex.get("all_runs", []))
                    print(f"    all_runs length     : {n_all_runs}")
            else:
                print(f"    [MISSING] {pred_p}")
            if metric_p.exists():
                size = metric_p.stat().st_size / 1024
                print(f"    consistency_metrics.json     : {size:.1f} KB")
            for fname in ["consistency_histogram", "reliability_diagram",
                          "confidence_vs_consistency"]:
                f = fig_dir / f"{fname}.png"
                if f.exists():
                    print(f"    {fname}.png : {f.stat().st_size/1024:.1f} KB  ✓")
                else:
                    print(f"    {fname}.png : MISSING")

    print(f"\n[INFO] All done. Outputs under: {eval_dir.resolve()}")


if __name__ == "__main__":
    main()
