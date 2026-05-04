"""
VLM Evaluation — Three Rigorous Tasks
======================================
Evaluates open-source Vision-Language Models on three independent tasks:

  Task 1  binary       Binary anomaly classification on the full 15K dataset.
                       Metrics: Accuracy, Balanced-Acc, Precision, Recall, F1,
                                MCC, AUROC, AUPRC + confusion matrix.

  Task 2  multiclass   11-class scene classification (10 anomaly classes + normal)
                       on the full 15K dataset.
                       Metrics: Per-class P/R/F1, Macro/Weighted-F1, Cohen's
                                Kappa, Balanced-Acc + normalised confusion matrix.

  Task 3  description  Free-text anomaly description on anomalous images only.
                       Metrics: BLEU-1/2/4, ROUGE-1/2/L, METEOR, BERTScore-P/R/F1.

Each task runs independently. A single comprehensive system prompt (no CoT) is
used: it defines all 10 anomaly classes precisely, lists what is always normal,
and prevents common classification mistakes. The model is asked for one
structured JSON output per image, per task.

Usage:
    # Run one task on one model
    python vlm_eval_tasks.py \\
        --task binary --model qwen25vl_7b \\
        --images-dir ./Data/images \\
        --dataset-json ./Data/dataset.json \\
        --output-dir ./vlm_eval_outputs

    # Run all models for a task (generates aggregate comparison figures at end)
    python vlm_eval_tasks.py --task multiclass --model all ...

    # Description task only runs on anomalous images
    python vlm_eval_tasks.py --task description --model internvl3_8b ...

    # 4-bit quantisation (saves VRAM)
    python vlm_eval_tasks.py --task binary --model llama32_11b --use-4bit ...

    # Regenerate aggregate figures from saved metrics (no inference)
    python vlm_eval_tasks.py --task binary --aggregate-only \\
        --output-dir ./vlm_eval_outputs

Requirements:
    pip install transformers accelerate torch torchvision Pillow tqdm einops
    pip install qwen-vl-utils bitsandbytes
    pip install scikit-learn matplotlib seaborn
    pip install nltk rouge-score bert-score
    python -m nltk.downloader punkt wordnet averaged_perceptron_tagger
"""

import argparse, gc, json, os, re, sys, time, warnings
from pathlib import Path
from collections import defaultdict

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
        matthews_corrcoef, roc_auc_score, average_precision_score,
        confusion_matrix, cohen_kappa_score, classification_report,
        roc_curve, precision_recall_curve,
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
    "pdf.fonttype":       42,   # embeds fonts for IEEE/ACM submission
    "ps.fonttype":        42,
})

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# =============================================================================
# TAXONOMY
# =============================================================================

NORMAL_CLASS = "normal"

ANOMALY_CLASSES = [
    "animal_on_road",
    "extreme_weather",
    "road_surface_hazard",
    "fallen_debris_or_vegetation",
    "strange_object_on_road",        # 549 samples in dataset
    "vehicle_incident",
    "infrastructure_failure",
    "human_presence_anomaly",
    "adverse_lighting",
    "oversized_or_unusual_vehicle",
    "multi_hazard_compound",
    "none_of_the_above",             # fallback — model couldn't identify class
]

# ALL_CLASSES used only for binary reference; multiclass uses ANOMALY_CLASSES only
# (multiclass runs exclusively on anomalous images — "normal" is not a valid GT label)
ALL_CLASSES = [NORMAL_CLASS] + ANOMALY_CLASSES


# =============================================================================
# PROMPTS
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
    Two or more of the above anomaly classes are SIMULTANEOUSLY
    visible in the same image (e.g. a vehicle incident AND fallen debris).

11. strange_object_on_road
    An object that is unusual, out-of-place, or unexpected for a road
    environment and does not fit neatly into the categories above: e.g.
    a shopping cart, mattress, large piece of furniture, scattered cargo,
    or any other atypical object sitting on or blocking the road surface.

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

# ── Task-specific output prompts ──────────────────────────────────────────────

BINARY_PROMPT = """\
Examine the dashcam image carefully.

Determine whether this driving scene contains a safety anomaly as defined
in the taxonomy above.

Return ONLY the following JSON — no extra text, no markdown fences:
{
  "anomaly_present": true | false,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one concise sentence stating the primary visual evidence>"
}
"""

MULTICLASS_PROMPT = f"""\
Examine the dashcam image carefully.

This image contains a safety anomaly. Classify it into exactly one of these
12 anomaly classes (do NOT use "normal" — every image here is anomalous):
  animal_on_road
  extreme_weather
  road_surface_hazard
  fallen_debris_or_vegetation
  strange_object_on_road
  vehicle_incident
  infrastructure_failure
  human_presence_anomaly
  adverse_lighting
  oversized_or_unusual_vehicle
  multi_hazard_compound
  none_of_the_above

Return ONLY the following JSON — no extra text, no markdown fences:
{{
  "anomaly_present": true,
  "scene_class": "<one of the 12 anomaly classes above>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one concise sentence stating the primary visual evidence>"
}}

Rules:
• anomaly_present is always true — every image in this task contains an anomaly.
• scene_class MUST be exactly one of the 12 classes listed above.
• Use none_of_the_above only if the anomaly does not fit any of the other 11 classes.
• Never return "normal" or any class outside the 12 listed above.
"""

DESCRIPTION_PROMPT = """\
This driving scene contains a safety anomaly.

Write a structured visual description of the anomaly. Be specific, factual,
and purely visual — describe what you can see, not what you infer.

Requirements:
• 3–5 sentences in third-person present tense.
• Start with the primary anomalous object and its location in the frame
  (e.g. "A fallen tree trunk lies across the left lane in the lower-centre
  of the frame...").
• Mention the anomaly's size, colour, and exact position relative to the
  lane markings.
• Note whether any lane is fully or partially blocked.
• Describe any secondary objects that contribute to the hazard.

Return ONLY the following JSON — no extra text, no markdown fences:
{
  "anomaly_description": "<your 3–5 sentence description here>"
}
"""


# =============================================================================
# MODEL REGISTRY  (same HF IDs as open_source_vlm.py)
# =============================================================================

MODEL_REGISTRY = {
    "internvl3_1b": {
        "hf_id":        "OpenGVLab/InternVL3-1B-hf",
        "display_name": "InternVL3-1B",
        "loader":       "internvl3",
    },
    "internvl3_2b": {
        "hf_id":        "OpenGVLab/InternVL3-2B-hf",
        "display_name": "InternVL3-2B",
        "loader":       "internvl3",
    },
    "internvl3_8b": {
        "hf_id":        "OpenGVLab/InternVL3-8B-hf",
        "display_name": "InternVL3-8B",
        "loader":       "internvl3",
    },
    "qwen2vl_2b": {
        "hf_id":        "Qwen/Qwen2-VL-2B-Instruct",
        "display_name": "Qwen2-VL-2B",
        "loader":       "qwen2vl",
    },
    "qwen25vl_3b": {
        "hf_id":        "Qwen/Qwen2.5-VL-3B-Instruct",
        "display_name": "Qwen2.5-VL-3B",
        "loader":       "qwen25vl",
    },
    "qwen25vl_7b": {
        "hf_id":        "Qwen/Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen2.5-VL-7B",
        "loader":       "qwen25vl",
    },
    "llava_onevision_7b": {
        "hf_id":        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "display_name": "LLaVA-OneVision-7B",
        "loader":       "llava_onevision",
    },
    "llava_13b": {
        "hf_id":        "llava-hf/llava-v1.6-vicuna-13b-hf",
        "display_name": "LLaVA-v1.6-13B",
        "loader":       "llava_next",
    },
    "llama32_11b": {
        "hf_id":        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "display_name": "Llama-3.2-11B",
        "loader":       "llama32",
        "note":         "Gated model — requires --hf-token.",
    },
}


# =============================================================================
# MODEL LOADERS  (identical to open_source_vlm.py — single-pass infer only)
# =============================================================================

def _patch_internvl3_cache():
    """
    Fix bugs in newly-downloaded InternVL3 custom model files:
    1. torch.linspace(...).item() called during __init__ crashes when transformers
       uses meta tensors (device_map="auto").
    2. 'all_tied_weights_keys' property missing from InternVLChatModel, required
       by transformers >=5.x accelerate integration during from_pretrained.
    Patched inline so it survives across model variants and cache refreshes.
    """
    import re as _re
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

    # Patch 1: fix meta-tensor .item() crash in modeling_intern_vit.py
    bad_item  = r"dpr\s*=\s*\[x\.item\(\)\s+for\s+x\s+in\s+torch\.linspace\([^)]+\)\]"
    good_item = ("dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1)"
                 " for i in range(config.num_hidden_layers)]")
    for fpath in hf_home.rglob("modeling_intern_vit.py"):
        text = fpath.read_text()
        if _re.search(bad_item, text):
            fpath.write_text(_re.sub(bad_item, good_item, text))
            print(f"[PATCH] Fixed meta-tensor .item() bug in {fpath}")

    # Patch 2: add missing all_tied_weights_keys property to InternVLChatModel
    # Transformers >=5.x accesses this during _init_infer_auto_device_map even
    # when device_map is not explicitly set.
    bad_tied  = "class InternVLChatModel(PreTrainedModel):"
    good_tied = ("class InternVLChatModel(PreTrainedModel):\n"
                 "    @property\n"
                 "    def all_tied_weights_keys(self):\n"
                 "        return getattr(self, '_tied_weights_keys', None) or {}\n")
    for fpath in hf_home.rglob("modeling_internvl_chat.py"):
        text = fpath.read_text()
        if bad_tied in text and "all_tied_weights_keys" not in text:
            fpath.write_text(text.replace(bad_tied, good_tied, 1))
            print(f"[PATCH] Added all_tied_weights_keys to InternVLChatModel in {fpath}")


def load_internvl3(model_id, use_4bit):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(device_map="auto", trust_remote_code=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{system}\n\n{user}"}]}]
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_qwen2vl(model_id, use_4bit):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    # Cap dynamic resolution: Qwen-VL tiles images up to 1280 patches by default.
    # Dashcam images at 512 patches (≈ 392×392 effective) are sufficient and
    # cut vision-token count by ~60 %, which speeds up prefill significantly.
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256*28*28, max_pixels=512*28*28)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{system}\n\n{user}"}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return model, processor, infer


def load_qwen25vl(model_id, use_4bit):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    # Cap dynamic resolution: same rationale as load_qwen2vl above.
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256*28*28, max_pixels=512*28*28)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{system}\n\n{user}"}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return model, processor, infer


def _patch_molmo_compat():
    """
    Molmo's custom MolmoForCausalLM lacks `all_tied_weights_keys`, which newer
    transformers/accelerate accesses inside _init_infer_auto_device_map.
    Patching that function directly is the only reliable intercept point: the
    model is already instantiated at that call but the failing line hasn't run yet.
    """
    try:
        import transformers.integrations.accelerate as _tia
        if not hasattr(_tia, "_init_infer_auto_device_map"):
            return
        _orig = _tia._init_infer_auto_device_map
        if getattr(_orig, "_molmo_patched", False):
            return  # already patched
        def _patched(model, *args, **kwargs):
            if not hasattr(model, "all_tied_weights_keys"):
                # all_tied_weights_keys must be a dict: .keys(), .values(), .items()
                # are all called on it in this version of accelerate.
                model.__class__.all_tied_weights_keys = property(
                    lambda self: {k: k for k in (getattr(self, "_tied_weights_keys", None) or [])})
            # Newer transformers calls tie_weights(missing_keys=...) but Molmo's
            # custom tie_weights() doesn't accept that argument — wrap it.
            _orig_tie = model.__class__.tie_weights
            def _tie_compat(self, missing_keys=None, **kw):
                import inspect
                sig = inspect.signature(_orig_tie)
                if "missing_keys" in sig.parameters:
                    return _orig_tie(self, missing_keys=missing_keys, **kw)
                return _orig_tie(self, **kw)
            model.__class__.tie_weights = _tie_compat
            return _orig(model, *args, **kwargs)
        _patched._molmo_patched = True
        _tia._init_infer_auto_device_map = _patched
    except Exception as e:
        print(f"[WARN] Molmo compat patch failed (may still work): {e}")


def load_molmo(model_id, use_4bit):
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    _patch_molmo_compat()
    kw = dict(trust_remote_code=True, device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    def infer(model, processor, image, system, user):
        inputs = processor.process(images=[image], text=f"{system}\n\n{user}")
        inputs = {k: v.to(_device).unsqueeze(0) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=128, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.decode(generated, skip_special_tokens=True)
    return model, processor, infer


def load_llava_next(model_id, use_4bit):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.float16
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_llava_onevision(model_id, use_4bit):
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.float16
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 repetition_penalty=1.1)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_llama32(model_id, use_4bit):
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = MllamaForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


LOADERS = {
    "internvl3":       load_internvl3,
    "qwen2vl":         load_qwen2vl,
    "qwen25vl":        load_qwen25vl,
    "molmo":           load_molmo,
    "llava_next":      load_llava_next,
    "llava_onevision": load_llava_onevision,
    "llama32":         load_llama32,
}


# =============================================================================
# UTILITIES
# =============================================================================

def load_image_pil(image_path: Path, max_size: int = 1024) -> Image.Image:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        img = img.convert("RGBA") if img.mode == "P" else img
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def find_images(input_dir: Path) -> list[Path]:
    imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        imgs.extend(input_dir.glob(ext))
    def sort_key(p):
        m = re.search(r"\d+", p.stem)
        return int(m.group()) if m else 0
    return sorted(set(imgs), key=sort_key)


def parse_json_response(raw: str) -> dict | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Grab the largest {...} block in the output
    match = re.search(r"(\{[\s\S]*\})", raw)
    if not match:
        return None
    s = match.group(1)
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    s = re.sub(r"//[^\n]*", "", s)
    # Fix unquoted true/false/null that appear as Python booleans
    s = re.sub(r":\s*True\b",  ": true",  s)
    s = re.sub(r":\s*False\b", ": false", s)
    s = re.sub(r":\s*None\b",  ": null",  s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _text_fallback_binary(raw: str) -> dict | None:
    """
    Last-resort extractor for models (e.g. Llama) that answer in prose
    instead of JSON.  Scans the raw text for clear boolean signals.
    Returns a minimal binary dict or None if no signal is found.
    """
    t = raw.lower()

    # 1. Explicit JSON field anywhere in the text
    m = re.search(r'"anomaly_present"\s*:\s*(true|false)', t)
    if m:
        val = m.group(1) == "true"
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
        conf = float(conf_m.group(1)) if conf_m else (0.8 if val else 0.2)
        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]{5,})"', raw, re.IGNORECASE)
        reason = reason_m.group(1) if reason_m else raw.strip()[:120]
        return {"anomaly_present": val, "confidence": conf, "reasoning": reason}

    # 2. Strong negative phrases → normal
    no_anomaly = [
        "no anomaly", "no safety anomaly", "not anomalous", "is normal",
        "normal driving", "no hazard", "no incident", "nothing unusual",
        "no abnormality", "scene is safe", "false", "anomaly_present.*false",
    ]
    if any(re.search(p, t) for p in no_anomaly):
        reason = raw.strip()[:120]
        return {"anomaly_present": False, "confidence": 0.5, "reasoning": reason}

    # 3. Strong positive phrases → anomaly
    yes_anomaly = [
        "anomaly is present", "anomaly present", "is anomalous",
        "safety anomaly", "hazard detected", "anomaly detected",
        "true", "anomaly_present.*true",
    ]
    if any(re.search(p, t) for p in yes_anomaly):
        reason = raw.strip()[:120]
        return {"anomaly_present": True, "confidence": 0.5, "reasoning": reason}

    return None


def _text_fallback_multiclass(raw: str) -> dict | None:
    """
    Last-resort extractor for multiclass task.  Tries to find scene_class
    or anomaly_present from prose output.
    """
    t = raw.lower()

    # Try to find scene_class field
    m = re.search(r'"scene_class"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    if m:
        cls = m.group(1).strip().lower().replace(" ", "_")
        if cls in ANOMALY_CLASSES:
            conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
            conf = float(conf_m.group(1)) if conf_m else 0.5
            return {"anomaly_present": True, "scene_class": cls,
                    "confidence": conf, "reasoning": raw.strip()[:120]}

    # Try to match any anomaly class name mentioned in the text
    for cls in ANOMALY_CLASSES:
        if cls.replace("_", " ") in t or cls in t:
            return {"anomaly_present": True, "scene_class": cls,
                    "confidence": 0.4, "reasoning": raw.strip()[:120]}

    # Last resort: model gave some signal but no class — use first anomaly class
    # as a wrong-but-valid placeholder so the record is still counted as evaluated
    binary = _text_fallback_binary(raw)
    if binary:
        return {"anomaly_present": True, "scene_class": ANOMALY_CLASSES[0],
                "confidence": 0.2, "reasoning": raw.strip()[:120]}

    return None


def load_ground_truth(dataset_json: Path) -> dict:
    """Returns the full dataset.json 'samples' dict keyed by filename."""
    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", {})


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# REPRESENTATION EXTRACTION
# =============================================================================
# Saves per-layer vision-encoder hidden states (mean-pooled over patches) and
# the final projected representation (the embedding the LLM actually sees).
#
# Stored per image as compressed numpy archives:
#   {task_dir}/representations/{image_id}.npz
#     "layers"    → float16 array  (n_vision_layers, vis_hidden_dim)
#     "final_rep" → float16 array  (llm_hidden_dim,)   [if projection found]
#
# These are the inputs for:
#   Idea 1 — layer-wise linear probing (which layer encodes anomaly best?)
#   Idea 2 — activation steering (anomaly direction vector)
#   Idea 3 — representation-based anomaly detection (frozen features + MLP)
# =============================================================================

def _extract_hidden(output) -> "torch.Tensor | None":
    """Pull the primary hidden-state tensor out of any layer output type."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return None


def _mean_pool(t: "torch.Tensor") -> "np.ndarray":
    """
    Mean-pool any vision-encoder output to a 1-D float16 numpy vector.
    Handles all shapes seen across model families:
      (batch, seq, dim)  → mean over seq → squeeze batch → (dim,)
      (seq, dim)         → mean over seq               → (dim,)
      (batch, dim)       → squeeze batch               → (dim,)
      (dim,)             → passthrough
      >3-D               → flatten all leading dims, mean → (dim,)
    """
    t = t.detach().float()
    # Collapse any leading batch / extra dims down to at most 3-D
    while t.ndim > 3:
        t = t.flatten(0, 1)     # e.g. (1,1,4,seq,dim) → (4,seq,dim) → …
    if t.ndim == 3:
        t = t.mean(dim=1)       # (batch, seq, dim) → (batch, dim)
    if t.ndim == 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)    # (1, dim) → (dim,)
        else:
            t = t.mean(dim=0)   # (seq, dim) with no batch wrapper → (dim,)
    return t.cpu().numpy().astype(np.float16)


def _find_vision_layers(model) -> list:
    """
    Locate the list of vision transformer blocks across different VLM families.
    Returns an empty list if the architecture is not recognised.
    """
    # Ordered from most-specific to least-specific
    ATTR_PATHS = [
        # LLaVA-Next / LLaVA-OneVision
        "vision_tower.vision_model.encoder.layers",
        "model.vision_tower.vision_model.encoder.layers",
        # Qwen2-VL, Qwen2.5-VL
        "visual.blocks",
        "model.visual.blocks",
        # InternVL3 (HF version)
        "vision_model.encoder.layers",
        "model.vision_model.encoder.layers",
        # Llama-3.2-Vision
        "vision_model.model.layers",
        "model.vision_model.model.layers",
        # Molmo
        "vision_backbone.image_vit.transformer.resblocks",
        "model.vision_backbone.image_vit.transformer.resblocks",
    ]
    for path in ATTR_PATHS:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 2:
                return list(obj)
        except AttributeError:
            continue

    # Generic fallback: find the first large ModuleList with a vision-related name
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        if len(module) < 4:
            continue
        if any(kw in name.lower() for kw in ("visual", "vision", "img", "patch", "vit")):
            return list(module)

    return []


def _find_vision_projection(model) -> "torch.nn.Module | None":
    """Find the linear/MLP projection that maps vision features into LLM space."""
    ATTR_PATHS = [
        "multi_modal_projector",
        "model.multi_modal_projector",
        "mm_projector",
        "model.mm_projector",
        "visual.merger",
        "model.visual.merger",
        "language_projection",
        "model.language_projection",
        "mlp1",          # InternVL
        "model.mlp1",
    ]
    for path in ATTR_PATHS:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.Module):
                return obj
        except AttributeError:
            continue
    return None


class ActivationExtractor:
    """
    Registers forward hooks on a loaded VLM to collect:
      - mean-pooled hidden states at each vision-encoder layer
      - output of the vision→language projection (final_rep)

    Usage inside the inference loop:
        extractor.reset()
        result = infer_fn_task(...)       # forward pass fires hooks
        layers, final = extractor.collect()
    """

    def __init__(self, model):
        self._buffer: list = []        # appended by each layer hook
        self.final_rep = None          # set by projection hook
        self.hooks:  list = []
        self.n_layers: int = 0

        vision_layers = _find_vision_layers(model)
        for layer in vision_layers:
            h = layer.register_forward_hook(self._layer_hook)
            self.hooks.append(h)
        self.n_layers = len(vision_layers)

        proj = _find_vision_projection(model)
        if proj is not None:
            h = proj.register_forward_hook(self._proj_hook)
            self.hooks.append(h)

        if self.n_layers > 0:
            print(f"[REPR] Hooked {self.n_layers} vision layers"
                  f"{' + projection' if proj is not None else ' (no projection found)'}.")
        else:
            print("[REPR] No vision layers found — representations will NOT be saved.")

    def _layer_hook(self, module, input, output):
        h = _extract_hidden(output)
        if h is not None:
            self._buffer.append(_mean_pool(h))

    def _proj_hook(self, module, input, output):
        h = _extract_hidden(output)
        if h is not None:
            rep = _mean_pool(h)
            # Some projections emit (seq, dim) → further pool if needed
            if rep.ndim == 2:
                rep = rep.mean(axis=0)
            self.final_rep = rep

    def reset(self):
        """Call before each image's forward pass to clear stale activations."""
        self._buffer.clear()
        self.final_rep = None

    def collect(self):
        """
        Return (layer_array, final_rep) from the most recent forward pass and reset.
        layer_array : float16 ndarray (n_layers, hidden_dim)  or None
        final_rep   : float16 ndarray (llm_hidden_dim,)       or None
        """
        if not self._buffer:
            return None, None

        # Take only the LAST n_layers entries to handle retry-induced duplicates
        last = self._buffer[-self.n_layers:] if self.n_layers > 0 else self._buffer
        try:
            layers = np.stack(last, axis=0)   # (n_layers, hidden_dim)
        except ValueError:
            # Mismatched dims across layers (rare) — skip
            layers = None

        final = self.final_rep
        self.reset()
        return layers, final

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def save_representation(repr_dir: Path, image_id: str, layers, final_rep):
    """Save per-image representation to a compressed .npz file."""
    data = {}
    if layers is not None:
        data["layers"] = layers.astype(np.float16)
    if final_rep is not None:
        data["final_rep"] = final_rep.astype(np.float16)
    if data:
        np.savez_compressed(repr_dir / f"{image_id}.npz", **data)


def consolidate_representations(task_dir: Path):
    """
    Aggregates all per-image .npz files into two consolidated arrays:
      all_layer_representations.npz  — keys: image_id, values: (n_layers, dim)
      all_final_representations.npz  — keys: image_id, values: (dim,)

    Run this after inference is complete to prepare inputs for probing/steering.
    """
    repr_dir = task_dir / "representations"
    files = sorted(repr_dir.glob("data_*.npz"))
    if not files:
        print(f"[REPR] No .npz files found in {repr_dir}")
        return

    print(f"[REPR] Consolidating {len(files):,} representation files ...")
    all_layers: dict[str, np.ndarray] = {}
    all_final:  dict[str, np.ndarray] = {}

    for f in tqdm(files, desc="Loading", unit="file"):
        data = np.load(f)
        iid  = f.stem
        if "layers"    in data: all_layers[iid] = data["layers"]
        if "final_rep" in data: all_final[iid]  = data["final_rep"]

    if all_layers:
        out = repr_dir / "all_layer_representations.npz"
        np.savez_compressed(out, **all_layers)
        print(f"[REPR] Layer representations → {out}  ({len(all_layers):,} images)")

    if all_final:
        out = repr_dir / "all_final_representations.npz"
        np.savez_compressed(out, **all_final)
        print(f"[REPR] Final representations  → {out}  ({len(all_final):,} images)")


# =============================================================================
# LATENCY HELPER
# =============================================================================

def _latency_stats(latencies: list[float]) -> dict:
    """Returns mean / median / p95 latency in seconds, or empty dict if no data."""
    if not latencies:
        return {}
    arr = np.array(latencies, dtype=float)
    return {
        "mean_s":   round(float(arr.mean()),                4),
        "median_s": round(float(np.median(arr)),            4),
        "p95_s":    round(float(np.percentile(arr, 95)),    4),
        "min_s":    round(float(arr.min()),                 4),
        "max_s":    round(float(arr.max()),                 4),
        "n":        len(latencies),
    }


# =============================================================================
# TASK 1 — BINARY INFERENCE
# =============================================================================

def infer_binary(model, processor, infer_fn, image_path: Path, model_key: str, retry: int = 2) -> dict:
    image_id = image_path.stem
    filename = image_path.name
    for attempt in range(1, retry + 1):
        try:
            image = load_image_pil(image_path)
            t0  = time.perf_counter()
            raw = infer_fn(model, processor, image, SYSTEM_PROMPT, BINARY_PROMPT)
            latency = round(time.perf_counter() - t0, 4)
            parsed = parse_json_response(raw)
            if parsed is None:
                parsed = _text_fallback_binary(raw)
            if parsed is None:
                if attempt < retry:
                    time.sleep(1); continue
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": "JSON parse failed", "raw": raw[:300]}
            return {
                "status":          "success",
                "image_id":        image_id,
                "filename":        filename,
                "model":           model_key,
                "anomaly_present": bool(parsed.get("anomaly_present", False)),
                "confidence":      float(parsed.get("confidence", 0.5)),
                "reasoning":       str(parsed.get("reasoning", "")),
                "latency_s":       latency,
            }
        except Exception as e:
            if attempt == retry:
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": str(e)}
            time.sleep(2)
    return {"status": "failed", "image_id": image_id, "filename": filename,
            "model": model_key, "error": "max retries"}


# =============================================================================
# TASK 2 — MULTICLASS INFERENCE
# =============================================================================

def infer_multiclass(model, processor, infer_fn, image_path: Path, model_key: str, retry: int = 2) -> dict:
    image_id = image_path.stem
    filename = image_path.name
    for attempt in range(1, retry + 1):
        try:
            image = load_image_pil(image_path)
            t0  = time.perf_counter()
            raw = infer_fn(model, processor, image, SYSTEM_PROMPT, MULTICLASS_PROMPT)
            latency = round(time.perf_counter() - t0, 4)
            parsed = parse_json_response(raw)
            if parsed is None:
                parsed = _text_fallback_multiclass(raw)
            if parsed is None:
                if attempt < retry:
                    time.sleep(1); continue
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": "JSON parse failed", "raw": raw[:300]}
            # Normalise scene_class — valid values are the 11 ANOMALY_CLASSES only.
            # The multiclass task runs exclusively on anomalous images so "normal"
            # is never a correct answer. If the model returns "normal" or something
            # unrecognised, try to recover from the raw text before giving up.
            raw_class = str(parsed.get("scene_class", "")).strip().lower().replace(" ", "_").replace(" ", "_")
            if raw_class not in ANOMALY_CLASSES:
                # Try fuzzy match: find any anomaly class name mentioned in the raw output
                recovered = None
                raw_lower = raw.lower()
                for cls in ANOMALY_CLASSES:
                    if cls == "none_of_the_above":
                        continue
                    if cls.replace("_", " ") in raw_lower or cls in raw_lower:
                        recovered = cls
                        break
                raw_class = recovered if recovered else "none_of_the_above"
            return {
                "status":          "success",
                "image_id":        image_id,
                "filename":        filename,
                "model":           model_key,
                "anomaly_present": bool(parsed.get("anomaly_present", False)),
                "scene_class":     raw_class,
                "confidence":      float(parsed.get("confidence", 0.5)),
                "reasoning":       str(parsed.get("reasoning", "")),
                "latency_s":       latency,
            }
        except Exception as e:
            if attempt == retry:
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": str(e)}
            time.sleep(2)
    return {"status": "failed", "image_id": image_id, "filename": filename,
            "model": model_key, "error": "max retries"}


# =============================================================================
# TASK 3 — DESCRIPTION INFERENCE
# =============================================================================

def infer_description(model, processor, infer_fn, image_path: Path, model_key: str, retry: int = 2) -> dict:
    image_id = image_path.stem
    filename = image_path.name
    for attempt in range(1, retry + 1):
        try:
            image = load_image_pil(image_path)
            t0  = time.perf_counter()
            raw = infer_fn(model, processor, image, SYSTEM_PROMPT, DESCRIPTION_PROMPT)
            latency = round(time.perf_counter() - t0, 4)
            parsed = parse_json_response(raw)
            desc = None
            if parsed:
                desc = parsed.get("anomaly_description") or parsed.get("description")
            if not desc:
                # Fallback: use the raw text if it's not JSON
                desc = raw.strip()
            return {
                "status":              "success",
                "image_id":            image_id,
                "filename":            filename,
                "model":               model_key,
                "anomaly_description": desc,
                "latency_s":           latency,
            }
        except Exception as e:
            if attempt == retry:
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": str(e)}
            time.sleep(2)
    return {"status": "failed", "image_id": image_id, "filename": filename,
            "model": model_key, "error": "max retries"}


# =============================================================================
# METRICS — BINARY
# =============================================================================

def compute_binary_metrics(predictions: list[dict], gt_map: dict) -> dict:
    rows = [p for p in predictions if p.get("status") == "success" and p["filename"] in gt_map]
    if not rows:
        return {}

    y_true  = [int(gt_map[r["filename"]].get("anomaly_present", False)) for r in rows]
    y_pred  = [int(r["anomaly_present"]) for r in rows]
    y_score = [r.get("confidence", 0.5) if r["anomaly_present"]
               else 1 - r.get("confidence", 0.5) for r in rows]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    latencies = [r["latency_s"] for r in rows if "latency_s" in r]
    return {
        "n_evaluated":        len(rows),
        "accuracy":           round(accuracy_score(y_true, y_pred),           4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_true, y_pred),  4),
        "precision":          round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":             round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":                 round(f1_score(y_true, y_pred, zero_division=0),        4),
        "mcc":                round(matthews_corrcoef(y_true, y_pred),                4),
        "auroc":              round(roc_auc_score(y_true, y_score),                   4),
        "auprc":              round(average_precision_score(y_true, y_score),         4),
        "confusion_matrix":   cm.tolist(),
        "class_distribution": {
            "gt_normal":        int(sum(1 for v in y_true if v == 0)),
            "gt_anomalous":     int(sum(1 for v in y_true if v == 1)),
            "pred_normal":      int(sum(1 for v in y_pred if v == 0)),
            "pred_anomalous":   int(sum(1 for v in y_pred if v == 1)),
        },
        "latency": _latency_stats(latencies),
        "_raw": {"y_true": y_true, "y_pred": y_pred, "y_score": y_score},
    }


# =============================================================================
# METRICS — MULTICLASS
# =============================================================================

def compute_multiclass_metrics(predictions: list[dict], gt_map: dict) -> dict:
    rows = [p for p in predictions if p.get("status") == "success" and p["filename"] in gt_map]
    if not rows:
        return {}

    # Multiclass runs only on anomalous images → label space is ANOMALY_CLASSES only.
    # "normal" is never a valid GT label here. "unknown" GT records are excluded.
    def gt_class(rec):
        cls = rec.get("anomaly_class", "")
        return cls if cls in ANOMALY_CLASSES else None   # None → exclude row

    # Build (gt, pred) pairs, dropping rows with unknown GT class
    pairs = []
    for r in rows:
        gt = gt_class(gt_map[r["filename"]])
        if gt is None:
            continue
        pred = r.get("scene_class", "")
        # If model predicted "normal" or an unrecognised class, map to least-likely
        # anomaly class so it counts as wrong without distorting label set
        if pred not in ANOMALY_CLASSES:
            pred = ANOMALY_CLASSES[0]
        pairs.append((gt, pred))

    if not pairs:
        return {}

    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]

    cm = confusion_matrix(y_true, y_pred, labels=ANOMALY_CLASSES)
    cm_float = cm.astype(float)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    # Use safe division: rows with zero support → all zeros (not garbage floats)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm_float / np.where(row_sums > 0, row_sums, 1.0), 0.0)

    report = classification_report(
        y_true, y_pred, labels=ANOMALY_CLASSES, zero_division=0, output_dict=True)

    latencies = [r["latency_s"] for r in rows if "latency_s" in r]
    return {
        "n_evaluated":       len(pairs),
        "n_excluded_unknown": len(rows) - len(pairs),
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1":          round(f1_score(y_true, y_pred, labels=ANOMALY_CLASSES, average="macro",    zero_division=0), 4),
        "weighted_f1":       round(f1_score(y_true, y_pred, labels=ANOMALY_CLASSES, average="weighted", zero_division=0), 4),
        "macro_precision":   round(precision_score(y_true, y_pred, labels=ANOMALY_CLASSES, average="macro",    zero_division=0), 4),
        "macro_recall":      round(recall_score(y_true, y_pred,    labels=ANOMALY_CLASSES, average="macro",    zero_division=0), 4),
        "cohen_kappa":       round(cohen_kappa_score(y_true, y_pred), 4),
        "per_class":         {k: v for k, v in report.items()
                              if isinstance(v, dict) and k in ANOMALY_CLASSES},
        "confusion_matrix":        cm.tolist(),
        "confusion_matrix_norm":   cm_norm.tolist(),
        "confusion_matrix_labels": ANOMALY_CLASSES,
        "latency": _latency_stats(latencies),
        "_raw": {"y_true": y_true, "y_pred": y_pred},
    }


# =============================================================================
# METRICS — DESCRIPTION
# =============================================================================

def compute_description_metrics(predictions: list[dict], gt_map: dict) -> dict:
    # Lazy imports so the script works even if some are missing
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        nltk.download("punkt",                     quiet=True)
        nltk.download("punkt_tab",                 quiet=True)
        nltk.download("wordnet",                   quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        has_nltk = True
    except ImportError:
        has_nltk = False
        print("[WARN] nltk not found — BLEU/METEOR skipped. pip install nltk")

    try:
        from rouge_score import rouge_scorer as rouge_lib
        has_rouge = True
    except ImportError:
        has_rouge = False
        print("[WARN] rouge-score not found — ROUGE skipped. pip install rouge-score")

    try:
        from bert_score import score as bert_score_fn
        has_bert = True
    except ImportError:
        has_bert = False
        print("[WARN] bert-score not found — BERTScore skipped. pip install bert-score")

    rows = [p for p in predictions
            if p.get("status") == "success"
            and p["filename"] in gt_map
            and gt_map[p["filename"]].get("anomaly_present")
            and gt_map[p["filename"]].get("anomaly_description")]
    if not rows:
        return {}

    hypotheses = [r.get("anomaly_description", "") or "" for r in rows]
    references  = [gt_map[r["filename"]]["anomaly_description"]  for r in rows]

    results = {"n_evaluated": len(rows), "total_images": len(rows)}

    # ── BLEU ─────────────────────────────────────────────────────────────────
    if has_nltk:
        from nltk.tokenize import word_tokenize
        smooth = SmoothingFunction().method1
        bleu1 = bleu2 = bleu4 = 0.0
        meteor_scores = []
        for hyp, ref in zip(hypotheses, references):
            hyp_tok = word_tokenize(hyp.lower())
            ref_tok = [word_tokenize(ref.lower())]
            bleu1 += sentence_bleu(ref_tok, hyp_tok, weights=(1, 0, 0, 0),     smoothing_function=smooth)
            bleu2 += sentence_bleu(ref_tok, hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            bleu4 += sentence_bleu(ref_tok, hyp_tok, weights=(0.25,)*4,        smoothing_function=smooth)
            try:
                meteor_scores.append(nltk_meteor([ref_tok[0]], hyp_tok))
            except Exception:
                meteor_scores.append(0.0)
        n = len(rows)
        results["bleu_1"]  = round(bleu1 / n, 4)
        results["bleu_2"]  = round(bleu2 / n, 4)
        results["bleu_4"]  = round(bleu4 / n, 4)
        results["meteor"]  = round(sum(meteor_scores) / n, 4)

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    if has_rouge:
        scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1 = r2 = rL = 0.0
        for hyp, ref in zip(hypotheses, references):
            s = scorer.score(ref, hyp)
            r1 += s["rouge1"].fmeasure
            r2 += s["rouge2"].fmeasure
            rL += s["rougeL"].fmeasure
        n = len(rows)
        results["rouge_1"] = round(r1 / n, 4)
        results["rouge_2"] = round(r2 / n, 4)
        results["rouge_L"] = round(rL / n, 4)

    # ── BERTScore ─────────────────────────────────────────────────────────────
    if has_bert:
        # Replace empty predictions with a single period to avoid BERTScore crash
        hyps_safe = [h if h.strip() else "." for h in hypotheses]
        try:
            P, R, F = bert_score_fn(hyps_safe, references,
                                    lang="en", model_type="distilbert-base-uncased",
                                    verbose=False)
            results["bertscore_precision"] = round(P.mean().item(), 4)
            results["bertscore_recall"]    = round(R.mean().item(), 4)
            results["bertscore_f1"]        = round(F.mean().item(), 4)
        except Exception as e:
            print(f"[WARN] BERTScore failed: {e}")

    latencies = [r["latency_s"] for r in rows if "latency_s" in r]
    results["latency"] = _latency_stats(latencies)
    return results


# =============================================================================
# FIGURES — BINARY
# =============================================================================

def plot_binary_single(metrics: dict, display_name: str, out_dir: Path):
    """ROC curve + confusion matrix for one model."""
    raw = metrics.get("_raw", {})
    if not raw:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    y_true  = raw["y_true"]
    y_score = raw["y_score"]

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
            label=f"AUROC = {metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {display_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.savefig(out_dir / "roc_curve.pdf")
    fig.savefig(out_dir / "roc_curve.png")
    plt.close(fig)

    # ── PR curve ──────────────────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, color=PALETTE[1], lw=2,
            label=f"AUPRC = {metrics['auprc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {display_name}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.savefig(out_dir / "pr_curve.pdf")
    fig.savefig(out_dir / "pr_curve.png")
    plt.close(fig)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomalous"],
                yticklabels=["Normal", "Anomalous"], ax=ax,
                linewidths=0.5, linecolor="gray", cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix — {display_name}")
    fig.savefig(out_dir / "confusion_matrix.pdf")
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)


def plot_binary_aggregate(all_metrics: dict[str, dict], out_dir: Path):
    """ROC, PR, and metrics bar chart for all models on one canvas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names   = list(all_metrics.keys())
    metric_labels = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
                     "mcc", "auroc", "auprc"]

    # ── Overlay ROC curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, m) in enumerate(all_metrics.items()):
        raw = m.get("_raw", {})
        if not raw: continue
        fpr, tpr, _ = roc_curve(raw["y_true"], raw["y_score"])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.8,
                label=f"{name} ({m.get('auroc', 0):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.savefig(out_dir / "roc_all_models.pdf")
    fig.savefig(out_dir / "roc_all_models.png")
    plt.close(fig)

    # ── Overlay PR curves ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, m) in enumerate(all_metrics.items()):
        raw = m.get("_raw", {})
        if not raw: continue
        prec, rec, _ = precision_recall_curve(raw["y_true"], raw["y_score"])
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=1.8,
                label=f"{name} ({m.get('auprc', 0):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — All Models")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.savefig(out_dir / "pr_all_models.pdf")
    fig.savefig(out_dir / "pr_all_models.png")
    plt.close(fig)

    # ── Grouped bar chart: key metrics ───────────────────────────────────────
    key_metrics = ["accuracy", "f1", "auroc", "auprc"]
    x      = np.arange(len(model_names))
    width  = 0.2
    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 1.1), 4.5))
    for j, metric in enumerate(key_metrics):
        vals = [all_metrics[n].get(metric, 0) for n in model_names]
        bars = ax.bar(x + j * width, vals, width, label=metric.upper(),
                      color=PALETTE[j], alpha=0.85)
    ax.set_xticks(x + width * (len(key_metrics) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Binary Classification — Model Comparison")
    ax.legend(ncol=4, fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    fig.savefig(out_dir / "metrics_comparison.pdf")
    fig.savefig(out_dir / "metrics_comparison.png")
    plt.close(fig)


# =============================================================================
# FIGURES — MULTICLASS
# =============================================================================

def plot_multiclass_single(metrics: dict, display_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalised confusion matrix ───────────────────────────────────────────
    cm_norm = np.array(metrics.get("confusion_matrix_norm", []))
    labels  = metrics.get("confusion_matrix_labels", ALL_CLASSES)
    if cm_norm.size > 0:
        short_labels = [l.replace("_", "\n") for l in labels]
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=short_labels, yticklabels=short_labels,
                    ax=ax, linewidths=0.3, linecolor="gray",
                    vmin=0, vmax=1, annot_kws={"size": 8})
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        ax.set_title(f"Normalised Confusion Matrix — {display_name}", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        fig.savefig(out_dir / "confusion_matrix_norm.pdf")
        fig.savefig(out_dir / "confusion_matrix_norm.png")
        plt.close(fig)

    # ── Per-class F1 bar chart ────────────────────────────────────────────────
    per_class = metrics.get("per_class", {})
    if per_class:
        classes = [c for c in ALL_CLASSES if c in per_class]
        f1_vals = [per_class[c].get("f1-score", 0) for c in classes]
        short   = [c.replace("_", "\n") for c in classes]
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(classes)), f1_vals, color=PALETTE[:len(classes)], alpha=0.85)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short, fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Per-Class F1 — {display_name}")
        ax.yaxis.grid(True, alpha=0.4)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        fig.savefig(out_dir / "per_class_f1.pdf")
        fig.savefig(out_dir / "per_class_f1.png")
        plt.close(fig)


def plot_multiclass_aggregate(all_metrics: dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_metrics.keys())

    # ── Macro metrics grouped bar chart ───────────────────────────────────────
    key_metrics = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    x     = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 1.2), 4.5))
    for j, metric in enumerate(key_metrics):
        vals = [all_metrics[n].get(metric, 0) for n in model_names]
        ax.bar(x + j * width, vals, width,
               label=metric.replace("_", " ").title(),
               color=PALETTE[j], alpha=0.85)
    ax.set_xticks(x + width * (len(key_metrics) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Multi-class Classification — Model Comparison")
    ax.legend(ncol=4, fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    fig.savefig(out_dir / "metrics_comparison.pdf")
    fig.savefig(out_dir / "metrics_comparison.png")
    plt.close(fig)

    # ── Per-class F1 heatmap across models ────────────────────────────────────
    per_class_data = {}
    for model, m in all_metrics.items():
        for cls in ANOMALY_CLASSES:
            val = m.get("per_class", {}).get(cls, {}).get("f1-score", 0)
            per_class_data.setdefault(cls, {})[model] = val

    matrix = np.array([[per_class_data.get(cls, {}).get(m, 0)
                         for m in model_names] for cls in ANOMALY_CLASSES])
    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=model_names, yticklabels=ANOMALY_CLASSES,
                ax=ax, linewidths=0.3, vmin=0, vmax=1, annot_kws={"size": 8})
    ax.set_title("Per-Class F1 — All Models", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.savefig(out_dir / "per_class_f1_heatmap.pdf")
    fig.savefig(out_dir / "per_class_f1_heatmap.png")
    plt.close(fig)


# =============================================================================
# FIGURES — DESCRIPTION
# =============================================================================

def plot_description_aggregate(all_metrics: dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_metrics.keys())
    key_metrics = ["bleu_4", "rouge_L", "meteor", "bertscore_f1"]
    labels_map  = {"bleu_4": "BLEU-4", "rouge_L": "ROUGE-L",
                   "meteor": "METEOR", "bertscore_f1": "BERTScore-F1"}

    present = [m for m in key_metrics if any(m in all_metrics[n] for n in model_names)]
    if not present:
        return

    x     = np.arange(len(model_names))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 1.2), 4.5))
    for j, metric in enumerate(present):
        vals = [all_metrics[n].get(metric, 0) for n in model_names]
        ax.bar(x + j * width, vals, width,
               label=labels_map.get(metric, metric),
               color=PALETTE[j], alpha=0.85)
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim([0, max(
        max(all_metrics[n].get(m, 0) for n in model_names) for m in present
    ) * 1.15 + 0.01])
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Description — Text Similarity Metrics")
    ax.legend(ncol=len(present), fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    fig.savefig(out_dir / "description_metrics.pdf")
    fig.savefig(out_dir / "description_metrics.png")
    plt.close(fig)


# =============================================================================
# PIPELINE
# =============================================================================

TASK_INFER_FN = {
    "binary":      infer_binary,
    "multiclass":  infer_multiclass,
    "description": infer_description,
}

TASK_METRICS_FN = {
    "binary":      compute_binary_metrics,
    "multiclass":  compute_multiclass_metrics,
    "description": compute_description_metrics,
}

TASK_PLOT_FN = {
    "binary":     (plot_binary_single,     plot_binary_aggregate),
    "multiclass": (plot_multiclass_single, plot_multiclass_aggregate),
    "description":(None,                  plot_description_aggregate),
}


def _print_sample(task: str, result: dict, gt_map: dict, n_ok: int, n_fail: int, total: int):
    """Print a one- or two-line GT vs prediction summary for a single image."""
    filename = result.get("filename", "?")
    image_id = result.get("image_id", filename)
    gt_rec   = gt_map.get(filename, {})

    if result["status"] != "success":
        print(f"  [{n_ok+n_fail:>5}/{total}] ✗ {image_id}  ERROR: {result.get('error','')[:60]}")
        return

    if task == "binary":
        gt_label   = "ANOMALY" if gt_rec.get("anomaly_present") else "NORMAL "
        pred_label = "ANOMALY" if result.get("anomaly_present") else "NORMAL "
        match      = "✓" if gt_label == pred_label else "✗"
        conf       = result.get("confidence", 0.0)
        reason     = result.get("reasoning", "")[:70]
        print(f"  [{n_ok+n_fail:>5}/{total}] {match}  {image_id:<16} "
              f"GT={gt_label}  PRED={pred_label}  conf={conf:.2f}")
        print(f"           reason: {reason}")

    elif task == "multiclass":
        gt_cls   = gt_rec.get("anomaly_class", NORMAL_CLASS) if gt_rec.get("anomaly_present") else NORMAL_CLASS
        pred_cls = result.get("scene_class", NORMAL_CLASS)
        match    = "✓" if gt_cls == pred_cls else "✗"
        conf     = result.get("confidence", 0.0)
        reason   = result.get("reasoning", "")[:70]
        print(f"  [{n_ok+n_fail:>5}/{total}] {match}  {image_id:<16} "
              f"GT={gt_cls:<32}  PRED={pred_cls:<32}  conf={conf:.2f}")
        print(f"           reason: {reason}")

    elif task == "description":
        gt_desc   = (gt_rec.get("anomaly_description") or "")[:100]
        pred_desc = (result.get("anomaly_description")  or "")[:100]
        print(f"  [{n_ok+n_fail:>5}/{total}]    {image_id:<16}")
        print(f"           GT  : {gt_desc}")
        print(f"           PRED: {pred_desc}")


def run_model_task(
    task:         str,
    model_key:    str,
    images:       list[Path],
    gt_map:       dict,
    output_dir:   Path,
    use_4bit:     bool,
    delay:        float,
    start_from:   int,
):
    cfg          = MODEL_REGISTRY[model_key]
    display      = cfg["display_name"]
    infer_fn_task = TASK_INFER_FN[task]

    task_dir  = output_dir / task / model_key
    task_dir.mkdir(parents=True, exist_ok=True)
    pred_path = task_dir / "predictions.json"

    print(f"\n{'='*65}")
    print(f"  Task        : {task.upper()}")
    print(f"  Model       : {display}")
    print(f"  Output dir  : {task_dir}")
    if "note" in cfg:
        print(f"  NOTE        : {cfg['note']}")
    print(f"{'='*65}")

    # ── Filter images for task ────────────────────────────────────────────────
    # Both multiclass and description run only on anomalous images:
    #   multiclass  — classifying WHICH of the 10 anomaly classes is present
    #   description — describing the anomaly in free text
    # Binary runs on the full 15K dataset.
    if task in ("multiclass", "description"):
        images = [p for p in images
                  if gt_map.get(p.name, {}).get("anomaly_present")]
        print(f"[INFO] {task.capitalize()} task: {len(images):,} anomalous images selected.")

    # ── Resume support ────────────────────────────────────────────────────────
    results       = []
    processed_ids = set()
    if pred_path.exists():
        with open(pred_path) as f:
            existing = json.load(f)
        results = existing.get("predictions", [])
        processed_ids = {r.get("image_id") for r in results}
        if processed_ids:
            print(f"[INFO] Resuming: {len(results):,} already done.")

    def img_idx(p):
        m = re.search(r"\d+", p.stem)
        return int(m.group()) if m else 0

    images_to_run = [
        p for p in images
        if p.stem not in processed_ids and img_idx(p) >= start_from
    ]
    if not images_to_run:
        print(f"[INFO] All images already processed for {display}/{task}.")
    else:
        # ── Load model ────────────────────────────────────────────────────────
        model, processor, infer_fn = LOADERS[cfg["loader"]](cfg["hf_id"], use_4bit)

        # ── Set up activation extractor ───────────────────────────────────────
        extractor  = ActivationExtractor(model)
        repr_dir   = task_dir / "representations"
        repr_dir.mkdir(parents=True, exist_ok=True)
        save_repr  = extractor.n_layers > 0

        # Track which image_ids already have representation files (resume support)
        existing_repr = {f.stem for f in repr_dir.glob("data_*.npz")} if save_repr else set()

        metadata = {
            "task":        task,
            "model_key":   model_key,
            "model_hf_id": cfg["hf_id"],
            "display":     display,
            "quantisation": "4bit" if use_4bit else "bf16/fp16",
            "started_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repr_n_layers": extractor.n_layers,
        }

        n_ok = n_fail = n_repr = 0
        total = len(images_to_run)
        for image_path in tqdm(images_to_run, desc=f"{display}/{task}"):
            # Reset extractor before each image so retries don't leave stale activations
            extractor.reset()

            result = infer_fn_task(model, processor, infer_fn, image_path, model_key)
            results.append(result)

            if result["status"] == "success":
                n_ok += 1
                # ── Save representations (skip if already saved from a prior run) ──
                if save_repr and result["image_id"] not in existing_repr:
                    layers, final_rep = extractor.collect()
                    if layers is not None:
                        save_representation(repr_dir, result["image_id"], layers, final_rep)
                        existing_repr.add(result["image_id"])
                        n_repr += 1
            else:
                n_fail += 1
                extractor.reset()   # discard any partial activations

            _print_sample(task, result, gt_map, n_ok, n_fail, total)

            # Incremental save (crash-safe)
            save_json(pred_path, {"metadata": metadata, "predictions": results})
            time.sleep(delay)

        metadata.update({
            "completed_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_success":      n_ok,
            "n_failed":       n_fail,
            "n_repr_saved":   n_repr,
        })
        save_json(pred_path, {"metadata": metadata, "predictions": results})

        extractor.remove()
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] VRAM freed. Successes={n_ok} | Failures={n_fail} | Repr saved={n_repr}")

    # ── Compute and save metrics ───────────────────────────────────────────────
    print(f"[INFO] Computing {task} metrics ...")
    metrics = TASK_METRICS_FN[task](results, gt_map)
    if metrics:
        # Strip internal _raw before saving (keep raw data in predictions.json)
        metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
        save_json(task_dir / "metrics.json", metrics_to_save)
        print(f"[INFO] Metrics saved → {task_dir / 'metrics.json'}")
        _print_metric_summary(task, display, metrics)

    # ── Per-model figures ─────────────────────────────────────────────────────
    plot_single, _ = TASK_PLOT_FN[task]
    if plot_single and metrics:
        try:
            plot_single(metrics, display, task_dir)
            print(f"[INFO] Figures saved → {task_dir}/")
        except Exception as e:
            print(f"[WARN] Figure generation failed: {e}")

    return metrics


def _print_metric_summary(task: str, display: str, metrics: dict):
    print(f"\n  ── {task.upper()} Metrics — {display} ──")
    if task == "binary":
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"]:
            print(f"    {k:<22}: {metrics.get(k, '—')}")
    elif task == "multiclass":
        for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]:
            print(f"    {k:<22}: {metrics.get(k, '—')}")
    elif task == "description":
        for k in ["bleu_1", "bleu_2", "bleu_4", "rouge_1", "rouge_2", "rouge_L",
                  "meteor", "bertscore_precision", "bertscore_recall", "bertscore_f1"]:
            if k in metrics:
                print(f"    {k:<22}: {metrics.get(k, '—')}")
    lat = metrics.get("latency", {})
    if lat:
        print(f"    {'latency_mean_s':<22}: {lat.get('mean_s', '—')}")
        print(f"    {'latency_median_s':<22}: {lat.get('median_s', '—')}")
        print(f"    {'latency_p95_s':<22}: {lat.get('p95_s', '—')}")
    print()


def run_aggregate_plots(task: str, output_dir: Path):
    """Loads metrics.json for every model that has been run and generates comparison figures."""
    task_dir = output_dir / task
    _, plot_agg = TASK_PLOT_FN[task]
    if plot_agg is None:
        return

    all_metrics = {}
    for model_key in MODEL_REGISTRY:
        mpath = task_dir / model_key / "metrics.json"
        if mpath.exists():
            with open(mpath) as f:
                m = json.load(f)
            # Reload raw data from predictions for ROC/PR curves
            ppath = task_dir / model_key / "predictions.json"
            if task == "binary" and ppath.exists():
                with open(ppath) as f:
                    preds_data = json.load(f)
                preds = preds_data.get("predictions", [])
                # Reconstruct _raw from predictions
                gt_path = task_dir / model_key / "metrics.json"
                # We don't have gt_map here — skip curve data for aggregate,
                # curves are already saved per-model.
                m["_raw"] = {}   # curves already saved per-model
            all_metrics[MODEL_REGISTRY[model_key]["display_name"]] = m

    if not all_metrics:
        print(f"[INFO] No metrics found for task '{task}' — run models first.")
        return

    agg_dir = task_dir / "aggregate"
    plot_agg(all_metrics, agg_dir)
    print(f"[INFO] Aggregate figures saved → {agg_dir}/")

    # Save combined metrics summary
    save_json(agg_dir / "all_metrics.json", {k: {mk: mv for mk, mv in v.items()
                                                  if not mk.startswith("_")}
                                               for k, v in all_metrics.items()})


# =============================================================================
# SANITY CHECK
# =============================================================================

_REQUIRED_BINARY_PRED_KEYS      = {"image_id", "filename", "status", "anomaly_present", "confidence", "reasoning"}
_REQUIRED_MULTICLASS_PRED_KEYS  = _REQUIRED_BINARY_PRED_KEYS | {"scene_class"}
_REQUIRED_DESCRIPTION_PRED_KEYS = {"image_id", "filename", "status", "anomaly_description"}

_REQUIRED_BINARY_METRIC_KEYS     = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"}
_REQUIRED_MULTICLASS_METRIC_KEYS = {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa"}
_REQUIRED_DESCRIPTION_METRIC_KEYS= {"total_images"}   # NLP metrics are optional (need nltk/rouge-score/bert-score)

_TASK_PRED_KEYS   = {"binary": _REQUIRED_BINARY_PRED_KEYS,
                     "multiclass": _REQUIRED_MULTICLASS_PRED_KEYS,
                     "description": _REQUIRED_DESCRIPTION_PRED_KEYS}
_TASK_METRIC_KEYS = {"binary": _REQUIRED_BINARY_METRIC_KEYS,
                     "multiclass": _REQUIRED_MULTICLASS_METRIC_KEYS,
                     "description": _REQUIRED_DESCRIPTION_METRIC_KEYS}

_TASK_FIGURES = {
    "binary":      ["roc_curve.png", "pr_curve.png", "confusion_matrix.png"],
    "multiclass":  ["confusion_matrix_norm.png", "per_class_f1.png"],
    "description": [],   # no per-model figures; description_metrics.png is in aggregate/
}


def _check(label: str, ok: bool, detail: str = ""):
    icon = "  ✓" if ok else "  ✗"
    line = f"{icon}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


def run_sanity_check(task: str, model_key: str, output_dir: Path):
    """
    Inspects the output directory for one (task, model) combination and prints
    a detailed report covering:
      - predictions.json  : existence, record count, required keys, GT match
      - metrics.json      : existence, required metric keys, value ranges
      - representations/  : .npz count, array shapes, dtype (float16)
      - figures/          : expected PNG files, non-zero sizes

    Idea coverage checklist (all four ideas from the paper):
      Idea 1 — layer probing   : need layers array in .npz  (n_layers × vis_dim)
      Idea 2 — act. steering   : need layers + final_rep     (same .npz)
      Idea 3 — repr. detection : need final_rep              (1-D projection)
      (Idea 4 skipped — requires anomaly region masks)
    """
    cfg      = MODEL_REGISTRY[model_key]
    display  = cfg["display_name"]
    task_dir = output_dir / task / model_key

    print(f"\n{'─'*65}")
    print(f"  SANITY CHECK  |  task={task}  |  model={display}")
    print(f"  Dir: {task_dir}")
    print(f"{'─'*65}")

    all_ok = True

    # ── 1. predictions.json ───────────────────────────────────────────────────
    pred_path = task_dir / "predictions.json"
    print(f"\n  [predictions.json]")
    if not _check("file exists", pred_path.exists()):
        print(f"  [SKIP] Cannot inspect predictions — file missing.\n")
        return
    all_ok &= True

    size_kb = pred_path.stat().st_size / 1024
    with open(pred_path) as f:
        pred_data = json.load(f)

    preds    = pred_data.get("predictions", [])
    metadata = pred_data.get("metadata", {})
    _check("has 'predictions' key",  bool(preds),           f"{len(preds)} records, {size_kb:.1f} KB")
    _check("has 'metadata' key",     bool(metadata))

    required_keys = _TASK_PRED_KEYS[task]
    if preds:
        ok_records = [r for r in preds if r.get("status") == "success"]
        fail_records= [r for r in preds if r.get("status") != "success"]
        _check("success records",   bool(ok_records),        f"{len(ok_records)} ok / {len(fail_records)} failed")

        if ok_records:
            sample = ok_records[0]
            missing_keys = required_keys - set(sample.keys())
            _check("required keys present", not missing_keys,
                   f"missing: {missing_keys}" if missing_keys else f"all {len(required_keys)} keys present")
            # Print one example record
            print(f"\n    Example record (image_id={sample.get('image_id')}):")
            for k, v in sample.items():
                if k not in ("reasoning", "description"):
                    print(f"      {k:<28}: {v}")
                else:
                    val_str = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
                    print(f"      {k:<28}: {val_str}")

        if fail_records:
            print(f"\n    Failed record example: {fail_records[0]}")

    # ── 2. metrics.json ───────────────────────────────────────────────────────
    metric_path = task_dir / "metrics.json"
    print(f"\n  [metrics.json]")
    if not _check("file exists", metric_path.exists()):
        all_ok = False
    else:
        with open(metric_path) as f:
            metrics = json.load(f)
        m_size = metric_path.stat().st_size / 1024
        required_m = _TASK_METRIC_KEYS[task]
        missing_m  = required_m - set(metrics.keys())
        _check("required metric keys", not missing_m,
               f"missing: {missing_m}" if missing_m else f"all {len(required_m)} keys  ({m_size:.1f} KB)")

        if task == "binary":
            for k in ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc"]:
                v = metrics.get(k)
                in_range = v is not None and 0.0 <= v <= 1.0
                _check(f"  {k} in [0,1]", in_range, f"{v}")

        elif task == "multiclass":
            for k in ["accuracy", "macro_f1", "weighted_f1"]:
                v = metrics.get(k)
                in_range = v is not None and 0.0 <= v <= 1.0
                _check(f"  {k} in [0,1]", in_range, f"{v}")

        elif task == "description":
            any_nlp = any(metrics.get(k) is not None for k in ["bleu_1", "rouge_1"])
            if not any_nlp:
                print(f"  ℹ   NLP metrics absent (nltk/rouge-score/bert-score not installed)")
                print(f"      Install: pip install nltk rouge-score bert-score")
            else:
                for k in ["bleu_1", "rouge_1"]:
                    v = metrics.get(k)
                    in_range = v is not None and 0.0 <= v <= 1.0
                    _check(f"  {k} in [0,1]", in_range, f"{v}")

    # ── 3. representations/ ───────────────────────────────────────────────────
    repr_dir     = task_dir / "representations"
    has_layers   = False
    has_final    = False
    layers_shape = (0, 0)
    final_shape  = (0,)
    print(f"\n  [representations/]  (Ideas 1, 2, 3)")
    if not repr_dir.exists():
        _check("directory exists", False, "NOT FOUND — vision hooks likely failed")
        all_ok = False
    else:
        npz_files = sorted(repr_dir.glob("data_*.npz"))
        _check("directory exists", True, f"{repr_dir}")
        _check(".npz files present", bool(npz_files), f"{len(npz_files)} files")

        has_layers   = False
        has_final    = False
        layers_shape = None
        final_shape  = None

        if npz_files:
            # Inspect first file in detail
            sample_f = npz_files[0]
            data = np.load(sample_f)
            keys = list(data.keys())
            _check("sample .npz keys", bool(keys), f"{sample_f.name} → {keys}")

            if "layers" in data:
                arr = data["layers"]
                layers_shape = arr.shape
                has_layers   = True
                _check("  'layers' dtype=float16",  arr.dtype == np.float16, f"shape={arr.shape}  dtype={arr.dtype}")
                _check("  'layers' ndim=2",         arr.ndim == 2,           f"(n_layers={arr.shape[0]}, vis_dim={arr.shape[1]})")
                _check("  Idea 1 — layer probing",  True,  f"✓ {arr.shape[0]} layers × dim {arr.shape[1]}")
                _check("  Idea 2 — act. steering",  True,  "(layers present; need final_rep too)")
            else:
                _check("  'layers' key", False, "MISSING — vision layer hooks did not fire")
                all_ok = False

            if "final_rep" in data:
                arr = data["final_rep"]
                final_shape = arr.shape
                has_final   = True
                _check("  'final_rep' dtype=float16", arr.dtype == np.float16, f"shape={arr.shape}  dtype={arr.dtype}")
                _check("  'final_rep' ndim=1",        arr.ndim == 1,           f"dim={arr.shape[0]}")
                _check("  Idea 3 — repr. detection",  True,  f"✓ projection dim={arr.shape[0]}")
            else:
                _check("  'final_rep' key", False, "MISSING — projection hook not found for this arch")
                # Not fatal — some archs don't expose a clear projection

            # Idea 2 requires both
            if has_layers and has_final:
                _check("  Idea 2 — act. steering (both keys)", True,
                       f"layers={layers_shape}  final_rep={final_shape}")
            elif has_layers:
                print(f"  ⚠   Idea 2: only layers saved; final_rep absent — steering may be partial")

            # Check file size
            sz = sample_f.stat().st_size / 1024
            _check("  sample file size > 0", sz > 0, f"{sz:.1f} KB")

        # Check if consolidated files exist (optional at this point)
        all_layer_f = repr_dir / "all_layer_representations.npz"
        all_final_f = repr_dir / "all_final_representations.npz"
        if all_layer_f.exists():
            _check("all_layer_representations.npz", True, f"{all_layer_f.stat().st_size/1024:.0f} KB")
        else:
            print(f"  ℹ   all_layer_representations.npz not yet consolidated (run --consolidate-repr)")
        if all_final_f.exists():
            _check("all_final_representations.npz", True, f"{all_final_f.stat().st_size/1024:.0f} KB")

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print(f"\n  [figures/]")
    expected_figs = _TASK_FIGURES.get(task, [])
    for fname in expected_figs:
        fpath = task_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            _check(fname, True, f"{fpath.stat().st_size/1024:.1f} KB")
        else:
            _check(fname, False, "MISSING or empty")
            # Figures only generate after metrics — may be absent on very small previews

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  Ideas coverage for {display} / {task}:")
    print(f"    Idea 1 — layer probing          : {'✓ layers saved' if has_layers else '✗ layers missing'}")
    if has_layers:
        print(f"             n_layers={layers_shape[0]}  vis_dim={layers_shape[1]}")
    print(f"    Idea 2 — activation steering    : {'✓ layers + final_rep' if (has_layers and has_final) else ('⚠ layers only' if has_layers else '✗ missing')}")
    print(f"    Idea 3 — repr. detection        : {'✓ final_rep saved' if has_final else '⚠ final_rep missing (arch-dependent)'}")
    print(f"    Idea 4 — spatial grounding      : ✗ skipped (no region masks in dataset)")
    print(f"  {'─'*55}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="VLM evaluation — binary / multiclass / description tasks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--task", required=True,
                   choices=["binary", "multiclass", "description"],
                   help=(
                       "binary      : anomaly vs. normal (binary classification)\n"
                       "multiclass  : 11-class scene classification\n"
                       "description : free-text anomaly description (anomalous images only)"
                   ))
    p.add_argument("--model", default="all",
                   choices=list(MODEL_REGISTRY.keys()) + ["all"],
                   help="Model key to run, or 'all' to run every model sequentially.")
    p.add_argument("--images-dir",   "-i", required=True,
                   help="Path to Data/images/ directory.")
    p.add_argument("--dataset-json", "-j", required=True,
                   help="Path to Data/dataset.json (ground truth).")
    p.add_argument("--output-dir",   "-o", default="./vlm_eval_outputs",
                   help="Root output directory (default: ./vlm_eval_outputs).")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip inference — regenerate aggregate figures from saved metrics.")
    p.add_argument("--consolidate-repr", action="store_true",
                   help=(
                       "After inference, merge all per-image .npz representation files into\n"
                       "two consolidated arrays (all_layer_representations.npz,\n"
                       "all_final_representations.npz) ready for probing/steering.\n"
                       "Can also be run standalone with --aggregate-only."
                   ))
    p.add_argument("--preview",    type=int,   default=0,
                   help="Run on first N images only (0 = all).")
    p.add_argument("--start-from", type=int,   default=1,
                   help="Skip images with index < N (for crash recovery).")
    p.add_argument("--use-4bit",   action="store_true",
                   help="Load model in 4-bit (saves VRAM, slight quality drop).")
    p.add_argument("--delay",      type=float, default=0.05,
                   help="Seconds to sleep between images (default: 0.05).")
    p.add_argument("--hf-token",   type=str,   default=None,
                   help="HuggingFace token for gated models (e.g. Llama-3.2).")
    p.add_argument("--sanity-check", action="store_true",
                   help=(
                       "Run on first 3 images only, then print a detailed report verifying:\n"
                       "  predictions.json structure and required keys\n"
                       "  metrics.json required metric keys and value ranges\n"
                       "  representations/ .npz shapes and dtypes (Ideas 1-3 coverage)\n"
                       "  figure files exist and are non-empty"
                   ))
    args = p.parse_args()

    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("[INFO] HuggingFace auth OK.")
        except ImportError:
            print("[WARN] huggingface_hub not installed.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        run_aggregate_plots(args.task, output_dir)
        if args.consolidate_repr:
            for model_key in MODEL_REGISTRY:
                task_dir = output_dir / args.task / model_key
                if (task_dir / "representations").exists():
                    consolidate_representations(task_dir)
        return

    # Load ground truth
    print(f"[INFO] Loading ground truth from {args.dataset_json} ...")
    gt_map = load_ground_truth(Path(args.dataset_json))
    print(f"[INFO] GT records: {len(gt_map):,}")

    # Sanity check forces a 3-image preview
    if args.sanity_check:
        if args.preview == 0:
            args.preview = 3
        print(f"[SANITY CHECK] Forcing --preview {args.preview}. Will verify all outputs after inference.")

    # Load image paths
    images = find_images(Path(args.images_dir))
    if not images:
        sys.exit(f"[ERROR] No images found in {args.images_dir}")
    if args.preview > 0:
        images = images[:args.preview]
    print(f"[INFO] Images: {len(images):,}  |  Task: {args.task}  |  4-bit: {args.use_4bit}")

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"\n{'='*65}")
    print(f"  VLM Eval — Task: {args.task.upper()}")
    print(f"  Models : {models_to_run}")
    print(f"  Output : {output_dir}")
    print(f"{'='*65}\n")

    for model_key in models_to_run:
        run_model_task(
            task=args.task,
            model_key=model_key,
            images=images,
            gt_map=gt_map,
            output_dir=output_dir,
            use_4bit=args.use_4bit,
            delay=args.delay,
            start_from=args.start_from,
        )
        if args.sanity_check:
            run_sanity_check(args.task, model_key, output_dir)

    # Generate aggregate comparison figures after all models are done
    if args.model == "all":
        print("\n[INFO] Generating aggregate comparison figures ...")
        run_aggregate_plots(args.task, output_dir)

    # Consolidate per-image representation files into single arrays
    if args.consolidate_repr:
        print("\n[INFO] Consolidating representation files ...")
        for model_key in models_to_run:
            task_dir = output_dir / args.task / model_key
            consolidate_representations(task_dir)

    print(f"\n[INFO] All done. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
