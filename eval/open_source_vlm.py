"""
Open-Source VLM Anomaly Annotation Pipeline (Unified Dataset)
==============================================================
Runs all open-source VLMs on the unified dataset (data_XXXXX.png) produced
by merge_dataset.py. Each model first decides binary anomaly/normal, then fills
the full annotation schema only if the scene is anomalous. Normal images get
null for all anomaly-specific fields.

Models:
  1. internvl3_1b   OpenGVLab/InternVL3-1B-hf
  2. qwen2vl_2b     Qwen/Qwen2-VL-2B-Instruct
  3. qwen25vl_7b    Qwen/Qwen2.5-VL-7B-Instruct
  4. molmo_7b       allenai/Molmo-7B-D-0924
  5. llava_13b      llava-hf/llava-v1.6-vicuna-13b-hf
  6. llama32_11b    meta-llama/Llama-3.2-11B-Vision-Instruct

Usage:
    python vlm_annotation_pipeline.py \
        --input-dir    ./unified_dataset/images \
        --dataset-json ./unified_dataset/dataset.json \
        --output-dir   ./vlm_predictions \
        --model all

    python vlm_annotation_pipeline.py \
        --input-dir    ./unified_dataset/images \
        --dataset-json ./unified_dataset/dataset.json \
        --model qwen25vl_7b --use-4bit --preview 20

Requirements:
    pip install transformers accelerate torch torchvision Pillow tqdm einops
    pip install qwen-vl-utils
    pip install bitsandbytes   # optional 4-bit
"""

import argparse, gc, json, os, re, sys, time
from pathlib import Path

try:
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install torch Pillow tqdm")


# =============================================================================
# TAXONOMY
# =============================================================================

ANOMALY_CLASSES = [
    "animal_on_road", "extreme_weather", "road_surface_hazard",
    "fallen_debris_or_vegetation", "vehicle_incident", "infrastructure_failure",
    "human_presence_anomaly", "adverse_lighting", "oversized_or_unusual_vehicle",
    "multi_hazard_compound",
]

RECOMMENDED_ACTIONS = [
    "stop", "reduce_speed", "maintain_speed",
    "steer_left", "steer_right", "reverse",
]


# =============================================================================
# PROMPTS
# =============================================================================

# PROMPT STRATEGY: Chain-of-Thought (CoT)
# ----------------------------------------
# The root problem: models default to scene-level descriptions ("normal street
# scene") instead of safety-level analysis ("is anything blocking the lane?").
# CoT fixes this by forcing the model to reason through 4 specific visual
# questions BEFORE making the binary decision. By the time it answers the
# JSON, it has already committed to describing the road surface, lane, and
# immediate hazards in detail — making it much harder to give a generic answer.
#
# Pipeline:
#   Pass 1 (CoT reasoning) → free text, no JSON pressure
#   Pass 2 (binary decision, conditioned on reasoning) → short JSON
#   Pass 3 (full annotation, only if anomaly) → full JSON

SYSTEM_PROMPT = """\
You are an autonomous driving safety analyst. Your job is to carefully examine
dashcam images and determine whether the driving scene contains a safety anomaly.

ANOMALY CLASSES — memorize these exactly:

1. animal_on_road
   Any animal (deer, dog, cow, horse, bird, etc.) that is ON the road surface
   or actively crossing in front of the vehicle.

2. extreme_weather
   Severe fog, heavy snow, black ice, or flooding that makes the road surface
   completely invisible or impassable.

3. road_surface_hazard
   Large potholes, cracks, collapsed pavement, sinkholes, or debris embedded
   in the road surface that directly threatens the vehicle's tyres or chassis.

4. fallen_debris_or_vegetation
   A fallen tree, large branches, rocks, construction materials, or any large
   object that has fallen ONTO the road and blocks a driving lane.

5. vehicle_incident
   A vehicle that is overturned, crashed, on fire, stopped sideways across
   lanes, or involved in a collision that blocks the road.

6. infrastructure_failure
   A traffic light pole, road sign, barrier, or bridge element that has
   physically collapsed, fallen, or is lying across the road.

7. human_presence_anomaly
   A person who is standing IN the road (not on a sidewalk), lying on the road,
   running into traffic, or present in a position that is dangerous and unexpected.

8. adverse_lighting
   Extremely poor or blinding lighting conditions (not just night driving) such
   as direct sun glare making the road completely invisible, total darkness with
   no streetlights, or blinding oncoming headlights.

9. oversized_or_unusual_vehicle
   A vehicle that is significantly larger than standard passenger cars AND
   dominates the scene due to its size, OR a vehicle that is clearly not
   a standard road vehicle for this road type. This includes:

   SIZE-BASED (large heavy vehicles that fill or dominate the lane):
   - Cement mixers, concrete mixers with rotating drums
   - Dump trucks, tipper trucks carrying loads
   - Crane trucks, vehicles with cranes or booms mounted
   - Large construction vehicles on public roads
   - Heavy machinery transporters
   - Any truck whose size makes it dominate the camera view
   NOTE: If you can see a cement mixer drum, crane arm, large mixing
   bowl, or oversized load — it IS this class. Do not second-guess it.

   TYPE-BASED (vehicles wrong for this road type):
   - Agricultural tractors or harvesters on a public road
   - Military vehicles, tanks, or armoured vehicles
   - Horse-drawn carriages or non-motorised vehicles on motorways
   - Abnormal load transporters carrying extremely wide objects

   NOT this class:
   - Standard semi-trucks or lorries driving normally in a highway lane
   - Buses or coaches on city streets
   - Delivery vans or box trucks

10. multi_hazard_compound
    Two or more of the above anomaly classes are simultaneously present.

A NORMAL scene has NONE of the above. Trucks, buses, motorcycles, and cars
driving in their lanes are always normal. Pedestrians on sidewalks are normal.
Standard construction zones with cones and signs are normal.
"""

# ── Pass 1: CoT Reasoning (free text, no JSON) ────────────────────────────────
COT_PROMPT = """\
Carefully examine this dashcam image. Answer each question below in detail.
Your answers will be used to make a safety classification, so be specific.

Q1 — What type of road is this? (highway, city street, rural road, etc.)
     What are the standard vehicles and conditions you would expect here?

Q2 — List every vehicle visible in the image. For each one, describe:
     - What type of vehicle is it?
     - Is it a standard vehicle for this road type, or does it look unusual/unexpected?
     - Is it driving normally in a lane, or is it in an unusual position/condition?

Q3 — Is there anything ON the road surface that should not be there?
     Look carefully for: animals, people, fallen objects, debris, damage, flooding.
     Describe exactly what you see, or state clearly that the road surface is clear.

Q4 — Does any vehicle, person, or object in this image match ANY of these
     specific anomaly types?
     - A cement mixer, concrete mixer, dump truck, crane truck, or any
       large construction vehicle on a public road — YES even if driving
       normally in a lane. These ARE anomalies due to their size.
     - An agricultural machine, military vehicle, or horse-drawn carriage
     - A crashed, overturned, or fire-damaged vehicle
     - A person standing or lying in the road (not on sidewalk)
     - A fallen tree, large debris, or rocks blocking a lane
     - An animal on the road surface
     - Severe weather making the road invisible
     - A fallen traffic sign or light pole
     Answer YES or NO for each type, and describe what you see specifically.
     If you see a large drum, mixing bowl, crane arm, or construction
     equipment on the back of a vehicle — answer YES for the first type.

Q5 — Based on your answers above, would a driver need to react differently
     than they would in standard traffic? Why or why not?

Write your answers as plain text. Be specific and visual. Do not write JSON.
"""

# ── Pass 2: Binary decision conditioned on CoT reasoning ─────────────────────
BINARY_DECISION_PROMPT = """\
Based on your visual analysis above, make the final safety classification.

Answer TRUE (anomaly) if your analysis identified ANY of these:
  - A cement mixer, concrete mixer with rotating drum, dump truck,
    crane truck, or large construction vehicle — TRUE even if driving
    normally. If you described seeing a mixing drum, crane arm, tipper
    body, or large construction equipment on a vehicle → answer TRUE.
  - An agricultural machine, military vehicle, or horse-drawn carriage
  - Animal on the road surface
  - Person standing/lying IN the road (not on sidewalk)
  - Crashed, overturned, or fire-damaged vehicle
  - Fallen tree, large debris, or rocks blocking a lane
  - Flooded, collapsed, or severely damaged road surface
  - Fallen traffic sign or light pole on the road
  - Severe weather making the road invisible

Answer FALSE (normal) ONLY if the scene shows exclusively:
  - Standard cars, SUVs, motorcycles, or regular vans driving in lanes
  - Standard semi-trucks or buses on a highway or city street
  - Pedestrians on sidewalks or at crossings
  - Normal weather conditions

Return ONLY this JSON, nothing else:
{
  "anomaly_present": true | false,
  "confidence": <float 0.0-1.0>,
  "specific_object": "<exact anomalous object/vehicle you identified, or null>",
  "anomaly_class": "<one of the 10 anomaly classes if anomalous, or null>",
  "brief_reason": "<one sentence: what specific thing makes this anomalous or normal>"
}
"""

# ── Pass 3: Full annotation (only called when anomaly_present=true) ───────────
FULL_ANNOTATION_PROMPT = f"""\
This driving scene is ANOMALOUS based on your analysis. Now provide the
complete structured annotation. Be consistent with what you described earlier.

Return ONLY this JSON — fill in every field:
{{
  "anomaly_present": true,
  "anomaly_class": "<one of: {', '.join(ANOMALY_CLASSES)}>",
  "anomaly_class_confidence": <float 0.0-1.0>,
  "severity": "<low | moderate | severe>",
  "severity_justification": "<one sentence>",
  "anomaly_description": "<3-5 sentences, third person present tense, purely visual, start with the primary anomalous object and its location in frame>",
  "anomaly_key_facts": ["<atomic verifiable fact 1>", "<fact 2>", "<fact 3>", "<fact 4>"],
  "anomaly_objects": ["<object>"],
  "anomaly_location_in_frame": "<left | center-left | center | center-right | right | full-frame | upper | lower>",
  "ego_risk_level": "<immediate | caution | monitor>",
  "ego_risk_justification": "<one sentence>",
  "recommended_action": "<stop | reduce_speed | maintain_speed | steer_left | steer_right | reverse>",
  "action_reason": "<one sentence>",
  "visibility_affected": true | false,
  "visibility_affected_description": "<one sentence or null>",
  "secondary_hazards": ["<hazard if any>"],
  "benchmark_difficulty": "<easy | medium | hard>",
  "benchmark_difficulty_reason": "<one sentence>"
}}
"""

# Kept for reference — not used in CoT mode
COMBINED_PROMPT = ""  # disabled
BINARY_PROMPT   = ""  # disabled


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {
    "internvl3_1b": {
        "hf_id": "OpenGVLab/InternVL3-1B-hf",
        "display_name": "InternVL3-1B",
        "loader": "internvl3",
        "vram_gb": 3,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "qwen2vl_2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "display_name": "Qwen2-VL-2B",
        "loader": "qwen2vl",
        "vram_gb": 6,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "qwen25vl_7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen2.5-VL-7B",
        "loader": "qwen25vl",
        "vram_gb": 16,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "molmo_7b": {
        "hf_id": "allenai/Molmo-7B-D-0924",
        "display_name": "Molmo-7B-D",
        "loader": "molmo",
        "vram_gb": 16,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "llava_13b": {
        "hf_id": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "display_name": "LLaVA-v1.6-Vicuna-13B",
        "loader": "llava_next",
        "vram_gb": 28,
        "prompt_mode": "cot",
        "multi_image": False,
        "note": "Single-image only — no visual few-shot support.",
    },
    "llava_onevision_7b": {
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "display_name": "LLaVA-OneVision-7B",
        "loader": "llava_onevision",
        "vram_gb": 16,
        "prompt_mode": "cot",
        "multi_image": True,
        "note": "Multi-image supported — visual few-shot ready.",
    },
    "llama32_11b": {
        "hf_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "display_name": "Llama-3.2-11B-Vision",
        "loader": "llama32",
        "vram_gb": 22,
        "prompt_mode": "cot",
        "multi_image": False,
        "note": "Gated model — requires --hf-token.",
    },
    # ── New state-of-the-art models ──────────────────────────────────────────
    "internvl3_2b": {
        "hf_id": "OpenGVLab/InternVL3-2B-hf",
        "display_name": "InternVL3-2B",
        "loader": "internvl3",
        "vram_gb": 5,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "internvl3_8b": {
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "display_name": "InternVL3-8B",
        "loader": "internvl3",
        "vram_gb": 18,
        "prompt_mode": "cot",
        "multi_image": True,
    },
    "qwen25vl_3b": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "display_name": "Qwen2.5-VL-3B",
        "loader": "qwen25vl",
        "vram_gb": 8,
        "prompt_mode": "cot",
        "multi_image": True,
    },
}

# Confidence threshold — applied after binary decision.
# Tune this if model is still too aggressive or too conservative.
# 0.0 = accept all anomaly predictions
# 0.5 = reject uncertain ones
# Set to 0.0 for CoT mode since CoT already provides strong calibration.
CONFIDENCE_THRESHOLD = 0.0


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def load_image_pil(image_path: Path, max_size: int = 1024) -> Image.Image:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def find_images(input_dir: Path) -> list[Path]:
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        images.extend(input_dir.glob(ext))
    def sort_key(p):
        m = re.search(r"\d+", p.stem)
        return int(m.group()) if m else 0
    return sorted(set(images), key=sort_key)


# =============================================================================
# JSON UTILITIES
# =============================================================================

def parse_json_response(raw: str) -> dict | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{[\s\S]*\})", raw)
    if not match:
        return None
    s = match.group(1)
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    s = re.sub(r"//[^\n]*", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def build_null_annotation(image_id, filename, model_key, brief_reason=None,
                          cot_reasoning=None, confidence=None, specific_object=None):
    rec = {
        "anomaly_present": False,
        "anomaly_class": None, "anomaly_class_confidence": None,
        "severity": None, "severity_justification": None,
        "anomaly_description": None, "anomaly_key_facts": None,
        "anomaly_objects": None, "anomaly_location_in_frame": None,
        "ego_risk_level": None, "ego_risk_justification": None,
        "recommended_action": None, "action_reason": None,
        "visibility_affected": None, "visibility_affected_description": None,
        "secondary_hazards": None, "benchmark_difficulty": None,
        "benchmark_difficulty_reason": None,
        "image_id": image_id, "filename": filename, "model": model_key,
        "annotation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if brief_reason:              rec["brief_reason"]      = brief_reason
    if cot_reasoning:             rec["cot_reasoning"]     = cot_reasoning
    if confidence is not None:    rec["binary_confidence"] = confidence
    if specific_object:           rec["specific_object"]   = specific_object
    return rec


def validate_annotation(ann, model_key, image_id, filename):
    # Always extract these before any early return so they are never lost
    cot  = ann.get("cot_reasoning")
    conf = ann.get("confidence") if ann.get("confidence") is not None \
           else ann.get("binary_confidence")
    obj  = ann.get("specific_object")

    # Model said not anomalous → null record but keep CoT + confidence
    if not ann.get("anomaly_present", True):
        return build_null_annotation(
            image_id, filename, model_key,
            brief_reason=ann.get("brief_reason"),
            cot_reasoning=cot,
            confidence=conf,
            specific_object=obj,
        )

    # Confidence threshold → treat as normal but keep debug info
    if isinstance(conf, (int, float)) and conf < CONFIDENCE_THRESHOLD:
        null_rec = build_null_annotation(
            image_id, filename, model_key,
            brief_reason=ann.get("brief_reason"),
            cot_reasoning=cot,
            confidence=conf,
            specific_object=obj,
        )
        null_rec["confidence_threshold_applied"] = True
        return null_rec
    # Validate anomaly class
    if ann.get("anomaly_class") not in ANOMALY_CLASSES:
        ann["anomaly_class_raw"] = ann.get("anomaly_class")
        ann["anomaly_class"] = "unknown"
        ann["anomaly_class_warning"] = "Outside taxonomy"
    # Validate action
    if ann.get("recommended_action") not in RECOMMENDED_ACTIONS:
        ann["recommended_action_raw"] = ann.get("recommended_action")
        ann["recommended_action"] = "reduce_speed"
        ann["action_warning"] = "Outside allowed set; defaulted to reduce_speed"
    # Ensure lists
    for f in ("anomaly_key_facts", "anomaly_objects", "secondary_hazards"):
        if not isinstance(ann.get(f), list):
            ann[f] = []
    ann["anomaly_present"] = True
    ann["image_id"] = image_id
    ann["filename"] = filename
    ann["model"] = model_key
    ann["annotation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return ann


def save_progress(output_path, results, metadata):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "total_annotated": sum(1 for r in results if r.get("status") == "success"),
            "total_failed": sum(1 for r in results if r.get("status") == "failed"),
            "predictions": results,
        }, f, indent=2, ensure_ascii=False)


# =============================================================================
# MODEL LOADERS
# =============================================================================

def load_internvl3(model_id, use_4bit):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(device_map="auto", trust_remote_code=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_qwen2vl(model_id, use_4bit):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = "auto"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return model, processor, infer


def load_qwen25vl(model_id, use_4bit):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return model, processor, infer


def load_molmo(model_id, use_4bit):
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    kw = dict(trust_remote_code=True, device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        inputs = processor.process(images=[image], text=f"{system}\n\n{user}")
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=1400, stop_strings="<|endoftext|>"), tokenizer=processor.tokenizer)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.decode(generated, skip_special_tokens=True)
    return model, processor, infer


def load_llava_next(model_id, use_4bit):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.float16
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        conversation = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_llama32(model_id, use_4bit):
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = MllamaForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    def infer(model, processor, image, system, user):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{system}\n\n{user}"}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, processor, infer


def load_llava_onevision(model_id, use_4bit):
    """
    LLaVA-OneVision — supports multiple images natively.
    Uses LlavaOnevisionForConditionalGeneration + AutoProcessor.
    Multi-image is passed as a list of image dicts in the conversation content.
    """
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(device_map="auto", low_cpu_mem_usage=True)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    else:
        kw["torch_dtype"] = torch.float16
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()

    def infer(model, processor, image, system, user):
        # Single-image inference (standard CoT/binary calls)
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{system}\n\n{user}"},
            ]}
        ]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1400, do_sample=False,
                repetition_penalty=1.1
            )
        return processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def infer_multiimage(model, processor, images: list, system: str, user: str) -> str:
        """
        Multi-image inference for few-shot examples.
        images: list of PIL images — [example1, example2, ..., query_image]
        The user prompt must reference images in order using <image> placeholders.
        """
        content = []
        for img in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": f"{system}\n\n{user}"})

        conversation = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            images=images, text=prompt, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1400, do_sample=False,
                repetition_penalty=1.1
            )
        return processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # Expose multi-image capability as an attribute on the infer function
    # so the few-shot pipeline can call infer_fn.multiimage([img1, img2, ...], sys, user)
    infer.supports_multiimage = True
    infer.multiimage = lambda images, system, user: infer_multiimage(
        model, processor, images, system, user
    )

    return model, processor, infer


LOADERS = {
    "internvl3":       load_internvl3,
    "qwen2vl":         load_qwen2vl,
    "qwen25vl":        load_qwen25vl,
    "molmo":           load_molmo,
    "llava_next":      load_llava_next,       # LLaVA-v1.6-13B (single image)
    "llava_onevision": load_llava_onevision,  # LLaVA-OneVision-7B (multi-image)
    "llama32":         load_llama32,
}


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference_on_image(model, processor, infer_fn, image_path, model_key, prompt_mode, retry_count=2):
    image_id = image_path.stem
    filename = image_path.name

    for attempt in range(1, retry_count + 1):
        try:
            image = load_image_pil(image_path)

            if prompt_mode == "cot":
                # ── Pass 1: CoT reasoning (free text, no JSON) ────────────────
                raw_cot = infer_fn(model, processor, image, SYSTEM_PROMPT, COT_PROMPT)

                # Truncate CoT to prevent context overflow in Pass 2.
                # LLaVA-13B has a 4096 token limit; system prompt + image tokens
                # already use ~1500, so we budget 600 chars (~150 tokens) for CoT.
                # Other models have larger contexts so this is conservative and safe.
                COT_MAX_CHARS = 600
                cot_truncated = raw_cot.strip()
                if len(cot_truncated) > COT_MAX_CHARS:
                    # Try to cut at a sentence boundary
                    cut = cot_truncated[:COT_MAX_CHARS].rfind(". ")
                    cot_truncated = cot_truncated[:cut + 1] if cut > 200 else cot_truncated[:COT_MAX_CHARS]
                    cot_truncated += " [truncated]"

                # ── Pass 2: Binary decision conditioned on CoT reasoning ──────
                binary_context = (
                    f"Your visual analysis of this image:\n{cot_truncated}\n\n"
                    f"{BINARY_DECISION_PROMPT}"
                )
                raw_binary = infer_fn(model, processor, image, SYSTEM_PROMPT, binary_context)
                binary = parse_json_response(raw_binary)

                if binary is None:
                    if attempt < retry_count:
                        time.sleep(1); continue
                    return {"status": "failed", "image_id": image_id, "filename": filename,
                            "model": model_key, "error": "binary decision parse failed",
                            "raw_cot": raw_cot[:300], "raw_binary": raw_binary[:300]}

                # Apply confidence threshold
                confidence = binary.get("confidence", 1.0)
                is_anomaly = binary.get("anomaly_present", True)
                if is_anomaly and isinstance(confidence, (int, float)) and confidence < CONFIDENCE_THRESHOLD:
                    is_anomaly = False
                    binary["anomaly_present"] = False
                    binary["brief_reason"] = (
                        f"[threshold] conf={confidence:.2f} < {CONFIDENCE_THRESHOLD}. "
                        + binary.get("brief_reason", "")
                    )

                # Always store full CoT (not truncated) for analysis
                binary["cot_reasoning"] = raw_cot.strip()

                if not is_anomaly:
                    ann = validate_annotation(binary, model_key, image_id, filename)
                    return {"status": "success", "annotation": ann}

                # ── Pass 3: Full annotation (only if anomalous) ───────────────
                raw_full = infer_fn(model, processor, image, SYSTEM_PROMPT, FULL_ANNOTATION_PROMPT)
                full = parse_json_response(raw_full)

                if full is None:
                    binary["anomaly_present"] = True
                    ann = validate_annotation(binary, model_key, image_id, filename)
                    return {"status": "success", "annotation": ann}

                full["binary_confidence"]      = confidence
                full["binary_reason"]          = binary.get("brief_reason")
                full["binary_specific_object"] = binary.get("specific_object")
                full["cot_reasoning"]          = raw_cot.strip()
                ann = validate_annotation(full, model_key, image_id, filename)
                return {"status": "success", "annotation": ann}

            else:  # combined (default)
                raw = infer_fn(model, processor, image, SYSTEM_PROMPT, COMBINED_PROMPT)
                parsed = parse_json_response(raw)
                if parsed is None:
                    if attempt < retry_count: time.sleep(1); continue
                    return {"status": "failed", "image_id": image_id, "filename": filename,
                            "model": model_key, "error": f"JSON parse failed after {retry_count} attempts",
                            "raw_output": raw[:600]}
                ann = validate_annotation(parsed, model_key, image_id, filename)
                return {"status": "success", "annotation": ann}

        except Exception as e:
            if attempt == retry_count:
                return {"status": "failed", "image_id": image_id, "filename": filename,
                        "model": model_key, "error": str(e)}
            time.sleep(2)

    return {"status": "failed", "image_id": image_id, "filename": filename,
            "model": model_key, "error": "max retries exceeded"}


# =============================================================================
# PER-MODEL PIPELINE
# =============================================================================

def run_model_pipeline(model_key, input_dir, output_dir, images, dataset_index,
                       use_4bit, delay, start_from):
    cfg = MODEL_REGISTRY[model_key]
    display = cfg["display_name"]
    prompt_mode = cfg.get("prompt_mode", "combined")
    output_path = output_dir / f"predictions_{model_key}.json"

    print(f"\n{'='*65}")
    print(f"  Model       : {display}")
    print(f"  Prompt mode : {prompt_mode}")
    if "note" in cfg: print(f"  NOTE        : {cfg['note']}")
    print(f"  Output      : {output_path}")
    print(f"{'='*65}")

    # Resume
    results = []
    processed_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("predictions", [])
        processed_ids = {r.get("image_id") for r in results}
        if processed_ids:
            print(f"[INFO] Resuming: {len(results)} already done.")

    images_to_run = [
        p for p in images
        if p.stem not in processed_ids
        and (lambda m: int(m.group()) if m else 0)(re.search(r"\d+", p.stem)) >= start_from
    ]

    if not images_to_run:
        print(f"[INFO] All images done for {display}. Skipping.")
        return

    print(f"[INFO] Images to process: {len(images_to_run):,}")

    model, processor, infer_fn = LOADERS[cfg["loader"]](cfg["hf_id"], use_4bit)

    metadata = {
        "pipeline": "vlm-anomaly-annotation-unified",
        "model_key": model_key, "model_hf_id": cfg["hf_id"],
        "model_display": display, "prompt_mode": prompt_mode,
        "quantization": "4bit" if use_4bit else "bfloat16/float16",
        "anomaly_classes": ANOMALY_CLASSES, "recommended_actions": RECOMMENDED_ACTIONS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    n_anom = n_norm = n_fail = 0

    for image_path in tqdm(images_to_run, desc=display):
        result = run_inference_on_image(
            model, processor, infer_fn, image_path, model_key, prompt_mode
        )
        results.append(result)

        if result["status"] == "success":
            a = result["annotation"]
            gt = dataset_index.get(image_path.name, {})
            gt_label = gt.get("anomaly_present", "?")
            conf = a.get("binary_confidence", a.get("confidence", "?"))
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
            # Show first 80 chars of CoT reasoning so you can see what the model focused on
            cot_snippet = str(a.get("cot_reasoning") or "").replace("\n", " ")[:80]
            threshold_flag = " [thresh]" if a.get("confidence_threshold_applied") else ""
            if a.get("anomaly_present"):
                n_anom += 1
                obj = a.get("binary_specific_object") or a.get("anomaly_class") or "?"
                print(f"  ✓ {a['image_id']:15s} PRED=ANOMALY  GT={gt_label} conf={conf_str}")
                print(f"    obj={str(obj):30s} sev={str(a.get('severity') or ''):8s} act={str(a.get('recommended_action') or '')}")
                print(f"    cot: {cot_snippet}")
            else:
                n_norm += 1
                reason = str(a.get("brief_reason") or "")[:80]
                print(f"  ✓ {a['image_id']:15s} PRED=NORMAL   GT={gt_label} conf={conf_str}{threshold_flag}")
                print(f"    cot: {cot_snippet}")
        else:
            n_fail += 1
            print(f"  ✗ {result.get('image_id','?'):15s} FAILED: {result.get('error','')[:70]}")

        save_progress(output_path, results, metadata)
        time.sleep(delay)

    metadata.update({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predicted_anomalous": n_anom, "predicted_normal": n_norm, "failed": n_fail,
    })
    save_progress(output_path, results, metadata)

    print(f"\n[INFO] {display}: anomalous={n_anom} normal={n_norm} failed={n_fail}")
    print(f"[INFO] Saved: {output_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] VRAM freed.")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="VLM binary anomaly detection + annotation pipeline.")
    p.add_argument("--input-dir",    "-i", required=True)
    p.add_argument("--dataset-json", "-j", required=True)
    p.add_argument("--output-dir",   "-o", default="./vlm_predictions")
    p.add_argument("--model", default="all",
                   choices=list(MODEL_REGISTRY.keys()) + ["all"],
                   help=(
                       "Which model(s) to run (10 total):\n"
                       "  internvl3_1b       — InternVL3-1B        (multi-image,  3GB)\n"
                       "  internvl3_2b       — InternVL3-2B        (multi-image,  5GB)\n"
                       "  internvl3_8b       — InternVL3-8B        (multi-image, 18GB)\n"
                       "  qwen2vl_2b         — Qwen2-VL-2B         (multi-image,  6GB)\n"
                       "  qwen25vl_3b        — Qwen2.5-VL-3B       (multi-image,  8GB)\n"
                       "  qwen25vl_7b        — Qwen2.5-VL-7B       (multi-image, 16GB)\n"
                       "  molmo_7b           — Molmo-7B-D          (multi-image, 16GB)\n"
                       "  llava_onevision_7b — LLaVA-OneVision-7B  (multi-image, 16GB)\n"
                       "  llava_13b          — LLaVA-v1.6-13B      (single-image,28GB)\n"
                       "  llama32_11b        — Llama-3.2-11B       (single-image,22GB)\n"
                       "  all                — run all 10 sequentially"
                   ))
    p.add_argument("--preview",    type=int, default=0)
    p.add_argument("--start-from", type=int, default=1)
    p.add_argument("--use-4bit",   action="store_true")
    p.add_argument("--delay",      type=float, default=0.1)
    p.add_argument("--hf-token",   type=str, default=None)
    args = p.parse_args()

    if args.hf_token:
        try:
            from huggingface_hub import login; login(token=args.hf_token)
            print("[INFO] HuggingFace auth OK.")
        except ImportError:
            print("[WARN] huggingface_hub not installed.")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_index = {}
    djp = Path(args.dataset_json)
    if djp.exists():
        with open(djp) as f: ds = json.load(f)
        dataset_index = ds.get("samples", {})
        print(f"[INFO] Dataset index: {len(dataset_index):,} records")
    else:
        print(f"[WARN] dataset.json not found — GT labels won't show during inference")

    images = find_images(input_dir)
    if not images: sys.exit(f"[ERROR] No images in {input_dir}")
    if args.preview > 0: images = images[:args.preview]

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"\n{'='*65}")
    print(f"  VLM Annotation Pipeline — Unified Dataset")
    print(f"  Images: {len(images):,}  |  Models: {models_to_run}")
    print(f"  4-bit: {args.use_4bit}  |  Output: {output_dir}")
    print(f"{'='*65}\n")

    for model_key in models_to_run:
        run_model_pipeline(
            model_key, input_dir, output_dir, images,
            dataset_index, args.use_4bit, args.delay, args.start_from
        )

    print(f"\n[INFO] All done. Results in: {output_dir.resolve()}")
    print(f"[INFO] Next: run benchmark_evaluation.py")


if __name__ == "__main__":
    main()