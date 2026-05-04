# Prompts

This document reproduces every prompt used in the project, both for VLM
inference and for the GPT-4o annotation / Claude validation pipelines.
For the canonical machine-readable versions, see the constants in
`eval/vlm_eval_tasks.py`.

## Task prompts (VLM inference)

### Binary anomaly detection

```
Inspect the image and decide if a driving anomaly is present.
Output JSON only:
{
  "anomaly_present": <bool>,
  "confidence": <float in [0,1]>,
  "reasoning": <short string>
}
```

### Multiclass classification

The system prompt enumerates the 11-class taxonomy with detailed
disambiguation rules (e.g., what counts as `extreme_weather` vs ordinary
rain) and the user prompt is:

```
Given that this image contains a driving anomaly, classify it into
exactly one of the 11 classes. Output JSON only:
{
  "scene_class": <one of the 11 strings>,
  "confidence": <float in [0,1]>,
  "reasoning": <short string>
}
```

### Free-form description

```
Describe the safety anomaly visible in this driving scene in 2–3
sentences. Focus on visible objects, their location in the frame,
and the immediate hazard to the ego vehicle.
```

## Annotation prompt (GPT-4o)

The full annotation prompt is reproduced in
`Dataset_Curation/annotation_prompt.txt` (when the dataset directory is
present) and includes:

- the 11-class taxonomy with disambiguation rules,
- a JSON schema enforcing the 7 attributes,
- consistency checks (e.g., `ego_risk_level=none` ⇒ `anomaly_present=false`),
- few-shot examples of well-formed and ill-formed outputs.

## Cross-validation prompt (Claude Sonnet 4.6)

```
You are an expert independent annotator for an autonomous-driving
anomaly-detection benchmark. You will be shown a dashcam image and a
candidate description, and asked to:
(a) classify the primary anomaly visible
(b) rate the description's accuracy against what you actually see.

[…11-class taxonomy…]

You MUST respond with only valid JSON — no preamble, no code fences:
{
  "your_class": "<one of the 11 strings above>",
  "class_confidence": <float 0.0-1.0>,
  "description_factuality": <int 1-5>,
  "description_completeness": <int 1-5>,
  "agrees_with_description": <true|false>,
  "issues": "<short string, or empty>"
}
```

The full text of both prompts (including all 11-class disambiguation
rules) is embedded in the Python source for exact reproducibility:

- Annotation prompt: `Dataset_Curation/annotation_prompt.txt` (separate)
- Validation prompt: top of `validation/dataset_validation_claude.py`
