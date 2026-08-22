# Vision Model Donor-Brain Research Brief

**Project:** Raspberry Pi 4B 8GB plant health MVP  
**Scope:** Vision model only (crop type + health level) — not tip LLM  
**Date:** 2026-08-12  
**Crops v1:** palay (rice), eggplant, lettuce, tomato, sili (chili/pepper)  
**Health:** healthy | mild | critical | dead (mapped from disease labels)  
**Deploy:** train PC/Colab → lightweight Pi; public datasets first

---

## Executive recommendation

| Role | Choice | Why |
|------|--------|-----|
| **Primary MVP** | **MobileNetV3-Small (or V2) multitask two-head**, ImageNet init → fine-tune on remapped public leaf sets → **INT8 TFLite** (or Torch→ONNX→TFLite) | Best fit for shared backbone + crop(5)+health(4); proven plant-disease transfer; Pi-friendly latency/RAM; permissive tooling (torchvision / TF) |
| **Fallback** | **EfficientNet-Lite0 / EfficientNet-B0** same two-head recipe, or **HF PlantVillage MobileNetV2 checkpoint** as warm start then re-head | Stronger accuracy headroom if MobileNet underfits; still edge-deployable with quantization |
| **Defer** | YOLOv8/v11-nano leaf detect, RF-DETR, BioCLIP, MobileViT/TinyViT bake-off, field auto-crop pipeline | Detection & foundation models add data/label/license/latency cost; MVP assumes leaf-close framing |

---

## Locked problem → model shape

One vision model must emit **both**:

1. **Crop** ∈ {rice, eggplant, lettuce, tomato, pepper} (5-way)
2. **Health** ∈ {healthy, mild, critical, dead} (4-way)

**Practical architecture (all CNN options below):**

```
backbone (shared) → GAP/flatten
  ├─ head_crop: Linear → Softmax(5)
  └─ head_health: Linear → Softmax(4)
loss = λ_c * CE(crop) + λ_h * CE(health)
```

Disease labels from public datasets are **not** the deploy taxonomy. Train-time recipe:

- Parse `crop` from folder / PlantVillage `crop` field / AgML metadata  
- Map disease name → health bucket with an explicit table (version-controlled)  
- Example heuristics (must be reviewed by agronomy later):  
  - `healthy` → healthy  
  - early blight / bacterial spot / leaf mold / brown spot (limited) → mild  
  - late blight / yellow leaf curl / mosaic / wilt / tungro → critical  
  - severe necrosis / fully desiccated / “dead” leaf if labeled → dead  
- **Dead** is scarce in public sets → expect weak class until you collect local dead leaves

---

## 1. Classic classifiers (multitask two-head)

### 1.1 MobileNetV2 / MobileNetV3

| | |
|--|--|
| **What** | Mobile inverted bottleneck CNNs for mobile/edge. V3 adds SE + h-swish / NAS search. |
| **License** | Architecture/papers open; **torchvision** weights usable under torchvision/PyTorch license terms ([torchvision](https://github.com/pytorch/vision)); TF Hub / Keras apps similarly open for research/commercial with attribution norms. Paper: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). |
| **Multitask fit** | Excellent. Shared trunk + two small heads is textbook. |
| **Pi 4 practicality** | Strong. Quantized MobileNet-class models commonly land ~**40–150 ms**/img on Pi 4 CPU depending on runtime/quant (see §6). Peak RAM for model+activations typically **≪ 500 MB** INT8 — leaves room for a later tiny LLM on 8GB. |
| **Fine-tune effort** | Low. Colab/PC transfer learning, freeze early blocks → unfreeze; 1–5 epochs head-only then full FT. |
| **Pros** | Fast path; huge plant-disease literature; TFLite-first; tiny binary. |
| **Cons** | Lab-trained PV accuracy ≠ field; small capacity if you later expand classes. |
| **Best use** | **MVP primary backbone.** |

Evidence: lightweight MobileNet/EfficientNet comparisons for leaf disease are common ([TEEJ cassava/tomato study](https://ph04.tci-thaijo.org/index.php/TEE_J/article/view/11291); [Discover IoT lightweight eval](https://link.springer.com/article/10.1007/s43926-026-00310-0)).

### 1.2 EfficientNet-Lite / EfficientNet-B0

| | |
|--|--|
| **What** | Compound-scaled CNNs; **Lite** variants strip squeeze-excite / hard-swish for TFLite friendliness ([EfficientNet paper](https://arxiv.org/abs/1905.11946); Lite lineage via TensorFlow Model Garden / Edge). |
| **License** | Apache-2.0 for Google TF EfficientNet implementations (verify exact package LICENSE). |
| **Multitask fit** | Same two-head pattern. |
| **Pi 4** | B0/Lite0 still practical with INT8; slower/heavier than MobileNetV3-Small. Expect **~1.5–3×** MobileNet latency ballpark on CPU. |
| **Fine-tune effort** | Low–medium (larger than MobileNet; more VRAM at train time still fine on Colab). |
| **Pros** | Often wins accuracy vs MobileNet on plant sets (see TEEJ EfficientNet-B0 > MobileNetV3 > ResNet18 on cassava/tomato). |
| **Cons** | Export/ops quirks; slightly less headroom for co-resident LLM. |
| **Best use** | **MVP fallback** if MobileNet plateaus. |

### 1.3 ResNet18

| | |
|--|--|
| **What** | Classic residual net; torchvision ImageNet weights. Paper: [Deep Residual Learning](https://arxiv.org/abs/1512.03385). |
| **License** | torchvision. |
| **Multitask fit** | Fine. |
| **Pi 4** | Heavier (~11M params). Usable but slower; weaker edge choice vs MobileNet. |
| **Fine-tune effort** | Low. |
| **Pros** | Strong debugging baseline; simple. |
| **Cons** | Worse latency/size for Pi+LLM coexistence. |
| **Best use** | Offline ablation only — **not** primary Pi deploy. |

---

## 2. Existing plant-disease models & datasets as donors

### 2.1 PlantVillage-trained classifiers

| | |
|--|--|
| **What** | Models fine-tuned on PlantVillage ~54k leaf images, 14 crops / 38 crop–disease classes. Dataset: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset), HF [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage). Paper: [Mohanty et al. 2016](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full). |
| **License** | Dataset historically distributed for research; HF card notes open access — treat redistribution carefully; community often cites CC-style openness (verify before commercial ship). |
| **Coverage vs your crops** | **Tomato: yes.** **Pepper (bell): yes.** **Eggplant: no.** **Rice: no.** **Lettuce: no.** |
| **HF examples** | [linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) (~95% on PV-style eval); [Kathir56/plant-disease-tamilnadu](https://huggingface.co/Kathir56/plant-disease-tamilnadu) (MobileNetV2); [animeshakr/plant-disease-efficientnetv2s](https://huggingface.co/animeshakr/plant-disease-efficientnetv2s) (~21M, 384² — heavy for Pi). |
| **Multitask fit** | **Donor weights only.** Heads are 38-way disease — must **replace** with crop+health heads and remapped labels. |
| **Pi 4** | MobileNetV2-class donors: good. EfficientNetV2S @384: poor MVP choice. |
| **Fine-tune effort** | Low if you steal backbone; medium to build remapping + multi-crop mix. |
| **Pros** | Instant ImageNet→plant feature prior; huge community. |
| **Cons** | Controlled lab photos → domain gap to Pi Camera NoIR field; missing rice/lettuce/eggplant; **no native health severity**. |
| **Best use** | Warm-start backbone for MVP tomato/pepper subset; expand with other datasets. |

**Leakage warning:** HF PlantVillage docs stress splitting by `leaf_id` — multiple shots of same leaf ([dataset README](https://huggingface.co/datasets/mohanty/PlantVillage)).

### 2.2 PlantDoc

| | |
|--|--|
| **What** | Field-ish plant disease images; classification ~2.5k–8k depending on variant; detection set also exists. Paper: [Singh et al., CoDS COMAD 2020](https://doi.org/10.1145/3371158.3371196). AgML: [plant_doc_classification](https://huggingface.co/datasets/Project-AgML/plant_doc_classification); GitHub [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset). |
| **License** | Check repo (research use common). |
| **Fit** | Better **domain realism** than PlantVillage; still not your 5-crop×4-health taxonomy. Includes tomato/pepper-ish classes; not rice/lettuce-focused. |
| **Pi** | Irrelevant to architecture — data donor. |
| **Effort** | Medium (smaller N, noisier labels). |
| **Best use** | Domain-adaptation / fine-tune stage 2 after PV pretrain. |

### 2.3 AgML catalog

| | |
|--|--|
| **What** | [Project-AgML/AgML](https://github.com/Project-AgML/AgML/) — loaders for PlantVillage, PlantDoc, and many ag datasets ([listing](https://project-agml.github.io/AgML/dataset_listing.html)). |
| **License** | Library Apache-style; datasets retain upstream licenses. |
| **Fit** | **Data plumbing**, not a Pi model. |
| **Best use** | Standardize downloads/splits for MVP training scripts. |

### 2.4 Pl@ntNet

| | |
|--|--|
| **What** | Species ID engine; **cloud API** ([Pl@ntNet docs](https://docs.plantnet.org/en/reference/api-plantnet/)). |
| **Fit** | Species ID ≠ disease health; **not local edge**. Conflicts with “not frontier cloud LLMs / prefer local.” |
| **Best use** | **Defer / skip** for vision MVP donor brain. |

---

## 3. Detection approaches (leaf/plant detect + class)

### 3.1 Ultralytics YOLOv8 / YOLO11 / YOLO26-nano

| | |
|--|--|
| **What** | One-stage detectors; nano variants ~2–3M params. Official Pi guide: [Ultralytics Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi). Export: NCNN recommended on Pi ARM. |
| **License** | **AGPL-3.0** for Ultralytics YOLO — commercial closed-source may need Ultralytics license ([Roboflow comparison notes AGPL](https://blog.roboflow.com/rf-detr-vs-alternatives/)). |
| **Multitask fit** | Detection classes can be `crop_health` joint labels (20 classes) **or** detect leaf then classify — heavier pipeline. Classification mode exists in Ultralytics but loses localization. |
| **Pi 4 practicality** | Feasible, **not real-time**. Published: YOLOv8n tomato disease on Pi ~**0.7 s**/inference ([IEEE eStream 2024](https://doi.org/10.1109/estream61684.2024.10542533)); community reports ~1–2 s PyTorch/ONNX on Pi 4. Ultralytics official benches are mostly **Pi 5** (YOLO26n NCNN ~67 ms @640 FP32 on Pi 5 — expect **materially slower on Pi 4**). |
| **Fine-tune effort** | Medium–high: need **boxes**; PlantDoc detection helps; PV is classification-only. |
| **Pros** | Later auto-detect / multi-leaf / less framing dependence. |
| **Cons** | Label cost; AGPL; latency; RAM higher than MobileNet classifier. |
| **Best use** | **Later** when Pi Camera free-framing needs leaf crop. |

Two-stage pattern already shipped by others: YOLOv8 leaf detect + MobileNetV3 classify ([albinnnnn/leaf-disease-detector](https://github.com/albinnnnn/leaf-disease-detector)) — good **post-MVP** blueprint.

### 3.2 RF-DETR-nano

| | |
|--|--|
| **What** | Roboflow DETR-style detector, DINOv2 backbone. Nano ~**30.5M params**, 384², Apache-2.0 core. Repo: [roboflow/rf-detr](https://github.com/roboflow/rf-detr); docs: [rfdetr.roboflow.com](https://rfdetr.roboflow.com/latest/). |
| **License** | Apache-2.0 (Nano–Large); XL/2XL PML. |
| **Multitask fit** | Detection labels only unless you invent multi-task heads yourself. |
| **Pi 4** | **Poor.** Vendor explicitly: on bare Raspberry Pi CPU, **YOLO CNN still preferred**; RF-DETR targets GPU/Jetson ([blog](https://blog.roboflow.com/rf-detr-vs-alternatives/)). Latency numbers (2.3 ms) are **T4 TensorRT**, not Pi. |
| **Fine-tune effort** | Medium (COCO-format data). |
| **Pros** | Strong accuracy / Apache license / NMS-free. |
| **Cons** | Too heavy for Pi 4 + LLM; overkill for MVP classifier-first. |
| **Best use** | **Defer** unless you move to Jetson-class hardware. |

---

## 4. Vision transformers (lite)

### 4.1 MobileViT / MobileViT-v2

| | |
|--|--|
| **What** | Hybrid CNN+transformer for mobile. Paper: [MobileViT ICLR'22](https://arxiv.org/abs/2110.02178); code: [apple/ml-cvnets](https://github.com/apple/ml-cvnets). XS/S ~1–6M params. |
| **License** | Apple CVNets repo license (check LICENSE — typically research-friendly; confirm for product). |
| **Multitask fit** | Same two-head swap. |
| **Pi 4** | Plausible with export+INT8, but **toolchain friction** (CoreML-oriented history; ONNX/TFLite path less turnkey than MobileNet). On phone NPUs MobileViT-S is fast; on Pi CPU expect **similar or slower than MobileNetV3** for equal care. |
| **Fine-tune effort** | Medium (ecosystem thinner than torchvision MobileNet). |
| **Pros** | Slightly better ImageNet accuracy per param in paper vs MobileNetV3. |
| **Cons** | Extra complexity for MVP; attention ops less forgiving on ARM CPU. |
| **Best use** | Optional accuracy bake-off **after** MobileNet baseline. |

### 4.2 TinyViT

| | |
|--|--|
| **What** | Distilled tiny ViTs (5M/11M/21M). [microsoft/Cream TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT). |
| **License** | MIT (Cream). |
| **Pi 4** | 5M variant possible; transformers often memory-bandwidth heavy on A72. |
| **Best use** | Experiment later — not MVP. |

### 4.3 DeiT-tiny

| | |
|--|--|
| **What** | Data-efficient Image Transformer tiny (~5.7M). Paper: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877); [facebookresearch/deit](https://github.com/facebookresearch/deit). |
| **License** | Apache-2.0. |
| **Pi 4** | Feasible quantized; typically **no win** vs MobileNetV3 on CPU latency for this task. |
| **Best use** | Academic comparison only. |

---

## 5. Foundation / agricultural models (edge-relevant)

### 5.1 BioCLIP / BioCLIP 2

| | |
|--|--|
| **What** | Biology CLIP models (species/tree-of-life). [imageomics/bioclip](https://huggingface.co/imageomics/bioclip) ViT-B/16; [bioclip-2](https://huggingface.co/imageomics/bioclip-2) ViT-L/14. Paper: [BioCLIP CVPR 2024](https://arxiv.org/abs/2311.18803) (via model card). MIT. |
| **Fit** | Strong **species** priors; evaluated on PlantVillage/PlantDoc zero-shot in model card — but **not** disease-severity heads. Zero-shot PV accuracy in card still modest vs supervised CNNs. |
| **Pi 4** | **No for MVP.** ViT-B/16 ~86M; ViT-L much larger. Won't coexist happily with a tiny LLM on Pi 4 CPU. |
| **Best use** | Optional **PC-side** embedding / few-shot labeling aid; distill later if ever. |

### 5.2 Torchvision ImageNet weights

| | |
|--|--|
| **What** | MobileNetV2/V3, ResNet18, EfficientNet, etc. ([torchvision models](https://pytorch.org/vision/stable/models.html)). |
| **Fit** | Best **generic** init for multitask fine-tune. |
| **Pi** | Train on PC; export for Pi. |
| **Best use** | **MVP default init** (or PV-finetuned MobileNet if available). |

### 5.3 AgML “pretrained models”

| | |
|--|--|
| **What** | Framework emphasizes datasets + future ag ML utilities ([AgML site](https://project-agml.github.io/)). |
| **Fit** | Dataset access > ready edge disease multitask model. |
| **Best use** | Data pipeline. |

### 5.4 Pl@ntNet weights

Cloud API — **out of scope** for local donor brain.

---

## 6. Deployment runtimes on Pi 4 (8GB) + later tiny LLM

**Hardware reality:** Cortex-A72 @1.8 GHz, **no useful GPU for DNN**. VideoCore does not replace CUDA. Budget roughly:

| Slice | Rough RAM |
|-------|-----------|
| OS + Python services | 0.8–1.5 GB |
| Tiny LLM later (e.g. 1B-class Q4) | ~0.8–1.5 GB |
| Vision INT8 classifier | model few–tens MB + runtime buffers **~0.1–0.5 GB** |
| Headroom / camera / app | rest of 8 GB |

**Implication:** Prefer **≤10M param** CNN, INT8, single-shot classification. Avoid loading full PyTorch + YOLO + LLM together.

| Runtime | Verdict on Pi 4 | Notes / sources |
|---------|-----------------|-----------------|
| **TFLite / LiteRT** | **Primary** | Best ARM story for MobileNet-class. Community Pi 4 MobileNetV2 INT8 ~**38–80 ms** ([edge-ai-benchmark](https://github.com/joeltadeu/edge-ai-benchmark)); other guides show FP32→INT8 speedups ~2–3× ([IEEE ICFEC 2023 quant study](https://doi.org/10.1109/icfec57925.2023.00009)). |
| **ONNX Runtime** | Good secondary | Easy PyTorch export; often **slower than TFLite** on Pi 4 for same MobileNet (~2× in some ARM benches). Fine for prototypes. |
| **NCNN** | **Best for YOLO** | Ultralytics: NCNN fastest among YOLO exports on Pi ([docs](https://docs.ultralytics.com/guides/raspberry-pi)). Extra conversion work for custom two-head CNNs. |
| **OpenVINO** | Conditional | Strong on Intel; on Pi historically tied to **NCS2/Myriad** USB sticks. Not first choice for pure CPU MobileNet vs TFLite. |
| **PyTorch** | Train / debug only | Heaviest RAM; slowest inference on Pi ([edge-ai-benchmark](https://github.com/joeltadeu/edge-ai-benchmark) PyTorch Mobile slowest). |

**Latency targets for MVP UX:**  
- Classifier MobileNet INT8: aim **&lt;200 ms** (comfortable).  
- YOLO nano detect: plan **0.5–2 s** on Pi 4 — OK for button-press, not continuous 30 FPS.  
- Co-resident LLM: run vision and tip **sequentially**, not both peak-compute at once.

---

## 7. Public datasets vs your five crops

### Coverage matrix

| Crop | PlantVillage | PlantDoc | Other public | Gap severity |
|------|--------------|----------|--------------|--------------|
| **Tomato** | Strong (many diseases + healthy) | Yes | Many | Low |
| **Sili / pepper** | Bell pepper bacterial spot + healthy | Some pepper | Solanaceae Kaggle sets | Low–medium (bell ≠ all chili cultivars) |
| **Eggplant** | **Absent** | Limited | [Mendeley eggplant 4089 imgs CC BY 4.0](https://data.mendeley.com/datasets/d3ypkphghb/1); [Data in Brief](https://doi.org/10.1016/j.dib.2025.111353); [SLIF-Brinjal ~9k field](https://doi.org/10.17632/6yg6vktrc2) | Medium — must merge external sets |
| **Palay / rice** | **Absent** | No | [UCI Rice Leaf Diseases](https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases) (only **120** imgs); [RiceLeafBD](https://data.mendeley.com/datasets/kx9rx8p2mz/1); [BanglaRiceLeaf](https://doi.org/10.7910/DVN/XAOBYW); Kaggle rice disease collections | **High** — must stitch multiple rice sets; style shift vs PV |
| **Lettuce** | **Absent** | No | Sparse / niche Kaggle-style sets; no PV-scale standard | **Highest** — plan local capture early |

### Health-level mapping gap

Public sets give **disease names**, almost never **healthy/mild/critical/dead**. You must:

1. Author a mapping CSV per disease string.  
2. Accept **dead** underrepresentation.  
3. Prefer multi-source rice/eggplant with “healthy” class retained.  
4. Use PlantDoc / field rice sets to reduce lab→field shock before NoIR Wide deploy.

### Severity / realism papers to steal ideas from (not drop-in models)

- Mohanty et al. PlantVillage DL ([Frontiers 2016](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full))  
- PlantDoc ACM paper ([doi:10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196))  
- Lightweight model bake-offs on tomato/cassava (TEEJ, Discover IoT — links in §1)

---

## Option scorecard (MVP lens)

| Option | Multitask | Pi 4 + later LLM | Fine-tune effort | License risk | Verdict |
|--------|-----------|------------------|------------------|--------------|---------|
| MobileNetV3 two-head | ★★★★★ | ★★★★★ | ★★★★★ | Low | **Primary** |
| EfficientNet-Lite0/B0 two-head | ★★★★★ | ★★★★ | ★★★★★ | Low | **Fallback** |
| HF PV MobileNet donor → rehead | ★★★★ | ★★★★★ | ★★★★ | Low–med (data) | Warm start |
| ResNet18 | ★★★★★ | ★★★ | ★★★★★ | Low | Baseline only |
| YOLOv8/11n | ★★★ | ★★★ | ★★ | **AGPL** | Later detect |
| RF-DETR-nano | ★★ | ★ | ★★★ | Apache (good) | Not Pi 4 |
| MobileViT / TinyViT / DeiT-t | ★★★★ | ★★★ | ★★★ | Varies | Later bake-off |
| BioCLIP | ★★ | ★ | ★★ | MIT | Too heavy |
| Pl@ntNet API | ★ | — | — | Cloud ToS | Skip |

---

## Recommended path

### Primary MVP

1. **Backbone:** `mobilenet_v3_small` (or `mobilenet_v2`) from torchvision / TF.  
2. **Heads:** crop(5) + health(4); loss weighted (start λ=1,1; up-weight rare health).  
3. **Data v1:**  
   - PlantVillage tomato + pepper → remap health  
   - Eggplant Mendeley/SLIF → remap  
   - Concat 2–3 rice leaf sets → remap  
   - Lettuce: any public scraps + **early local photos**  
4. **Train:** Colab/PC; leaf_id-aware splits; heavy augmentation (color/lighting for future NoIR).  
5. **Export:** INT8 **TFLite** (or ONNX if PyTorch-first, then convert).  
6. **Pi:** sequential capture → classify → (later) tip LLM; do not keep YOLO+LLM+FP32 PyTorch resident.

### Fallback

- Same pipeline with **EfficientNet-Lite0/B0**, or warm-start from [HF MobileNetV2 PlantVillage](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) then **replace classifier** with two heads and continue FT on your remapped mix.

### Defer

- YOLO/RF-DETR auto leaf detect  
- BioCLIP / large ViT foundation on-device  
- MobileViT accuracy chase  
- Continuous video  
- Perfect “dead” class without local data  

---

## Concrete next engineering steps (vision only)

1. Freeze health mapping table for all diseases in chosen datasets.  
2. Build unbalanced-aware metrics: crop accuracy + health macro-F1 (dead will tank macro — track separately).  
3. Train MobileNetV3-Small multitask; measure Pi 4 TFLite latency/RAM with LLM stub loaded.  
4. If health F1 weak: add PlantDoc fine-tune stage; collect lettuce/dead locally.  
5. Only then prototype YOLO-nano leaf cropper as optional preprocessor.

---

## Key primary citations (URLs)

- PlantVillage dataset: https://github.com/spMohanty/PlantVillage-Dataset  
- PlantVillage HF: https://huggingface.co/datasets/mohanty/PlantVillage  
- Mohanty et al. 2016: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full  
- PlantDoc paper: https://doi.org/10.1145/3371158.3371196  
- AgML: https://github.com/Project-AgML/AgML/  
- MobileNetV3: https://arxiv.org/abs/1905.02244  
- EfficientNet: https://arxiv.org/abs/1905.11946  
- MobileViT: https://arxiv.org/abs/2110.02178 · https://github.com/apple/ml-cvnets  
- DeiT: https://arxiv.org/abs/2012.12877  
- TinyViT: https://github.com/microsoft/Cream/tree/main/TinyViT  
- BioCLIP: https://huggingface.co/imageomics/bioclip  
- Ultralytics Pi guide: https://docs.ultralytics.com/guides/raspberry-pi  
- RF-DETR: https://github.com/roboflow/rf-detr · https://blog.roboflow.com/rf-detr-vs-alternatives/  
- HF MobileNet PV: https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification  
- UCI rice leaves: https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases  
- Eggplant dataset (CC BY 4.0): https://data.mendeley.com/datasets/d3ypkphghb/1  
- Edge runtime benches: https://github.com/joeltadeu/edge-ai-benchmark  
- YOLOv8 Pi tomato disease: https://doi.org/10.1109/estream61684.2024.10542533  

---

*End of brief.*
