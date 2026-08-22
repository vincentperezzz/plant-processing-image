# VGG19 for plant / leaf recognition (Philippine papers)

**Question:** How is VGG19 actually implemented in plant/leaf papers that claim it is good or best — especially Philippine work — and does that recipe beat or even fit our MobileNetV3-Small Pi 4B 8GB stack?

**Date:** 2026-08-13  
**Companion:** [vision-model-donor-brain-brief.md](vision-model-donor-brain-brief.md), [implementation-guide.md](../implementation-guide.md)

**Access note:** IEEE AIoT 2023, IEEE DASA 2023, and IEEE ICAIHC 2023 have **no open-access PDF** (Semantic Scholar `openAccessPdf.status=CLOSED`). Claims below that come only from IEEE/Crossref abstracts, the UP OVPAA summary, or the public PhilMedic dataset card are marked. Full freeze/unfreeze, optimizer, LR, epoch, and batch tables for those three papers are **not recoverable from public text**.

---

## Verdict (read this first)

| Claim you will hear | What the sources actually say |
| --- | --- |
| “VGG19 is best for Philippine medicinal plants” | Custodio (AIoT 2023) used **VGG19 as the only classifier** on PhilMedic (40 species, 4,922 leaves) and reported **92.67%**. There is **no bake-off** in the public abstract, so “best” is not proven vs ResNet/MobileNet. [DOI 10.1109/aiiot58121.2023.10174335](https://doi.org/10.1109/aiiot58121.2023.10174335) |
| “VGG19 is best for Philippine rice disease” | **False in the two UP Manila Magboo papers.** DASA 2023 / OVPAA: **VGG16 won at 96%**. ICAIHC 2023 (the paper that *did* run VGG19): **ResNet50 won** on Matthews Correlation Coefficient. [OVPAA](https://ovpaa.up.edu.ph/research/deep-learning-with-convolutional-neural-networks-to-assess-rice-plant-diseases-performs-very-well/), [DOI 10.1109/dasa59624.2023.10286749](https://doi.org/10.1109/dasa59624.2023.10286749), [DOI 10.1109/icaihc59020.2023.10430482](https://doi.org/10.1109/icaihc59020.2023.10430482) |
| “VGG19 + segmentation hits 99.72% on tomato” | True **on PlantVillage lab photos**, not a Philippine field set. Nguyen et al. froze VGG-19 convs, retrained the FC head, HSV-cut the leaf onto black. [DOI 10.3390/agriengineering4040056](https://doi.org/10.3390/agriengineering4040056) |
| “VGG19 is ~138M params” | That number is **VGG16** (config D). **VGG19 (config E) is ~144M** in the original paper and **143,667,240** in torchvision. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556), [torchvision vgg19](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html) |

VGG19 is a 2014 ImageNet CNN. The “Philippine” part is the **dataset**, not a special architecture.

---

## 0. VGG19 architecture (Simonyan & Zisserman 2014)

**Paper:** Karen Simonyan, Andrew Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) · [DOI 10.48550/arXiv.1409.1556](https://doi.org/10.48550/arXiv.1409.1556)

### What “19 weight layers” means

Configuration **E** in Table 1 of the paper: **16 conv + 3 fully-connected = 19 layers with weights**. ReLU and pooling do not count.

| Block | Layers (config E) | Channels | Spatial (224 input) |
| --- | --- | --- | --- |
| 1 | conv3-64, conv3-64, maxpool | 64 | 224 → 112 |
| 2 | conv3-128, conv3-128, maxpool | 128 | 112 → 56 |
| 3 | conv3-256 × 4, maxpool | 256 | 56 → 28 |
| 4 | conv3-512 × 4, maxpool | 512 | 28 → 14 |
| 5 | conv3-512 × 4, maxpool | 512 | 14 → 7 |
| Head | FC-4096, FC-4096, FC-1000, softmax | — | — |

All conv filters are **3×3**, stride 1, pad 1. Max-pool is **2×2**, stride 2. Input is a fixed **224×224 RGB** crop. Training preprocess in the paper is mean-RGB subtract only (no per-channel std). Modern Keras/torchvision ports use ImageNet mean/std (torchvision: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`; Keras VGG also BGR + caffe means).

### Parameter count (do not mix VGG16 and VGG19)

| Source | VGG16 (D) | VGG19 (E) |
| --- | --- | --- |
| Original paper Table 2 (millions, rounded) | 138 | **144** |
| torchvision `num_params` | 138,357,544 | **143,667,240** |
| Keras Applications table | 138.4M, 528 MB | **143.7M, 549 MB** |

Sources: [arXiv:1409.1556 Table 2](https://arxiv.org/abs/1409.1556); [torchvision vgg19](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html) (`_file_size` **548.1 MB**, **19.63 GFLOPS**); [Keras Applications](https://keras.io/api/applications/).

~138M / ~500MB is **VGG16**. VGG19 FP32 weights are **~548–549 MB**.

### Original ImageNet training recipe (not used by the plant papers)

From §3.1 of the VGG paper:

| Knob | Value |
| --- | --- |
| Optimizer | SGD + momentum 0.9 |
| Batch | 256 |
| LR | 1e-2, drop ×10 when val plateaus (3 drops) |
| Weight decay | 5e-4 |
| Dropout | 0.5 on first two FC layers |
| Epochs | 74 (370K iterations) |
| Augment | random 224 crop from rescaled image, horizontal flip, RGB color shift |
| Hardware | 4× NVIDIA Titan Black; **2–3 weeks** per net |

That is from-scratch ImageNet. Plant papers below almost always **load those (or Caffe/Keras/torchvision) ImageNet weights** and swap the 1000-way head.

### Why it is heavy vs MobileNetV3-Small

Think of VGG19 as a brick wall: every brick is a full 3×3 convolution that looks at every input channel. MobileNetV3 is a folding bike: depthwise 3×3 (one filter per channel) plus a cheap 1×1 mix, inverted residuals, squeeze-excite, and a tiny head.

| | VGG19 | MobileNetV3-Small |
| --- | --- | --- |
| Params | 143,667,240 | **2,542,856** |
| FP32 file | 548.1 MB | **9.8 MB** |
| GFLOPS (224) | 19.63 | **0.06** |
| ImageNet top-1 (torchvision recipe) | 72.376% | 67.668% |
| Keras CPU step (EPYC, batch 32) | 84.8 ms | MobileNetV2 analogue 25.9 ms (V3-Small not in that table) |

Sources: [torchvision vgg19](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html), [torchvision mobilenet_v3_small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html), [Keras Applications](https://keras.io/api/applications/), [MobileNetV3 paper](https://arxiv.org/abs/1905.02244).

Most of VGG’s mass is the **two Dense(4096)** layers, not the conv stack. Plant transfer learning often drops `include_top` and adds a small softmax, which cuts deploy size — but the conv stack is still dense 3×3 and still ~20 GFLOPS. On a Pi 4 CPU that is the difference between “snap a leaf” and “go make coffee.”

---

## 1. IEEE AIoT 2023 — Philippine medicinal plants (VGG19 only)

**Citation:** Epie F. Custodio. *Classifying Philippine Medicinal Plants Based on Their Leaves Using Deep Learning.* 2023 IEEE World AI IoT Congress (AIIoT), Seattle, 7–10 June 2023, pp. 29–35. [DOI 10.1109/aiiot58121.2023.10174335](https://doi.org/10.1109/aiiot58121.2023.10174335) · IEEE: [document 10174335](https://ieeexplore.ieee.org/document/10174335)  
**Affiliation:** Mindoro State University, College of Computer Studies, Calapan City.

**Dataset card (open):** *PhilMedic: Philippine Medicinal Plant Leaf Dataset.* Mendeley Data, v1, 12 Mar 2024. [DOI 10.17632/tsvdyhbphs.1](https://doi.org/10.17632/tsvdyhbphs.1) · [data.mendeley.com/datasets/tsvdyhbphs](https://data.mendeley.com/datasets/tsvdyhbphs)

| Field | Publicly documented value | Source |
| --- | --- | --- |
| Dataset | PhilMedic, 4,922 leaf images, 40 species | IEEE abstract; Mendeley |
| Classes | Species ID (Hibiscus, guava, lagundi, malunggay, sili, cassava, …). **Not** health grading | Mendeley class list |
| Capture | Natural light, 48 MP Android; **front and back** of each leaf; 30 Sep–9 Dec 2022 | Mendeley |
| Input size | **Not in abstract.** VGG19 default is 224×224 | Inferred from [arXiv:1409.1556](https://arxiv.org/abs/1409.1556), not from Custodio methods |
| Transfer learning | Mendeley categories include “Transfer Learning.” Abstract does not name ImageNet or freeze/unfreeze | Mendeley; IEEE abstract |
| Freeze / unfreeze | **Not in public text** | Paywall |
| Optimizer / LR / epochs / batch | **Not in public text** | Paywall |
| Augmentation | Yes. Abstract: limited data → augmentation “to reduce computational burden and increase variance” | IEEE abstract |
| Segment leaves? | **Not stated** in abstract or dataset card. References include segmentation papers; that is not proof they ran HSV/masking | IEEE abstract; Mendeley |
| Accuracy | **92.67%** average on the 4,922-image set | IEEE abstract |
| Did VGG19 win vs others? | **No comparison reported.** VGG19 was the classifier. Academia.edu “key takeaways” even list future work to try **architectures beyond VGG19** | IEEE abstract; [Academia.edu record](https://www.academia.edu/143461151/Classifying_Philippine_Medicinal_Plants_Based_on_Their_Leaves_Using_Deep_Learning) |
| Hardware / params | **Not in public text** | Paywall |
| Field prototype | 3 leaves × 40 classes. Perfect 3/3 on 21 species; 2/3 on 10; 1/3 on 8; **0/3 on 1** | IEEE abstract |

**What this paper is:** medicinal **species** ID from a locally collected leaf album. Overlap with our farm taxonomy is incidental (sili, cassava, sweet potato appear as species names). It does **not** output healthy/mild/critical/dead.

**Follow-on (same author, still not a Pi model):** Custodio 2024 ensemble of ResNet50/101/152 + VGG16 + VGG19 with transfer learning reached **98.22%** overall — which is evidence the 2023 VGG19-only number was not a ceiling. [DOI 10.1109/icitri62858.2024.10698832](https://doi.org/10.1109/icitri62858.2024.10698832)

---

## 2. IEEE ICAIHC 2023 — batch size on Philippine rice (VGG19 did **not** win)

**Citation:** Vincent Peter C. Magboo, Ma. Sheila A. Magboo. *Determination of Batch Size in Convolutional Neural Networks Applied to Philippine Rice Disease.* 2023 International Conference on Artificial Intelligence in Human-Computer Interaction (ICAIHC). [DOI 10.1109/icaihc59020.2023.10430482](https://doi.org/10.1109/icaihc59020.2023.10430482)  
**Affiliation:** University of the Philippines Manila.

| Field | Publicly documented value | Source |
| --- | --- | --- |
| Dataset | “Philippine Rice Disease Dataset” (image count / class list **not in abstract**) | IEEE abstract |
| Classes | Rice **diseases** (not species ID). Exact class count not in abstract. A later PH rice paper from the same literature uses 14 disease classes | Abstract; related [DOI 10.1109/icsgrc62081.2024.10690946](https://doi.org/10.1109/icsgrc62081.2024.10690946) |
| Input size | **Not in abstract.** Omdena/Kaggle “Philippines Rice Disease” copies are often pre-resized to 224×224; **unconfirmed** as Magboo’s exact files | [GTS dataset page](https://gts.ai/dataset-download/philippines-rice-disease-image-dataset/) |
| Models | **Base CNN, VGG19, ResNet50, InceptionV3** | IEEE abstract |
| Transfer learning / freeze / ImageNet | Abstract says “CNN models” including named pretrained architectures. Freeze recipe **not public** | IEEE abstract |
| Optimizer / LR / epochs | **Not in public text** | Paywall |
| Batch | The independent variable. Authors recommend **lower batch size**; increasing batch **did not** boost diagnostic capability. Exact grid (e.g. 16/32/64) **not in abstract** | IEEE abstract |
| Augmentation / segmentation | **Not in abstract** | Paywall |
| Accuracy | **90–96%** (range across models, not a VGG19-only score). Recall 91–97%, precision 96–98%, F1 94–98% | IEEE abstract |
| Did VGG19 win? | **No. ResNet50 won** (highest Matthews Correlation Coefficient) | IEEE abstract |
| Hardware / params | **Not in public text** | Paywall |

This is the Philippine rice paper that actually **named VGG19**. VGG19 is a contestant. ResNet50 took the trophy.

---

## 3. UP OVPAA summary + IEEE DASA 2023 — VGG16 won at 96% (not VGG19)

**OVPAA page (08 Jul 2025):** *Deep learning with convolutional neural networks to assess rice plant diseases performs very well.* UP Manila. [ovpaa.up.edu.ph/…/rice-plant-diseases…](https://ovpaa.up.edu.ph/research/deep-learning-with-convolutional-neural-networks-to-assess-rice-plant-diseases-performs-very-well/)

**Full paper the summary points at:** Vincent Peter C. Magboo, Ma. Sheila A. Magboo. *Philippine Rice Disease Classification Using Deep Learning.* 2023 International Conference on Decision Aid Sciences and Applications (DASA), pp. 158–163. [DOI 10.1109/dasa59624.2023.10286749](https://doi.org/10.1109/dasa59624.2023.10286749) · IEEE: [document 10286749](https://ieeexplore.ieee.org/document/10286749)

OVPAA reprints the IEEE abstract. It is **not** an independent experiment.

| Field | Publicly documented value | Source |
| --- | --- | --- |
| Dataset | Philippine Rice Disease Dataset | OVPAA; IEEE DASA abstract |
| Classes / N images / input size | **Not in public abstract** | Paywall |
| Models | “Base convolutional neural and pre-trained networks.” Winner named: **VGG16**. InceptionV3 “also generated superior performance.” Base CNN weaker. **VGG19 is not named** | OVPAA; DASA abstract |
| Transfer learning | Implied by “pre-trained networks.” Freeze/LR/epochs/batch **not public** | Abstract |
| Augmentation / segmentation | **Not in abstract** | Paywall |
| Accuracy | **VGG16: 96% acc, 99% sensitivity, 97% precision, 98% F1, 0.834 normalized MCC** | OVPAA; DASA abstract |
| Did VGG19 win? | **No. VGG16 won.** VGG19 is not in the public model list | OVPAA; DASA abstract |
| Hardware / params | **Not in public text** | Paywall |

**Do not collapse DASA and ICAIHC.** Same authors, same dataset name, two papers:

1. DASA 2023 → **VGG16** best (VGG19 not mentioned in the abstract).  
2. ICAIHC 2023 → ran **VGG19** among others → **ResNet50** best on MCC.

Neither paper supports “VGG19 is best for Philippine rice.”

---

## 4. MDPI AgriEngineering 2022 — VGG-19 + HSV segmentation on tomato (open access)

**Citation:** Thanh-Hai Nguyen, Thanh-Nghia Nguyen, Ba-Viet Ngo. *A VGG-19 Model with Transfer Learning and Image Segmentation for Classification of Tomato Leaf Disease.* AgriEngineering 2022, 4(4), 871–887. [DOI 10.3390/agriengineering4040056](https://doi.org/10.3390/agriengineering4040056) · HTML/PDF: [mdpi.com/2624-7402/4/4/56](https://www.mdpi.com/2624-7402/4/4/56)

This is **Vietnamese authors + PlantVillage**, not a Philippine field study. It is the cleanest public VGG-19 plant-disease recipe of the five targets.

| Field | Value | Source |
| --- | --- | --- |
| Dataset | PlantVillage tomato, **16,010** images | §2.1, Table 1 |
| Classes | 10: bacterial spot (2127), Septoria leaf spot (1771), mosaic virus (373), leaf mold (952), target spot (1404), early blight (1000), yellow leaf curl (3208), late blight (1908), two-spotted spider mites (1676), healthy (1591) | Table 1 |
| Input size | Resized to **224×224** (text also mentions 256×256 as a VGG-19 option) | §2.1, §3 |
| Transfer learning | Parameter-based TL; **ImageNet-style pretrained VGG-19**; **freeze convolution layers, re-train fully-connected layers** | §2.5 |
| Also compared | AlexNet, GoogLeNet, ResNet50 with “some layers frozen, some re-trained”; same split | §2.2, §3.5 |
| Optimizer | Not named as Adam/SGD. Hidden activation **Tansig**, output **Softmax** (MATLAB-style) | §3.2 |
| LR | Best **1e-5**; also tried 1e-4 (99.34%) and 1e-3 (99.02%) | Table 4 |
| Epochs | Best **300**; grid 50/100/200/300/400 | Table 5 |
| Batch | **60** | §3.2 |
| Split | 80% train+val / 20% test; then 80/20 inside train+val → **64% / 16% / 20%** | §3.2 |
| Augmentation | **Not** the headline trick. They note mosaic (373 images) is imbalanced and *suggest* upsampling / class weights as future work | §3.2 |
| Segment leaves? | **Yes.** RGB→HSV, extract leaf, paint background **black** | §2.3 |
| Accuracy | **99.72%** (headline / Table 4). Confusion-matrix writeup also says 99.71% average | Abstract; Table 4; §3.2 |
| Did VGG-19 win vs others? | **Yes, on this lab set.** AlexNet 99.16%, ResNet50 99.19%, GoogLeNet 99.38%, **VGG-19 99.72%** (same params) | §3.5 / Figure 11 |
| Segmentation vs raw | Segmented 99.72% vs non-segmented 99.63%; train time 2.75×10⁵ s vs 2.98×10⁵ s; test 29.30 s vs 35.27 s | Table 6 |
| Hardware | **Not stated.** 2.75×10⁵ s ≈ **76 hours** of training | Table 6 |
| Model size / params | **Not stated** (standard VGG-19 implied) | — |

**Caveat:** 99.7% on PlantVillage tomato is the usual studio-photo ceiling. Our `label_map.yaml` maps those same folders to **health buckets**, not 10 disease names. Copying 99.72% as a product KPI would be lying to the robot.

---

## 5. How they implement VGG19 vs our MobileNetV3-Small (Pi 4B 8GB)

Our stack (locked): `src/model.py` TwoHeadNet = torchvision `mobilenet_v3_small(weights=DEFAULT)` + crop/health/gate heads; `training/train.py` 224 ImageNet-norm; freeze backbone 3 epochs @ 1e-3 then unfreeze last 4 blocks 5 epochs @ 3e-4 AdamW; batch 16; no HSV cutout. Deploy target: INT8 on Pi 4B 8GB, classify **&lt;200 ms**, leave RAM for a later tiny LLM. [implementation-guide.md](../implementation-guide.md)

| Knob | Typical VGG19 plant paper | Ours |
| --- | --- | --- |
| Job | One softmax: species **or** disease name | Two heads: 5 crops × 4 health levels |
| Backbone | VGG19 (~144M, ~549 MB FP32) | MobileNetV3-Small (~2.54M, 9.8 MB FP32; INT8 much smaller) |
| Init | ImageNet (explicit in Nguyen; implied elsewhere) | ImageNet (`MobileNet_V3_Small_Weights.DEFAULT`) |
| Freeze | Nguyen: freeze **all conv**, train FC only. Magboo/Custodio freeze schedule **unknown** | Heads 3 ep, then last **4** inverted-residual blocks |
| Input | 224×224 (VGG default; Nguyen confirmed) | Resize 256, train crop 224, ImageNet mean/std |
| Augment | Custodio: yes (unspecified). Nguyen: HSV black-bg instead of heavy geo aug | RRC, flip, ±18° rot, color jitter, blur |
| Segment | Nguyen **yes** (HSV). Philippine IEEE abstracts **do not say** | **No.** Leaf-close framing assumed |
| Batch | Nguyen 60; Magboo ICAIHC: **smaller is better** | 16 |
| Epochs | Nguyen 300 | 3 + 5 |
| Optimizer | Nguyen: Tansig/Softmax, LR 1e-5. Others unknown | AdamW 1e-3 then 3e-4, wd 1e-4 |
| Winner on PH rice | **Not VGG19** | We never claimed VGG19 |
| Edge | Keras CPU ~85 ms/step on a 92-core EPYC at batch 32 — not a Pi number | Target &lt;200 ms INT8 on Pi 4 CPU |

---

## 6. What we should steal (and what we should not)

Steal:

- **PhilMedic** as extra PH leaf photos (other-plant / unknown negatives, or a species teacher), CC BY 4.0. [Mendeley](https://data.mendeley.com/datasets/tsvdyhbphs)
- Magboo’s reminder that **MCC + a real bake-off** beats a single-model accuracy press release
- Nguyen’s observation that a **consistent leaf-on-black cutout** can shave train time — only if our camera crop is already leaf-dominant. HSV that assumes studio contrast will fail in a paddy

Do not steal:

- VGG19 as the Pi inspector. 548 MB FP32 and ~20 GFLOPS do not coexist happily with a later on-device LLM on 8 GB ([implementation-guide §6](../implementation-guide.md))
- 99.72% PlantVillage as a field KPI
- The sentence “VGG19 is best for Philippine rice”

If we ever want a **PC teacher** for distillation, VGG19 is a legal, boring choice. The **student** stays MobileNetV3-Small (or EfficientNet-Lite0 if Small underfits).

---

## Sources

1. Simonyan & Zisserman (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
2. Custodio (2023). *Classifying Philippine Medicinal Plants Based on Their Leaves Using Deep Learning.* [https://doi.org/10.1109/aiiot58121.2023.10174335](https://doi.org/10.1109/aiiot58121.2023.10174335)
3. Custodio (2024). *PhilMedic* dataset. [https://doi.org/10.17632/tsvdyhbphs.1](https://doi.org/10.17632/tsvdyhbphs.1)
4. Magboo & Magboo (2023). *Determination of Batch Size… Philippine Rice Disease.* [https://doi.org/10.1109/icaihc59020.2023.10430482](https://doi.org/10.1109/icaihc59020.2023.10430482)
5. Magboo & Magboo (2023). *Philippine Rice Disease Classification Using Deep Learning.* [https://doi.org/10.1109/dasa59624.2023.10286749](https://doi.org/10.1109/dasa59624.2023.10286749)
6. UP OVPAA (2025). Summary of (5). [https://ovpaa.up.edu.ph/research/deep-learning-with-convolutional-neural-networks-to-assess-rice-plant-diseases-performs-very-well/](https://ovpaa.up.edu.ph/research/deep-learning-with-convolutional-neural-networks-to-assess-rice-plant-diseases-performs-very-well/)
7. Nguyen, Nguyen, Ngo (2022). *A VGG-19 Model with Transfer Learning and Image Segmentation…* [https://doi.org/10.3390/agriengineering4040056](https://doi.org/10.3390/agriengineering4040056)
8. torchvision VGG19 weights card. [https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html)
9. torchvision MobileNetV3-Small weights card. [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
10. Keras Applications size table. [https://keras.io/api/applications/](https://keras.io/api/applications/)
11. Howard et al. (2019). *Searching for MobileNetV3.* [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
12. Custodio (2024). Ensemble follow-on. [https://doi.org/10.1109/icitri62858.2024.10698832](https://doi.org/10.1109/icitri62858.2024.10698832)
