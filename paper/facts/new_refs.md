# New 2023-2026 references for the related work

Verified 2026-11-06. 16 papers, none already in `references.bib`; BibTeX in `new_refs.bib`
(same keys). For each paper the DOI and metadata come from the Crossref work record, the
BibTeX from doi.org content negotiation (`Accept: application/x-bibtex`), and the abstract
from Crossref, Semantic Scholar or arXiv, as listed under "Sources". Every fetched title
matches the intended title (normalized similarity 1.00). The descriptions say only what the
abstracts say; the "Relation to FCT" lines are our positioning.

Counts: A 5, DG 2, B 5, C 4.

---

## A. Polyp segmentation networks and generalization to new centres

### cai2024vanet
- **Title:** Know your orientation: A viewpoint-aware framework for polyp segmentation
- **Authors:** Linghan Cai, Lijiang Chen, Jianhao Huang, Yifeng Wang, Yongbing Zhang
- **Venue:** Medical Image Analysis, vol. 97, art. 103288
- **Year:** 2024
- **DOI:** 10.1016/j.media.2024.103288
- **Sources:** https://api.crossref.org/works/10.1016/j.media.2024.103288 ; https://doi.org/10.1016/j.media.2024.103288 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.media.2024.103288
- **What it does:** Noting that physicians' adjustments of the colonoscope tip change the viewpoint of colonoscopy images and diversify polyp appearance, VANet localizes polyps with class activation maps from a viewpoint-classification task and refines features with a viewpoint-aware Transformer and a boundary-aware Transformer, reporting state-of-the-art results on seven public datasets.
- **Relation to FCT:** Network design, complementary: it shares FCT's premise that orientation changes hinder polyp segmentation but handles them with viewpoint-aware modules, whereas FCT leaves the network unchanged and constrains predictions across flipped views in the loss.

### xiao2024ctnet
- **Title:** CTNet: Contrastive Transformer Network for Polyp Segmentation
- **Authors:** Bin Xiao, Jinwu Hu, Weisheng Li, Chi-Man Pun, Xiuli Bi
- **Venue:** IEEE Transactions on Cybernetics, vol. 54, no. 9, pp. 5040-5053
- **Year:** 2024
- **DOI:** 10.1109/tcyb.2024.3368154
- **Sources:** https://api.crossref.org/works/10.1109/tcyb.2024.3368154 ; https://doi.org/10.1109/tcyb.2024.3368154 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/tcyb.2024.3368154
- **What it does:** CTNet combines a contrastive Transformer backbone with a self-multiscale interaction module and a collection information module to segment camouflaged polyps of different sizes, and reports gains over PraNet on Kvasir-SEG, CVC-ClinicDB, EndoScene, ETIS and CVC-ColonDB, largest on ETIS (18.2%) and CVC-ColonDB (10.1%).
- **Relation to FCT:** Network design, complementary: it is evaluated on the same five test sets, with its largest gains on the two hardest unseen ones, and changes the architecture, whereas FCT changes only the training objective.

### shao2024polyper
- **Title:** Polyper: Boundary Sensitive Polyp Segmentation
- **Authors:** Hao Shao, Yang Zhang, Qibin Hou
- **Venue:** Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2024), vol. 38, no. 5, pp. 4731-4739
- **Year:** 2024
- **DOI:** 10.1609/aaai.v38i5.28274
- **Sources:** https://api.crossref.org/works/10.1609/aaai.v38i5.28274 (record and abstract) ; https://doi.org/10.1609/aaai.v38i5.28274 (BibTeX)
- **What it does:** Polyper extracts low-confidence boundary regions and high-confidence interior regions from an initial segmentation map with differentiable morphological operators and uses the interior features in a boundary-sensitive attention that enhances features near the boundary, reaching state-of-the-art results on five public datasets with ResNet-50, MiT-B1 and Swin Transformer encoders.
- **Relation to FCT:** Network design, complementary: it focuses on uncertain boundary pixels found from the model's own prediction, whereas FCT reweights each pixel's consistency pull with the label and leaves the architecture unchanged.

### li2024asps
- **Title:** ASPS: Augmented Segment Anything Model for Polyp Segmentation
- **Authors:** Huiqian Li, Dingwen Zhang, Jieru Yao, Longfei Han, Zhongyu Li, Junwei Han
- **Venue:** Medical Image Computing and Computer Assisted Intervention - MICCAI 2024 (LNCS), pp. 118-128
- **Year:** 2024
- **DOI:** 10.1007/978-3-031-72114-4_12 (arXiv:2407.00718, comment "Accepted by MICCAI2024")
- **Sources:** https://api.crossref.org/works/10.1007/978-3-031-72114-4_12 ; https://doi.org/10.1007/978-3-031-72114-4_12 (BibTeX) ; abstract: https://export.arxiv.org/api/query?id_list=2407.00718
- **What it does:** ASPS adapts the Segment Anything Model (SAM) to polyp segmentation by integrating a trainable CNN encoder branch with SAM's frozen ViT encoder to add domain-specific local and high-frequency features, and by using SAM's IoU score to regularize prediction uncertainty during training, which improves out-of-distribution performance.
- **Relation to FCT:** SAM-based network design aimed at out-of-distribution performance, complementary: it relies on a large pretrained foundation model and the model's own quality estimate, whereas FCT adds no parameters and weights its consistency term with the ground truth.

### du2025umnet
- **Title:** UM-Net: Rethinking ICGNet for polyp segmentation with uncertainty modeling
- **Authors:** Xiuquan Du, Xuebin Xu, Jiajia Chen, Xuejun Zhang, Lei Li, Heng Liu, Shuo Li
- **Venue:** Medical Image Analysis, vol. 99, art. 103347
- **Year:** 2025
- **DOI:** 10.1016/j.media.2024.103347
- **Sources:** https://api.crossref.org/works/10.1016/j.media.2024.103347 ; https://doi.org/10.1016/j.media.2024.103347 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.media.2024.103347
- **What it does:** UM-Net extends ICGNet with a colour transfer operation that weakens the relation between colour and polyps, whose colour distribution varies with equipment, location and polyp properties, and with an uncertainty estimate rectified by variance, giving competitive accuracy and generalization on five polyp datasets.
- **Relation to FCT:** Training-side remedy for a nuisance factor that varies across centres (colour), complementary to FCT, which targets orientation by enforcing flip consistency.

---

## DG. Domain generalization and adaptation for polyp segmentation

### lu2025dclps
- **Title:** Domain-Interactive Contrastive Learning and Prototype-Guided Self-Training for Cross-Domain Polyp Segmentation
- **Authors:** Ziru Lu, Yizhe Zhang, Yi Zhou, Ye Wu, Tao Zhou
- **Venue:** IEEE Transactions on Medical Imaging, vol. 44, no. 9, pp. 3563-3573
- **Year:** 2025
- **DOI:** 10.1109/tmi.2024.3443262
- **Sources:** https://api.crossref.org/works/10.1109/tmi.2024.3443262 ; https://doi.org/10.1109/tmi.2024.3443262 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/tmi.2024.3443262
- **What it does:** DCL-PS is an unsupervised domain adaptation framework for polyp segmentation on target datasets from different imaging devices that combines domain-interactive contrastive learning with domain-mixed prototypes, contrastive cross-consistency training on prototypes of the main and perturbed auxiliary decoder outputs, and prototype-guided self-training that weights each pixel to filter out unreliable pseudo-labels.
- **Relation to FCT:** Cross-device adaptation with per-pixel reliability weighting of pseudo-labels: it needs unlabeled target-domain images and estimates reliability, whereas FCT sees no target data and weights each pixel's consistency pull with the ground truth.

### zhang2024illumination
- **Title:** Generalizable Polyp Segmentation via Randomized Global Illumination Augmentation
- **Authors:** Zuyu Zhang, Yan Li, Byeong-Seok Shin
- **Venue:** IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 4, pp. 2138-2151
- **Year:** 2024
- **DOI:** 10.1109/jbhi.2024.3363910
- **Sources:** https://api.crossref.org/works/10.1109/jbhi.2024.3363910 ; https://doi.org/10.1109/jbhi.2024.3363910 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/jbhi.2024.3363910
- **What it does:** The method decomposes colonoscopy images into reflectance, local and global illumination, augments them with synthesized target-like global illumination maps, and introduces an illumination variance insensitiveness measure that correlates with generalizability, outperforming competing methods on unseen domains among CVC-ClinicDB, CVC-ColonDB, ETIS-Larib and Kvasir-SEG.
- **Relation to FCT:** Augmentation-based domain generalization for a nuisance factor (illumination), complementary: it diversifies the training images, whereas FCT makes the predictions on flipped views of each image agree.

---

## B. Consistency regularization and transformation equivariance

### yang2023dac
- **Title:** Sample Efficiency of Data Augmentation Consistency Regularization
- **Authors:** Shuo Yang, Yijun Dong, Rachel Ward, Inderjit S. Dhillon, Sujay Sanghavi, Qi Lei
- **Venue:** Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023), PMLR vol. 206, pp. 3825-3853
- **Year:** 2023
- **DOI / arXiv:** no DOI; arXiv:2202.12230
- **Sources:** venue, authors and pages: https://proceedings.mlr.press/v206/ (proceedings index; entry links to https://proceedings.mlr.press/v206/yang23c.html) ; abstract and authors: https://export.arxiv.org/api/query?id_list=2202.12230
- **What it does:** The paper shows, for linear regression with label-invariant augmentations and then for misspecified augmentations and non-linear models (with generalization bounds), that data augmentation consistency regularization is more sample-efficient than empirical risk minimization on augmented data, and confirms it in a controlled comparison on CIFAR-100 with WideResNet.
- **Relation to FCT:** Consistency regularization under full supervision: it compares the two ways of using augmented views that FCT contrasts (training on augmented copies with their labels versus adding a consistency term), while FCT further uses the label to decide which view teaches at each pixel.

### wu2023aclnet
- **Title:** ACL-Net: Semi-supervised Polyp Segmentation via Affinity Contrastive Learning
- **Authors:** Huisi Wu, Wende Xie, Jingyin Lin, Xinrong Guo
- **Venue:** Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023), vol. 37, no. 3, pp. 2812-2820
- **Year:** 2023
- **DOI:** 10.1609/aaai.v37i3.25382
- **Sources:** https://api.crossref.org/works/10.1609/aaai.v37i3.25382 (record and abstract) ; https://doi.org/10.1609/aaai.v37i3.25382 (BibTeX)
- **What it does:** ACL-Net is a semi-supervised polyp segmentation framework that aligns affinity maps between student and teacher networks, with a memory-bank cross-image affinity aggregation module, to refine pseudo-labels through combined contrastive and consistency learning, evaluated on Kvasir-SEG, CVC-ClinicDB, CVC-300, CVC-ColonDB and ETIS.
- **Relation to FCT:** Teacher-student consistency on unlabeled polyp images, evaluated on the same five test sets; FCT applies consistency to labeled images and lets the label choose the teaching view.

### xu2023acmt
- **Title:** Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation
- **Authors:** Zhe Xu, Yixin Wang, Donghuan Lu, Xiangde Luo, Jiangpeng Yan, Yefeng Zheng, Raymond Kai-yu Tong
- **Venue:** Medical Image Analysis, vol. 88, art. 102880
- **Year:** 2023
- **DOI:** 10.1016/j.media.2023.102880
- **Sources:** https://api.crossref.org/works/10.1016/j.media.2023.102880 ; https://doi.org/10.1016/j.media.2023.102880 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.media.2023.102880
- **What it does:** AC-MT extends the mean teacher by selecting ambiguous voxels of unlabeled images with plug-and-play criteria based on entropy, model uncertainty or label-noise self-identification and incorporating the resulting ambiguity map into the consistency loss, improving left atrium and brain tumour segmentation.
- **Relation to FCT:** Selective consistency on unlabeled scans: it chooses where consistency applies from label-free ambiguity estimates, whereas FCT sets the strength and direction of each pixel's pull with the ground truth.

### chi2024abd
- **Title:** Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation
- **Authors:** Hanyang Chi, Jian Pang, Bingfeng Zhang, Weifeng Liu
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), pp. 4070-4080
- **Year:** 2024
- **DOI:** 10.1109/cvpr52733.2024.00390
- **Sources:** https://api.crossref.org/works/10.1109/cvpr52733.2024.00390 ; https://doi.org/10.1109/cvpr52733.2024.00390 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/cvpr52733.2024.00390
- **What it does:** ABD strengthens consistency learning in semi-supervised medical image segmentation by generating new samples through bidirectional patch displacement, guided by reliable prediction confidence for unlabeled images and by inverse confidence for labeled images, reaching state-of-the-art results.
- **Relation to FCT:** Consistency under input perturbations guided by the model's own confidence; FCT uses flips as the perturbation and guides its consistency term with the label.

### liu2025tracoco
- **Title:** Translation Consistent Semi-Supervised Segmentation for 3D Medical Images
- **Authors:** Yuyuan Liu, Yu Tian, Chong Wang, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis, Gustavo Carneiro
- **Venue:** IEEE Transactions on Medical Imaging, vol. 44, no. 2, pp. 952-968
- **Year:** 2025
- **DOI:** 10.1109/tmi.2024.3468896
- **Sources:** https://api.crossref.org/works/10.1109/tmi.2024.3468896 ; https://doi.org/10.1109/tmi.2024.3468896 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/tmi.2024.3468896
- **What it does:** TraCoCo enforces consistency between views of unlabeled 3D scans that differ in spatial input context, so that the model learns segmentation patterns from the foreground objects rather than the context, and adds a confident regional cross-entropy loss robust to pseudo-labelling mistakes, with state-of-the-art results on LA, Pancreas-CT, BraTS19 and ACDC.
- **Relation to FCT:** Consistency under spatial transformations on unlabeled scans, motivated, like FCT, by discouraging reliance on context rather than on the object; FCT applies flip consistency to labeled images and weights it with the ground truth.

---

## C. Self-distillation and label-aware or correctness-aware distillation

### lee2023sddropout
- **Title:** Self-knowledge distillation via dropout
- **Authors:** Hyoje Lee, Yeachan Park, Hyun Seo, Myungjoo Kang
- **Venue:** Computer Vision and Image Understanding, vol. 233, art. 103720
- **Year:** 2023
- **DOI:** 10.1016/j.cviu.2023.103720 (preprint arXiv:2208.05642)
- **Sources:** https://api.crossref.org/works/10.1016/j.cviu.2023.103720 ; https://doi.org/10.1016/j.cviu.2023.103720 (BibTeX) ; abstract: https://export.arxiv.org/api/query?id_list=2208.05642 (preprint v1; Semantic Scholar has no record for the DOI)
- **What it does:** SD-Dropout is a self-knowledge distillation method that distils between the posterior distributions of dropout-sampled versions of a single network, with no extra trainable modules, analyses forward and reverse KL divergences, and improves generalization, calibration, adversarial robustness and out-of-distribution detection on classification, detection and distribution-shift tasks.
- **Relation to FCT:** Self-distillation between stochastic views of one network under full supervision; FCT uses flipped inputs as the views and the label to decide which view teaches at each pixel.

### jeong2025selfdistill
- **Title:** Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels
- **Authors:** Hyeonsu Jeong, Hye Won Chung
- **Venue:** International Conference on Learning Representations (ICLR 2025)
- **Year:** 2025
- **DOI / arXiv:** no DOI; arXiv:2402.10482 (comment "ICLR 2025")
- **Sources:** https://export.arxiv.org/api/query?id_list=2402.10482 (authors, abstract, venue comment)
- **What it does:** The paper shows, for linear probing on fixed features, that multi-round self-distillation averages labels among instances with highly correlated features, which reduces the effect of label noise, and proposes a single-round variant whose student learns from partial labels formed by the teacher's top two softmax outputs.
- **Relation to FCT:** Analysis of how self-distillation reshapes the supervision signal, as FCT's identity is; here the teacher corrects noisy labels, whereas FCT trusts the label and uses it to judge the teacher.

### sun2025rld
- **Title:** Knowledge Distillation with Refined Logits
- **Authors:** Wujie Sun, Defang Chen, Siwei Lyu, Genlang Chen, Chun Chen, Can Wang
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2025), pp. 1110-1119
- **Year:** 2025
- **DOI:** 10.1109/iccv51701.2025.00111 (arXiv:2408.07703, comment "ICCV 2025")
- **Sources:** https://api.crossref.org/works/10.1109/iccv51701.2025.00111 ; https://doi.org/10.1109/iccv51701.2025.00111 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/iccv51701.2025.00111 and https://export.arxiv.org/api/query?id_list=2408.07703
- **What it does:** Refined logit distillation (RLD) uses the label to dynamically refine the teacher's logits, so that the teacher's incorrect predictions no longer conflict with the cross-entropy objective while class correlations are preserved, outperforming existing methods on CIFAR-100 and ImageNet.
- **Relation to FCT:** Label-aware handling of an incorrect teacher in classification; FCT also uses the label against an incorrect teacher, but per pixel, between views of one network, and by reweighting the pull rather than editing the teacher's output.

### lan2025labelrevision
- **Title:** Improve Knowledge Distillation via Label Revision and Data Selection
- **Authors:** Weichao Lan, Yiu-ming Cheung, Qing Xu, Buhua Liu, Zhikai Hu, Mengke Li, Zhenghua Chen
- **Venue:** IEEE Transactions on Cognitive and Developmental Systems, vol. 17, no. 6, pp. 1377-1388
- **Year:** 2025
- **DOI:** 10.1109/tcds.2025.3559881 (preprint arXiv:2404.03693)
- **Sources:** https://api.crossref.org/works/10.1109/tcds.2025.3559881 ; https://doi.org/10.1109/tcds.2025.3559881 (BibTeX) ; abstract: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/tcds.2025.3559881
- **What it does:** The method improves knowledge distillation by rectifying the teacher's inaccurate predictions with the ground truth (label revision) and by selecting which training samples the teacher supervises (data selection), reducing the impact of erroneous supervision, and it can be combined with other distillation approaches.
- **Relation to FCT:** Correctness-aware distillation: it corrects wrong teacher targets and drops unsuitable samples, whereas FCT keeps the teacher's target and scales each pixel's pull by comparing teacher and student with the label.

---

## Verification notes

- **Years.** BibTeX years follow the issue year given by doi.org. Three papers appeared online
  earlier: lu2025dclps (2024 online), du2025umnet (2024 online) and liu2025tracoco (arXiv
  2022).
- **Author name.** For lu2025dclps, Crossref gives the first author as "Ziru Lu" and Semantic
  Scholar as "Zirui Lu"; the publisher's Crossref form is used.
- **Venues without DOI.** yang2023dac (AISTATS, PMLR) and jeong2025selfdistill (ICLR) have no
  DOI. AISTATS acceptance and pages come from the PMLR v206 proceedings page, and ICLR
  acceptance from the arXiv comment. ICLR gives no pages.
- **Venue strength.** JBHI (zhang2024illumination), IEEE TCDS (lan2025labelrevision), CVIU
  (lee2023sddropout) and AISTATS (yang2023dac) are respected peer-reviewed venues but are
  not on the preferred list. They are kept because they fit the paper directly.
- **Checked and dropped (no usable abstract or off-topic abstract):**
  - Cross-level Feature Aggregation Network for Polyp Segmentation (Pattern Recognition
    2023, 10.1016/j.patcog.2023.109555): the Semantic Scholar abstract is truncated and
    there is no arXiv version.
  - Polyp-Mamba (MICCAI 2024, 10.1007/978-3-031-72111-3_48): no abstract available.
  - Progressive contour-consistent self-distillation for polyp segmentation
    (Neurocomputing 2026, 10.1016/j.neucom.2026.134234): no abstract available.
  - VPTTA (CVPR 2024, 10.1109/cvpr52733.2024.01063): a multi-centre test-time adaptation
    paper whose abstract does not mention polyps.
- **Verified alternates (record and abstract checked, not in `new_refs.bib`):**
  - Polyp-Mixer, IEEE TCSVT 2023, 10.1109/tcsvt.2022.3197643
  - CoInNet, IEEE TMI 2023, 10.1109/tmi.2023.3320151
  - MedSAM, Nature Communications 2024, 10.1038/s41467-024-44824-z
  - Rethinking polyp segmentation from an out-of-distribution perspective, Machine
    Intelligence Research 2024, 10.1007/s11633-023-1472-2
  - FBGANet, domain-generalizable polyp segmentation, Artificial Intelligence in Medicine
    2026, 10.1016/j.artmed.2025.103288
  - UniMatch V2, IEEE TPAMI 2025, 10.1109/tpami.2025.3528453
  - Bidirectional copy-paste, CVPR 2023, 10.1109/cvpr52729.2023.01108
  - AD-MT, ECCV 2024, 10.1007/978-3-031-72652-1_14
  - Reliability-aware semi-supervised gastric polyp segmentation, IEEE TIP 2026,
    10.1109/tip.2026.3731602
  - Logit standardization in KD, CVPR 2024, 10.1109/cvpr52733.2024.01489
  - Confidence maximization for mixed-supervision KD in segmentation, Medical Image
    Analysis 2023, 10.1016/j.media.2022.102670
