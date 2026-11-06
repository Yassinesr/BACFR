## Openers (first sentence), n = 1759

- other: 34%
- task definition or importance ("X aims to…", "X is a fundamental…"): 25%
- a problem or challenge: 18%
- starts with the paper itself ("We propose…", "In this paper…"): 14%
- recent progress ("Recent advances in…"): 9%

  e.g. Accurate segmentation of 3D vascular structures is essential for various medical imaging applications. | Segmentation of curvilinear structures is important in many applications, such as retinal blood vessel segmentation for early detection of vessel diseases and pavement crack segmentation for road condition evaluation and maintenance. | Placenta Accreta Spectrum (PAS) is a rare but highly dangerous obstetric disease.

  e.g. Transformers have shown remarkable performance in 3D medical image segmentation, but their high computational requirements and need for large amounts of labeled data limit their applicability. | Deep learning models for medical image segmentation can fail unexpectedly and spectacularly for pathological cases and images acquired at different centers than training images, with labeling errors that violate expert knowledge. | Annotating medical data for training AI models is often costly and limited due to the shortage of specialists with relevant clinical expertise.

  e.g. Recent studies on contrastive learning have achieved remarkable performance solely by leveraging few labels in the context of medical image segmentation. | The rapid development of deep learning has made a great progress in image segmentation, one of the fundamental tasks of computer vision. | In recent years, sparse voxel-based methods have become the state-of-the-arts for 3D semantic segmentation of indoor scenes, thanks to the powerful 3D CNNs.

## Theory sentences (T), verbatim sample, with the sentence that follows

166 theory sentences in 133 abstracts; position in the abstract (sentence index / length): median 0.50

- (TPAMI) **Learning-based image reconstruction models, such as those based on the U-Net, require a large set of labeled images if good generalization is to be guaranteed.** → _In some imaging domains, however, labeled data with pixel- or voxel-level label accuracy are scarce due to the cost of acquiring them._
- (TPAMI) **The theoretical analysis, coupled with the experimental results, can guide researchers in selecting the best method for their intended vision applications.** → _Finally, this study provides insights for potentially interesting future research directions to attract more attention from researchers to this field._
- (CVPR) **To mitigate the negative impact of noisy labels, a loss consistency-based collaborative optimization method is proposed, and its effectiveness is theoretically demonstrated.** → _Specifically, the label confidence parameters are introduced to dynamically adjust the influence of labels of different confidence levels during model training,_
- (ICCV) **Towards the former, we provide theoretical insights to develop a multi-head framework trained with a virtually extended multi-source dataset, aiming to balance generalization and specificity.** → _Towards the latter, we utilize the multi-head framework to extract reliable target pseudo-labels for self-training._
- (TIP) **However, to guarantee WSSS task feasible, we need to generate pseudo labels by expanding the seeds from CAMs which is complex and time-consuming, thus hindering the design of efficient end-to-end (single-stage) WSSS approaches.** → _To tackle the above dilemma, we resort to the off-the-shelf and readily accessible saliency maps for directly obtaining pseudo labels given the image-level clas_
- (IJCV) **The attention mechanism has been proven effective on various visual tasks in recent years.** → _In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Visio_
- (CVPR) **From an information theoretic standpoint, we prove that BackSplit increases the expected Fisher Information relative to conventional binary training, leading to tighter asymptotic bounds and more stable optimization.** → _With extensive experiments across multiple datasets and architectures, we empirically show that BackSplit consistently boosts small-lesion segmentation performa_
- (IJCAI) **EyeSeg outputs an uncertainty score and the segmentation result, weighting and fusing multiple gaze estimates for robustness, which proves to be effective especially under motion blur, eyelid occlusion and cross-domain challenges.** → _Moreover, empirical results suggest that EyeSeg achieves segmentation improvements of MIoU, E1, F1, and ACC surpassing previous approaches._
- (ICCV) **To this end, our theoretical analyses reveal that directly refining image embeddings is feasible to approach the same goal as parametric updates under the MedSAM architecture, which enables us to realize high computational efficiency and segmentation performance without the risk of catastrophic forgetting.** → _Under this framework, we propose to encourage maximizing factorized conditional probabilities of the posterior prediction probability using a proposed distribut_
- (ICML) **In this paper, we propose a novel cell instance segmentation method inspired by the four-color theorem.** → _By conceptualizing cells as countries and tissues as oceans, we introduce a four-color encoding scheme that ensures adjacent instances receive distinct labels._
- (NeurIPS) **Applied to a dataset of 373 facades with known dimensions from ten cities, the results show that installable BIPV potential is significantly lower than theoretical potential, thus providing valuable insights for reliable urban energy planning.** → _With the growing availability of facade imagery, the proposed pipeline can be scaled to support BIPV planning in cities worldwide._
- (AAAI) **Through theoretical analysis, we find the pre-trained large vision models are helpful in optical flow estimation, and we notice that the recently famous Segment Anything Model (SAM) demonstrates a strong ability to segment complete objects, which is suitable for solving the fragmentation problem.** → _We thus propose a solution to embed the frozen SAM image encoder into FlowFormer to enhance object perception._
- (NeurIPS) **As a consequence, good performance on standard benchmarks does not guarantee success in real-world scenarios.** → _To address these problems, we present Touchstone, a large-scale collaborative segmentation benchmark of 9 types of abdominal organs._
- (CVPR) **Given these circumstances, not only is data acquisition challenging, but increasing the frame rate for each dataset also proves difficult.** → _To address this challenge, this paper proposes a simple yet effective Unsupervised Volumetric Interpolation framework, UVI-Net._
- (TPAMI) **Our theoretical analysis confirms that the diffusion denoising process significantly enhances the model's ability to learn high-frequency features; however, we also observe that these models exhibit insufficient semantic inference for low-frequency features when guided solely by the original image.** → _Therefore, we integrate the strengths of both discriminative and generative learning, proposing the Integration of Discriminative and diffusion-based Generative_
- (CVPR) **To this end, a dual-teacher-single-student architecture is designed and tag-guided contrast is conducted, which guarantee the correctness of knowledge and further facilitate the discrepancy among co-contexts.** → _We streamline the multi-staged WSSS pipeline end-to-end and tackle this issue without external supervision._
- (CVPR) **We find that a) the more challenging task of panoptic segmentation usually results in worse performance while high performance variance between datasets and backbones indicates that generalization is not guaranteed, b) time series samples can be useful for specific configurations, but in many cases are not worth the cost, c) sample diversity shows the most promise in the downstream task of calibration, but otherwise fails to beat simpler alternatives, d) a deterministic approach is adequate for some downstream tasks, but ensembles allow for significant improvements if the right conditions can be achieved in deployment.** → __
- (TPAMI) **The MuCon loss proves beneficial even in the fully supervised setting.** → __
- (CVPR) **The effectiveness of our approach has been validated with significant performance improvements, better collaboration fairness, better performance fairness, and comprehensive analytical studies.** → __
- (ICCV) **We also prove the versatility of our HomoFusion approach by applying it to the problem of water puddle segmentation and achieving SOTA performance.** → __
- (CVPR) **To further ensure physical coherence, our Energy Consistency Loss enforces the work-energy theorem, aligning kinetic energy change with the work done by the net force.** → _The learned dynamics then drive a Spatio-Temporal Modulation module: Spatially, generalized forces are fused with spatial representations to provide more discri_
- (NeurIPS) **AMS adaptively partitions the KV cache based on the spatial distribution of attention mass, ensuring structurally vital reasoning segments receive guaranteed memory quotas.** → _To ensure stability during iterative decoding, an EMA-based smoothing mechanism is incorporated to prevent jitter in segment boundaries._

## What follows an insight or theory sentence (first three words of the next sentence)

"In this paper," (19); "To address this" (12); "To address these" (10); "In this work," (9); "To this end," (8); "Based on this" (7); "Building on this" (6); "To address this," (6); "We propose a" (5); "Extensive experiments show" (4); "We find that" (4); "Extensive experiments demonstrate" (3); "To bridge this" (3); "Experimental results on" (3); "Building on this," (2); "We also provide" (2); "To mitigate the" (2); "We achieve this" (2); "With extensive experiments" (2); "We also propose" (2); "We argue that" (2); "The code is" (2); "Motivated by these" (2); "In addition to" (2); "Experimental results demonstrate" (2)

connectives: "therefore" (10); "based on" (9); "building on" (8); "to this end" (8); "motivated by" (7); "thus" (3); "leveraging" (2); "following" (2); "building upon" (2); "guided by" (2); "in light of" (1); "inspired by" (1); "accordingly" (1); "hence" (1)

## Insight sentences (O), verbatim sample

- (ICCV) Additionally, we argue about the importance of having a prior that contains information about objects, or their parts, and discuss several possibilities to obtain such a prior in an unsupervised manner.
- (CVPR) Interestingly, we find that gradient-based influence alone favors easy-to-optimize samples that cause large parameter shifts but lack deep reasoning chains, while difficulty alone selects noisy or overly complex textual cases that fail to guide stable optimization.
- (CVPR) We find that while recent models are significantly more robust, they are not overall more reliable in terms of uncertainty estimation.
- (NeurIPS) We surprisingly observe that, solely trained with synthetic images, we already achieve comparable performance with real ones (e.g., 48.3 vs. 48.5 mIoU on ADE20K, and 49.3 vs. 50.5 on COCO-Stuff).
- (ICML) However, we argue that the current progress is clouded by a twofold overconfidence problem.
- (AAAI) Based on our analysis, we define a disambiguation rewriting task to regenerate an input to be unambiguous while preserving information about the original content.
- (ACMMM) We investigate transductive zero-shot point cloud semantic segmentation, where the network is trained on seen objects and able to segment unseen objects.
- (CVPR) We identify a key obstacle to this unified formulation: VAE latents of binary masks are sharply distributed, noise robust, and linearly separable, distinct from natural image latents.
- (NeurIPS) Our key insight is that the success of every known MIA is fundamentally bounded by the local batch size relative to the model's leakage capacity; once this limit is exceeded, distinct samples collide and reconstructions collapse into indistinguishable mixtures.
- (TPAMI) We study the problem of efficient semantic segmentation of large-scale 3D point clouds.
- (ICCV) We show that exploiting available camera intrinsic data and ground plane assumption for cross-frame correspondence can lead to a light-weight network with significantly improved performances in speed and accuracy.
- (AAAI) In addition, we analyze the entropy of USS features, which is the first step towards understanding USS from the perspective of information theory.
- (AAAI) Although post-training quantization (PTQ) offers a promising compression solution, existing methods yield unsatisfactory results when applied to SAM, owing to its specialized model components and promptable workflow: (i) The mask decoder's attention exhibits extreme activation outliers, and we find that aggressive clipping (even 100x), without smoothing or isolation, is effective in suppressing outliers while maintaining performance.
- (TPAMI) A key idea to tackle this problem is to perform both image-level and feature-level adaptation jointly.

## Gap sentences (G): first two words

"However, the" (81); "However, existing" (65); "However, these" (33); "Despite the" (24); "However, they" (24); "This paper" (22); "Existing methods" (22); "However, it" (19); "However, most" (18); "However, such" (18); "To address" (17); "However, current" (16); "However, this" (15); "We address" (14); "However, their" (14); "However, in" (13); "Semantic segmentation" (12); "In addition," (12); "However, due" (12); "Existing approaches" (12)

## Result sentences (R): first three words, and what they claim

"Extensive experiments on" (156); "Extensive experiments demonstrate" (82); "Extensive experiments show" (51); "Experimental results demonstrate" (40); "Experimental results on" (38); "Experimental results show" (29); "Experiments show that" (28); "Extensive experimental results" (24); "We demonstrate the" (21); "We evaluate our" (15); "Experiments demonstrate that" (14); "We conduct extensive" (14); "Extensive experiments conducted" (13); "We demonstrate that" (13); "Extensive experiments are" (12); "Extensive experiments across" (12); "The experimental results" (11); "Our method achieves" (10); "Our experimental results" (10); "Through extensive experiments" (10)

- name a count of datasets/benchmarks ("five benchmarks", "three datasets"): 11%
- "consistently": 8%; "across": 12%; "state-of-the-art": 51%; "outperform": 27%; "significantly": 14%
- a number in the result sentence: 18%

medical result sentences, sample:
- (ICCV) Notably, on Kvasir, it achieves 76.93% IoU, surpassing the state-of-the-art by 11.44%.
- (AAAI) Utilizing the commonly used application of diabetic retinopathy, we first demonstrate that state-of-the-art SDG techniques fail to achieve generalized performance across data domains.
- (NeurIPS) Extensive experiments on the LC dataset and other public benchmarks demonstrate the effectiveness of our method, which outperforms state-of-the-art approaches.
- (AAAI) Extensive experiments show that our WEFT not only outperforms 21 state-of-the-art (SOTA) methods on three ORSIs datasets, but also achieves optimal results in camouflage, natural, and medical scenarios.
- (CVPR) Extensive experiments on Computed Tomography (CT) and Magnetic Resonance (MR) denoising demonstrate that RelativeFlow significantly outperforms existing methods, taming MID with noisy references.
- (ICCV) We demonstrate state-of-the-art geometric and correspondence accuracy, while only taking 0.385 seconds to compute a mesh with 10K vertices, which is three orders of magnitude faster than traditional techniques.
- (NeurIPS) In a series of experiments on seven held-out datasets, we demonstrate that our model can significantly outperform existing foundation models in producing several plausible whole-image segmentations, that are semantically coherent across images.
- (IJCAI) Experimental results on different benchmarks demonstrate that our Scale-Former outperforms the current state-of-the-art methods.
- (TIP) Notably, on the Pancreas dataset, CVBM outperforms fully supervised methods (i.e., DSC: 84.57% vs. 83.89%) while utilizing only 20% of the labeled data.
- (ACMMM) We evaluate ScribbleVC on three benchmark datasets and compare it with state-of-the-art methods.

## Numbers when present: sample of sentences with numbers

- (TPAMI) Without applying any simulated data for pre-training, our CFBI+ achieves the performance (J&F) of 82.9% and 82.8%, outperforming all the other state-of-the-art methods.
- (CVPR) Experimental results demonstrate that CALICO, with just 0.3% of its parameters finetuned, achieves strong performance on this challenging task.
- (ICML) Compared to a previous published ViT-L/8 MAE, our new model achieves a 60% improvement in linear separability of genetic perturbations and obtains the best overall performance on whole-genome biological relationship recall and replicate consistency benchmarks.
- (ICCV) Our approach surpasses existing methods in terms of effectiveness on various datasets, including ImageNet, UCF-101, and Kinetics-400.
- (NeurIPS) On a single A100, DIGS completes mapping and instance encoding on LERF-OVS in 1.0-3.2 min per scene, depending on whether LLM fusion is enabled, and serves native 3D queries in 5.4 ms.
- (ICML) ARTIST improves average accuracy by 6.46 absolute percentage points over the strongest baseline.
- (ACMMM) Extensive experiments on two widely-used synthetic-to-real transfer tasks show that the proposed approach surpasses previous state-of-the-art by up to 5.25% of mIoU with fair model selection criteria.
- (CVPR) Experimental results demonstrate that fine-tuning VLA policies on AOMGen data increases the success rate from 0% to 88.7%, and the policies are tested on unseen objects and layouts.
- (NeurIPS) On LFQ-1024, retuning the D3PM and SEDD samplers on a held-out validation split reduces FID-192 from 0.44/0.41 at the default budget to 0.09/0.10 at lower NFE, replicated across seeds, although the continuous references were not given an equivalent sampler sweep.
- (NeurIPS) Our code is available at https://github.com/blue-531/pref-ovss.

## Titles

- with a colon ("Name: description"): 54%
- question titles: 1.1% (20)
- median words: 9
- question titles: OMG-Seg: Is One Model Good Enough For All Segmentation ? | Reliability in Semantic Segmentation : Are We on the Right Track? | How Much Temporal Long-Term Context is Needed for Action Segmentation ? | To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation | Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts | Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation ? | Promising or Elusive? Unsupervised Object Segmentation from Real-world Single Images | Anatomical and Diagnostic Bayesian Segmentation in Prostate MRI $-$Should Different Clinical Objectives Mandate Different Loss Functions? | Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation ? | How Do Images Align and Complement LiDAR? Towards a Harmonized Multi-modal 3D Panoptic Segmentation | Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation | Market or Markets? Investigating Google Search's Market Shares Under Horizontal and Vertical Segmentation | Is Visual in-Context Learning for Compositional Medical Tasks within Reach? | Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering | How Does Diverse Interpretability of Textual Prompts Impact Medical Vision-Language Zero-Shot Tasks? | Deep Learning in Medical Image Registration: Magic or Mirage? | Do Large Language Models have Shared We

## How "no inference cost" and generality claims are phrased

- (TPAMI) As verified in extensive experiments using public datasets, we demonstrate that Contextrast++ substantially improves semantic segmentation performance over existing contrastive learning-based state-of-the-art approaches, while introducing no additional computational overhead during inference.
- (TPAMI) The experiments reveal strong open-set performance in spite of negligible computational overhead.
- (TIP) Experiments demonstrate that DA-Cal seamlessly integrates with existing self-training frameworks across multiple UDA segmentation benchmarks, significantly improving target domain calibration while delivering performance gains without inference overhead.
- (TIP) Combining these two modules, DSD framework has no extra parameters and only a minimal increase in FLOPs.
- (TMM) Our REFL-Net achieves 1.8% mIoU gain on ScanNetV2 and 1.7% mIoU gain on S3DIS with negligible computational cost compared with backbone models.
- (CVPR) To support efficient reconstruction, we design a lightweight auxiliary module, Rebuilder, which is jointly trained with the segmentation network but removed during inference, introducing zero test-time overhead.
- (CVPR) SCA requires no additional parameters, enabling seamless incorporation into current DiT-based architectures.
- (CVPR) Our proposed method, SAMWISE, achieves state-of-the-art across various benchmarks, by adding a negligible overhead of less than 5 M parameters.
- (CVPR) The aggregator re-parameterizes interpolation-first modules in a convolution-first way, which significantly speeds up the pipeline without any additional costs.
- (CVPR) Importantly, the TP path can be freely removed after training, leading to no additional inference cost.
- (ICCV) Extensive experiments on four datasets, including ADE20K, Cityscapes, COCO-Stuff-164K, and Pascal Context, demonstrate consistent improvements with negligible parameters.
- (ICCV) Training-free open-vocabulary semantic segmentation (OVS) aims to segment images given a set of arbitrary textual categories without costly model fine-tuning.
