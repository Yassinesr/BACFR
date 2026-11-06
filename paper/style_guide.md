# Writing guide for the FCT paper, measured on CCF-A papers

Every section of our paper follows this guide. Every number here was measured, not recalled. There are two sources, both collected from arXiv on 2026-11-05, keeping only papers whose authors' comment or journal reference says the paper was **accepted** at a CCF-A venue:

- **Abstracts:** 1,759 papers from TPAMI, IJCV, TIP, TMM, CVPR, ICCV, NeurIPS, ICML, AAAI, IJCAI and ACM MM, of which 1,212 are segmentation papers and 395 medical.
- **Full texts:** 207 journal papers (TPAMI, IJCV, TIP, TMM) and 149 conference papers, parsed section by section from arXiv's HTML. The conference papers are medical, insight-first and random CVPR/ICCV/NeurIPS papers.

We are writing a **journal** paper, so the targets below use the **journal medians**. Conference numbers appear only for contrast.

---

## Part 0. The paper at a glance

| Item | CCF-A journal median (IQR) | Target for FCT |
|---|---|---|
| Pages (from the authors' comment) | 15.5 | 14–16 |
| Body words (no captions, references or appendix) | 6,974 (5,919–8,250) | about 7,000 |
| Top-level sections | 6 | 7 (an analysis section is added, see Part 2) |
| Subsections | 13 | 12–14 |
| Figures | 10 (7–12) | 8–9 |
| Tables | 9 (6–12) | 8–9 |
| Display equations (lines) | 13 (9–19) | 15–20 (theory-led) |
| Theorem-type blocks | 4% of papers have any | 1 proposition and 1 corollary |
| References | 74 (60–93) | 60–80 |
| References from the last 3 years | 47% | about half |
| Appendix | 17% of journal papers (54% of conference papers) | yes: proofs and per-seed tables |
| Words per sentence | 21.7 | about 22 |
| Sentences per paragraph | 5.2 | 4–6 |
| Citations per 1,000 words | 13.3 | about 13 |

**Where the words go** (median share of body words):

| Section | Share | Words |
|---|---|---|
| Introduction | 14% | 980 |
| Related work | 11% | 780 |
| Method (including preliminaries and analysis) | 28% | 1,990 |
| Experiments | 38% | 2,620 |
| Discussion (when present) | 5% | 350 |
| Conclusion | 2% | 157 |

Experiments are the largest part of a CCF-A journal paper. The method comes second.

**Skeleton:**

| Order | Journal papers |
|---|---|
| Introduction → Related work → Method → Experiments → Conclusion | 53% (109 of 207) |
| A Discussion or Limitations section added before or after the Conclusion | 17% |
| A Preliminaries section before the Method | 11% |
| A separate Analysis/Theory section | 6% |

Related work, present in 91% of papers, comes right after the introduction and never at the end.

---

## Part 1. The story: from abstract to references

A CCF-A paper asks one question and answers it section by section. Each section hands the next one a claim to support:

| Section | Job | What it hands over |
|---|---|---|
| Title | names the idea and the task | the promise |
| Abstract | the whole argument in about 200 words | that the paper is worth reading |
| Introduction | funnel: context → gap → (insight) → proposal → how → evidence → contributions | the claims to be proved |
| Related work | places the idea among its neighbours and ends each theme with how ours differs | what is new |
| Analysis / theory | the result that makes the method necessary | the principle |
| Method | the rule derived from that principle, with every symbol defined | what to test |
| Experiments | each claim tested, in the order the claims were made | the evidence |
| Discussion | what the evidence means, where it stops, and the limits | an honest scope |
| Conclusion | one paragraph: problem, idea, finding, next step | closure |
| References and appendix | the sources, the proofs and the per-seed numbers | trust |

The introduction funnel, measured paragraph by paragraph in 207 journal introductions:

| Position in the introduction | Role |
|---|---|
| 0.17 (start) | context paragraphs (about 2 per introduction) |
| 0.29 | the gap ("However, …"), in 64% of introductions as its own paragraph |
| 0.50 | an insight or observation paragraph, rare (9%) |
| 0.62 | the proposal ("In this paper, we propose …") |
| 0.67 | how it works |
| 0.69–0.83 | an evaluation summary |
| 1.00 (end) | the contributions paragraph or bullets; a roadmap in 18% |

### Our arc: theory, then maths, then engineering

1. **Observation** (introduction, Figure 1). Orientation means nothing in colonoscopy, yet trained networks are flip-inconsistent, and most of all on the unseen sets where they fail.
2. **Gap** (introduction). Consistency regularization comes from semi-supervised learning. With labels it can pull a correct view toward a wrong one.
3. **Theory** (Section 3). Proposition 1, the identity: a consistency pull is a reweighted label pull. Corollary 1, the lag condition: a pull helps only when the teacher view is closer to the label.
4. **Maths to rule** (Section 4). The lag-or-margin weight follows from Corollary 1, its constant 2 is forced (Proposition 2 or a remark), and the upright pull completes it.
5. **Engineering proof** (Section 5). Each theoretical claim becomes a prediction, tested in turn:
   - **P1:** FCT beats each model's own vanilla on every seed.
   - **P2:** FCT shrinks the flip gap, which is the mechanism.
   - **P3:** the label-aware weight beats naive consistency.
   - **P4:** the derived constant beats the alternatives.
   - **P5:** the gain is more than flip augmentation.
6. **Discussion** (Section 6). Why the gains concentrate on unseen data, the cost, and the limits: binary labels only, and the selection protocol.

The data supports this order: theory-first papers place one or two formal results *inside or before* the method and keep the proofs short or in an appendix. Examples:
- **Margin Calibration (IJCV 2021):** Method = Problem setup and notations → Method overview → *Theoretical motivation* (1 theorem, 1 corollary) → Practical implementation → Complexity analysis.
- **ARCO (NeurIPS 2023, medical):** 1 lemma and 1 theorem in the method, with the proofs in a 3,500-word appendix.
- **EM motion segmentation (TPAMI 2022):** a short formulation section that opens "The core idea of our work is to leverage the EM framework to design in a well-founded manner the loss …"

---

## Part 2. Section by section

### 2.1 Title
- 54% of CCF-A titles are "Name: what it does", with a median of 9 words.
- Only 1.1% are questions, and almost all of those are analysis or benchmark papers.
- **Rule:** state the method and the task. A question title is a deliberate choice for an analysis-led paper, not the default.

### 2.2 Abstract (1,759 abstracts)
- **Length:** median 185 words (journals 201), 8–9 sentences of about 22 words each.
- **Order:** the standard is context, gap, proposal, method, results, code. The insight-first order is used by 10% of abstracts (IJCV 16%, ICML 17%), which state the finding or result before the method. **Ours is insight-first.**
- **Maths:** 13% contain any symbol. Theory is stated as what it means.
- **Numbers:** 29% quote any, at a median of 2; the norm is one headline figure after "Notably,".
- **Endings:** 47% end with code availability and 33% with a result in words.
- **Tense and voice:** present tense ("we propose" 1,970 times against 4 for "we proposed") and active "we".

### 2.3 Introduction (journal median 980 words, 7 paragraphs, 14% of the body)

| Feature | Journal papers |
|---|---|
| "However" | 87% |
| A figure in the introduction (Figure 1, the teaser) | 85% |
| A reference to a figure | 86% |
| A contributions list in bullets | 76%, median 3 items |
| The word "contribution(s)" | 92% |
| "novel" | 53% |
| "the first" / "for the first time" | 43% |
| "we observe / find / show" | 30% (54% in insight-first conference papers) |
| "to the best of our knowledge" | 14% |
| A roadmap paragraph ("The rest of this paper …") | 18% |
| A question | 12% |
| A display equation | 0% |

**Paragraph template for FCT** (7 paragraphs and 3 bullets, about 1,000 words):
1. **Context:** why polyp segmentation matters and why unseen centres break it (with citations).
2. **Observation:** colonoscopy is orientation-free, yet models are flip-inconsistent, and most on unseen sets. Refer to Figure 1.
3. **Gap:** augmentation and consistency regularization. Their semi-supervised roots mean they ignore which view is correct; consequence: pulls toward wrong views.
4. **Insight, theory in words:** with labels, every consistency pull is a reweighted label pull. It helps only when the teacher view is closer to the label. No equations here.
5. **Proposal:** "In this paper, we propose …". What FCT does, in words: lag-or-margin, the constant that follows from the analysis, and the upright pull. It plugs in and costs nothing at inference.
6. **Evidence summary:** three architectures, five test sets, six seeds, gains largest on unseen data, and a smaller flip gap.
7. **"Our contributions are summarized as follows:"** three bullets, each a claim backed later:
   - the analysis;
   - the method;
   - the evaluation.

   Each bullet opens with a verb ("We show …", "We propose …", "We demonstrate …").

### 2.4 Related work (journal median 780 words, 6 paragraphs, 11%; 91% of papers have it, always early)
- **Citation density:** 39 per 1,000 words, three times the paper average. "We" is rare (2.7 per 1,000 words).
- **Organisation:** 2–4 themed subsections or bold run-in paragraph heads. Ours:
  1. polyp segmentation and generalization to new centres;
  2. consistency regularization and equivariance;
  3. self-distillation and correctness-aware distillation.
- **Closing:** 71% of related-work sections contrast prior methods with their own ("Unlike …", "Different from …", "In contrast, …"). **Rule:** end every subsection with one sentence on how FCT differs.

### 2.5 Analysis / theory section (rare as its own section: 6%; theorem blocks in 4% of journal papers, 15% of conference papers and 28% of insight-first conference papers)
- **Placement:** after related work and before the method, or as the first subsection of the method ("Problem setup and notation" → "Theoretical motivation").
- **Order inside:**
  1. notation;
  2. one proposition, stated in words before the symbols;
  3. the proof, if it is short; otherwise a sketch here and the full proof in the appendix;
  4. one corollary that gives the design rule;
  5. a remark that reads the result back in plain language.

  One illustration figure carries the intuition, for example the direction of the pull as a function of teacher and student error.
- **Restraint:** at most two formal blocks. The rest is prose with numbered equations.
- **Insight-first papers** use more formal structure than average: a separate theory section in 18%, definitions in 16% and theorem blocks in 28%, with proofs in the appendix in 75%. That is still a minority, so our two blocks are well within the norm.

### 2.6 Method (journal median 1,990 words including preliminaries, 19 paragraphs, 5 subsections, 28%)

| Feature | Journal papers |
|---|---|
| An overview figure in the method | 91% |
| Display equations | 12 (6.1 per 1,000 method words) |
| Equations numbered | 84% |
| "where … denotes" definitions per equation | about 2 |
| Passive voice | highest of any section (14 per 1,000 words) |
| Present tense | 96% |

**Rule:** start with an overview paragraph and the overview figure, then one subsection per component:
1. three supervised views;
2. the lag-or-margin weight (derived from Corollary 1);
3. the upright pull;
4. the full objective and training (λ schedule, shared stochastic depth, no inference change).

Define every symbol the first time it appears ("where p_o denotes …").

### 2.7 Experiments (journal median 2,620 words, 24 paragraphs, 8 tables, 4 figures, 38%)

| Subsection covers | Journal papers | Medical papers |
|---|---|---|
| Datasets, setup and metrics | 88% | 82% |
| Ablation | 79% | 69% |
| Comparison with other methods | 73% | 62% |
| Implementation details | 52% | 38% |
| Further analysis | 44% | 50% |
| Qualitative / visualization | 29% | 18% |
| Generalization / robustness | 14% | 14% |
| Efficiency / complexity | 11% | 4% |
| Reports spread or repeats (±, std, seeds, significance) | 23% | 42% |

- Results are referred to about 9 times per experiments section ("As shown in Table 2", about 1.3 per 1,000 words).
- **Rule:** present tense for what tables show ("Table 2 shows that …") and the past only for what was done in a specific run. Lead each subsection with its finding, then the evidence.

**For FCT**, order the subsections by the predictions in Part 1:
1. setup: datasets, the 1,288-image training set, metrics, six seeds, paired statistics and the selection protocol, disclosed;
2. implementation: three hosts, each with its own recipe and no retuning;
3. main results (P1);
4. mechanism: flip gap (P2);
5. ablations: the weight variants (P3), the constant sweep (P4) and the augmentation control (P5);
6. qualitative examples;
7. cost.

Our six-seed paired design is something only 23% of journal papers (42% of medical papers) report in any form. Show it in every table (mean ± SD and the paired 95% CI).

### 2.8 Discussion and limitations (a separate section in 32% of journal papers: discussion 27%, limitations 7%)
- About 350 words (journal median), in short themed paragraphs, often with bold run-in heads. StealthMark (TIP 2026) uses: False-positive analysis, Robustness, Out-of-distribution generalization, Clinical harmlessness, Limitations and Future Work.
- **For FCT:**
  - why the gains concentrate on unseen sets;
  - the binary-label scope of the identity;
  - the cost (about 2× training time, none at inference);
  - the protocol limits.

### 2.9 Conclusion (journal median 157 words, 1 paragraph, 2%)
- 59% mention future work and 19% limitations.
- 82% have no citations and no equations.
- It is the only section where past tense is common (31% "we proposed / introduced").
- **Rule:** one paragraph:
  1. the problem;
  2. what we showed (the identity);
  3. what we built (FCT);
  4. what it achieved, in words;
  5. one forward-looking sentence.

### 2.10 References (journal median 74, IQR 60–93; 47% from the last three years)
- **Rule:** 60–80 references, about half from 2023–2026. Every claim of "existing methods …" cites at least two.

### 2.11 Appendix (17% of journal papers, 54% of conference papers)
- **For FCT:**
  - the proofs of Propositions 1–2;
  - per-seed tables for each host;
  - the full sweep;
  - extra qualitative results.

---

## Part 3. Figures (journal median 10 per paper, IQR 7–12)
- **Placement:**
  - Figure 1 is in the introduction in 85% of papers.
  - An overview figure is in the method in 91%.
  - About 5 figures go in the experiments.
- **Figure 1 shows:**

  | Content | Share of papers |
  |---|---|
  | The problem or motivation | 34% |
  | A comparison with existing methods | 31% |
  | Visual examples | 25% |
  | A method overview | 23% |
  | A performance or trade-off plot | 16% |

  Its caption is the longest in the paper (median 58 words).
- **Captions:**
  - median 38 words, with 73% of two or more sentences;
  - self-contained: first what is shown, with panels (a), (b), (c), then what to notice ("The results demonstrate that …");
  - method overview captions open "Overview of the proposed X." and walk through the flow in 2–4 sentences.
- **For FCT** (8–9 figures):
  1. Figure 1, teaser: flip gap against Dice per test set for vanilla and FCT, plus one image where the vanilla mask changes under a flip and the FCT mask does not.
  2. Figure 2: the overview of the three views and four pulls.
  3. Figure 3: the theory, i.e. the pull direction over teacher and student error.
  4. Figure 4: the weight map of lag-or-margin.
  5. Experiments: the flip gap per set, the constant sweep, the per-seed gains and qualitative masks.

## Part 4. Tables (journal median 9 per paper, 8 in the experiments)
- **Captions:**
  - above the table in 90% of journal papers (IEEE style);
  - median 23 words;
  - 47% name the dataset or metric;
  - 22% explain the bold or underline marks.
- **Caption types:** comparison ("Comparison with … on …", 36%) and ablation ("Ablation study on …", 24%).
- **Rule:** title, then dataset and metric, then what bold marks. Examples:
  - "Ablation study on numbers of sampling points in CEM."
  - "Numbers in bold and italics indicate the best and the second-best scores, respectively."
- **For FCT:**
  - a main table per host (mean ± SD, paired difference with 95% CI, seeds better);
  - a flip-gap table;
  - a weight-variants table;
  - a sweep table;
  - an augmentation-control table;
  - a cost table.

## Part 5. Formulas
- **Density:** 13 display equations per journal paper (IQR 9–19), about 2 per 1,000 words overall and 6 per 1,000 in the method. Inline maths runs at about 21 per 1,000 words.
- **Where they go:**
  - never in the abstract (87% have no symbol);
  - never in the introduction (no journal introduction has one);
  - never in the conclusion (82%);
  - only in theory and method.
- **Conventions:**
  - number the display equations (84%) and refer to them by number;
  - follow each with its definitions ("where … denotes …", about 2 per equation);
  - state the idea in words before the symbols.
- **Theorem-type blocks** are rare: 4% of journal papers, 15% of conference papers and 28% of insight-first conference papers. Use at most two (Proposition 1, Corollary 1).
- **Proofs:** journal papers mostly keep them in the main text (4 of the 6 with proofs); conference papers move them to the appendix (11 of 14). For us: a short proof in the text, a long one in the appendix.

## Part 6. Tense and voice, by section (journal papers)

| Section | Present tense ("we use" against "we used") | "we" per 1,000 words | Passives per 1,000 words |
|---|---|---|---|
| Introduction | 98% | 12.7 | 9.3 |
| Related work | 98% | 2.7 | 8.6 |
| Method | 96% | 14.3 | 14.1 |
| Experiments | 91% | 15.6 | 9.5 |
| Discussion | 84% | 11.6 | 9.4 |
| Conclusion | 69% | 19.8 | 6.2 |

**Rule:** present tense for claims, method and what tables show. Past only for one-off actions ("we trained each model with six seeds") and in the conclusion's recap. Use active "we" in the introduction, method, experiments and conclusion, and keep related work about the cited works.

## Part 7. Density and rhythm
- Sentences of 21–24 words and paragraphs of about 5 sentences in every section.
- Citations per 1,000 words: related work 39, introduction 14, experiments 12, method 6, conclusion 0.
- Cross-references per paper: about 16 to figures, 12 to tables and 4 to sections.
- **Rule:** one idea per paragraph, opened by a topic sentence that states it. The first sentences of a section's paragraphs, read alone, should tell its story.

## Part 8. Novelty language
- "novel" appears in 71% of journal papers, a median of 3 times each.
- "The first" appears in 43% of journal introductions.
- "To the best of our knowledge" appears in 23% of journal papers. Of its 85 uses there, 50 are in introductions, 17 in related work, 10 in conclusions and 8 in experiments.
- **Rule for FCT:** claim novelty precisely and once per idea.
  - Use "to the best of our knowledge" once, in the introduction, for the identity and the label-aware weight. The novelty check supports this.
  - Use "novel" at most three times.
  - Never use "state-of-the-art" for FCT itself: we improve each model's own baseline and do not claim a new state of the art.

## Part 9. The FCT paper plan, with word budgets

| Section | Content | Words | Figures | Tables | Equations |
|---|---|---|---|---|---|
| Abstract | insight-first, done | 217 | | | 0 |
| 1 Introduction | 7 paragraphs and 3 bullets | 1,000 | 1 | | 0 |
| 2 Related work | 3 themes, each ending with a contrast | 800 | | | 0 |
| 3 When does flip consistency help? | notation, Proposition 1, Corollary 1, remark | 700 | 1 | | 6–8 |
| 4 Label-aware flip-consistency training | overview, three views, lag-or-margin (why 2), upright pull, objective | 1,500 | 2 | | 8–10 |
| 5 Experiments | setup, implementation, P1–P5, qualitative, cost | 2,600 | 3–4 | 6–7 | 0–2 |
| 6 Discussion and limitations | 4 themed paragraphs | 400 | | | 0 |
| 7 Conclusion | 1 paragraph | 150 | | | 0 |
| References | | 60–80 entries | | | |
| Appendix | proofs, per-seed tables, sweep | | | 2–3 | |
| **Total** | | **about 7,150** | **8–9** | **8–10** | **15–20** |

## Part 10. Checklist for every section draft
- [ ] Its first sentence states the section's point, and each paragraph opens with a topic sentence.
- [ ] Its length is within the budget in Part 9, with sentences of about 22 words and paragraphs of about 5 sentences.
- [ ] Present tense, active "we", every symbol defined at first use.
- [ ] Citation density fits the section (related work about 40 per 1,000 words, others 6–14).
- [ ] Every figure and table is referred to in the text before it appears and has a self-contained caption (figures about 40 words; tables above, about 23 words).
- [ ] Every claim made in the introduction is tested in the experiments, in the same order.
- [ ] Novelty is stated once, precisely, with no "state-of-the-art" for FCT itself.
