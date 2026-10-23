# Third-party components

- **Smoothie**: HazyResearch/smoothie, commit
  `8f8bd153058d3551955cf9fd62a501ee1e6a324f`.
  The estimator is in `src/saga/vendor/smoothie.py`.
  License: `third_party/Smoothie_LICENSE` (MIT).
- **Crowd-Kit**: Toloka/crowd-kit, commit
  `c6717c40560cd8e675fc1dda2ab5461295657d07`.
  `src/saga/baselines.py` adapts the RASA/HRRASA updates to batched arrays.
  License: `third_party/CrowdKit_LICENSE` (Apache 2.0).

- **BAU**: Braylan and Lease, *Modeling and Aggregation of Complex Annotations
  via Annotation Distances* (WWW 2020). The batched implementation follows the
  empirical-prior mean-distance estimator in Praznat/annotationmodeling,
  commit `96772932ca0de649c4abb13d5cfb053040bcab36`.
  License: `third_party/AnnotationModeling_LICENSE` (BSD-3-Clause).
- **AQER**: Yahav et al., doi:10.1287/isre.2023.0426. The implementation uses
  iterative min-max-scaled source grades and weighted semantic proxies,
  with E5-large-v2 response embeddings.
- **UAS**: Agrawal et al., *Uncertainty-Aware Answer Selection for Improved
  Reasoning in Multi-LLM Systems* (Findings of EMNLP 2025). The voting rule
  follows the main entry in Aakriti05/multi-llm-uncertainty,
  commit `887ca37f6808c2999a9249c67e39ba1973c6a15c`.
- **USC**: Chen et al., *Universal Self-Consistency for Large Language Model
  Generation*, arXiv:2311.17311. The consistency-selection prompt is applied
  to four collected candidates with Qwen as selector.

External datasets, model checkpoints, and dependencies retain their own licenses.
