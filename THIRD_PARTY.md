# Third-party sources / Сторонние источники

No model weights, tokenizers or source datasets are distributed in this repository. The Docker build retrieves the following demo assets from upstream:

- [M-FAC/bert-tiny-finetuned-sst2 (pinned revision)](https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2/tree/3545b172246737cf07975f902be6bf73adff900d) — the pinned model card does not specify an explicit redistribution license. See the [M-FAC paper](https://arxiv.org/abs/2107.03356).
- [GLUE / SST-2](https://huggingface.co/datasets/nyu-mll/glue) — upstream dataset card marks the license as `other`; review the original dataset terms, including the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

Dependencies are installed from their own distributions. Consult their individual licenses; no claim is made that a project-code license covers third-party models, datasets or dependencies. Historical benchmark code may retrieve additional pretrained models and datasets; their terms apply separately.

The [Apache-2.0 project-code license](LICENSE) does not grant rights to third-party assets. Do not redistribute the prebuilt image, weights or datasets until their terms have been verified.
