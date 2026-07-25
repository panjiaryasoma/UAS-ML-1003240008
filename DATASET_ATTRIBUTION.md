# Dataset Attribution

## Dataset

This project uses the **SmSA (Sentiment Analysis)** dataset from the
**IndoNLU** benchmark for Indonesian text sentiment classification.

- Upstream project: IndoNLU
- Upstream repository: https://github.com/IndoNLP/indonlu/tree/master
- Task used: SmSA
- Source files used:
  - `train_preprocess.tsv`
  - `valid_preprocess.tsv`
- Labels used:
  - `negative`
  - `neutral`
  - `positive`

The original dataset files are not committed to this repository. They are
downloaded or restored into the local `data/` directory by the project setup
workflow, while `data/` remains excluded through `.gitignore`.

## Local Processing

For this project, the two labeled source splits are loaded, validated, and
combined for modeling. Exact duplicate rows are removed deterministically.
The resulting data is then divided using a stratified 80:20 split with
`random_state=42`.

The local preprocessing and modeling steps do not claim ownership over the
original dataset text or labels.

## License Notice

The upstream IndoNLU repository contains a `LICENSE` file with the full
Apache License 2.0 text and upstream copyright notices. A verbatim copy of
that file is included at the root of this repository as `LICENSE`.

At the time this attribution file was prepared, the upstream README license
badge and the upstream `LICENSE` file were not fully consistent. Therefore,
this project preserves the upstream license file verbatim and does not make a
broader claim about rights beyond the notices supplied by the upstream
repository.

Any redistribution of dataset-derived files should retain the applicable
license and attribution notices. Files containing original dataset text, such
as detailed prediction audit exports, should be reviewed before publication.

## Citation

Users of IndoNLU components are requested by the upstream project to cite:

> Bryan Wilie, Karissa Vincentio, Genta Indra Winata, Samuel Cahyawijaya,
> Xiaohong Li, Zhi Yuan Lim, Sidik Soleman, Rahmad Mahendra, Pascale Fung,
> Syafri Bahar, and Ayu Purwarianti. 2020. IndoNLU: Benchmark and Resources
> for Evaluating Indonesian Natural Language Understanding. Proceedings of
> AACL-IJCNLP 2020.

BibTeX:

```bibtex
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```

## Project Use

The dataset is used solely as the source corpus for the UAS Machine Learning
End-to-End project **SentimenID API** by Panji Arya Soma
(`UAS-ML-1003240008`).
