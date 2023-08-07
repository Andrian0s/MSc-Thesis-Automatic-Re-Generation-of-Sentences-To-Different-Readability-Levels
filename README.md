# This is the repository for the Master Thesis under [Mrinmaya lab](https://www.mrinmaya.io) at ETH Zurich titled Automatic Re-generation of Sentences To Different Readability Levels.

This repository is a continuation of:

# Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks
This repo contains the PyTorch implementation of the ACL, 2021 paper
[Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks](https://aclanthology.org/2021.acl-long.47.pdf).

The changes to this repository can be summarised as following:
- Added of the ability to customly initialisate the task embeddings with a representation of the readability
- Added the possibility to use separate Task Embedding Controllers between Encoder and Decoder networks.
- Added support for the onestop parallel dataset (sentence or text based) (outside of huggingface datasets)
- Added support for many metrics, including Language Models which evaluate fluency and preservence of original meaning
- Made some minor improvements throughout the codebase, including a nicer log system for automatic postprocessing of results

Future Work:
- Cleanup the code to remove unused changes that are no longer used
- Update the repository to use a more recent version of hugging face and Pytorch V2

In contrast to the original paper, all of our models were trained on a single T4 GPU and to make experimenting easier we created experiment pipeline experiments, found under the scripts folder.
