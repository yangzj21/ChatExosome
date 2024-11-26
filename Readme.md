# ChatExosome: an artificial intelligence (AI) Agent based on Deep Learning of Exosomes Spectroscopy for Hepatocellular Carcinoma (HCC) Diagnosis 

## Abstract

Large language models (LLMs) hold significant promise in the field of medical diagnosis. While there are still many challenges in the direct diagnosis for hepatocellular carcinoma (HCC). Alpha-fetoprotein (AFP) is a commonly used tumor marker for liver cancer. However, relying on AFP can result in missed diagnoses of HCC. We developed an artificial intelligence (AI) agent centered around LLMs, named ChatExosome, which created an interactive and convenient system for clinical spectroscopic analysis and diagnosis. ChatExosome consists of two main components: the first is the deep learning of the Raman fingerprinting of exosomes derived from HCC. Feature Fusion Transformer (FFT) was designed based on patch-based 1D self-attention mechanism and downsampling to process the Raman spectra of exosomes from cell-derived exosomes and 165 clinical samples, which achieved an accuracy of 95.8% and 94.1% respectively. The second component is the interactive chat agent based on LLM. Retrieval-Augmented Generation (RAG) method was utilized to enhance the knowledge related to exosomes. Overall, LLM serves as the core of this interactive system, which is capable of identifying users’ intentions and invoking the appropriate plugins to process the Raman data of exosomes. This is the first AI agent focusing on exosomes spectroscopy and diagnosis, which bridges the gap between the spectroscopic analysis and clinical diagnosis and shows great potential in intelligent diagnosis.



## Files

This repo should contain the following files:

- Checkpoints - Training Weights for the cell-derived exosome dataset.
- Data - demo data for ChatExosome.
- Model - Feature Fusion Transformer.
- Utils - Toolkits for data preparation.
- Main.py - main method for data prediction and LLM connection.