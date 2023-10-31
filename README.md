# VoucherVision

[![VoucherVision](https://LeafMachine.org/img/VV_Logo.png "VoucherVision")](https://LeafMachine.org/)

Table of Contents
=================

* [Table of Contents](#table-of-contents)
<!-- * [VoucherVision](#VoucherVision-user-interface-for-vouchervision)
* [Try our public demo!](#try-our-public-demo)
* [Installing VoucherVision](#installing-VoucherVision)
   * [Prerequisites](#prerequisites)
   * [Installation - Cloning the VoucherVision Repository](#installation---cloning-the-VoucherVision-repository)
   * [About Python Virtual Environments](#about-python-virtual-environments)
   * [Installation - Ubuntu 20.04](#installation---ubuntu-2004)
      * [Virtual Environment](#virtual-environment)
      * [Installing Packages](#installing-packages)
   * [Installation - Windows 10+](#installation---windows-10)
      * [Virtual Environment](#virtual-environment-1)
      * [Installing Packages](#installing-packages-1) -->

---

# About
Note: VoucherVision is currently in beta-testing. For inquiries, [please fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSe2E9zU1bPJ1BW4PMakEQFsRmLbQ0WTBI2UXHIMEFm4WbnAVw/viewform?usp=sf_link). 

VoucherVision is a project led by the University of Michigan Herbarium that leverages large language models (LLMs) to automate transcription of natural history specimen labels. We use LeafMachine2 to extract text from specimen labels, Google Vision OCR to extract text, and a growing list of LLMs (GPT-3.5, GPT-4, PaLM 2, Azure instances of OpenAI models) to parse the OCR content into a uniform spreadsheet that can then be uploaded to a database of your choice (Specify, Symbiota, BRAHMS). For quality control, we provide the [VoucherVisionEditor](https://github.com/Gene-Weaver/VoucherVisionEditor) tool. 

VoucherVision and VoucherVisionEditor are currently separate packages so that the editor can be easily installed on lower performance computers. VoucherVisionEditor should run on just about any modern computer, whereas to fully utilize VoucherVision requires a GPU to use LeafMachine2 label collages and to run Retrieval Augmented Generation (RAG) prompts (the version 1-style prompts). ***NOTE:*** VoucherVision can be run on a computer that does not have a GPU, you just can't do RAG and if you use LeafMachine2 without a GPU it just takes a much longer time to process. 

# Installing VoucherVisionEditor

## Prerequisites
- Python 3.10 or later 
- Optional: an Nvidia GPU + CUDA for running LeafMachine2

## Installation - Cloning the VoucherVisionEditor Repository
1. First, install Python 3.10, or greater, on your machine of choice. We have validated up to Python 3.11.
    - Make sure that you can use `pip` to install packages on your machine, or at least inside of a virtual environment.
    - Simply type `pip` into your terminal or PowerShell. If you see a list of options, you are all set. Otherwise, see
    either this [PIP Documentation](https://pip.pypa.io/en/stable/installation/) or [this help page](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)
2. Open a terminal window and `cd` into the directory where you want to install VoucherVisionEditor.
3. In the [Git BASH terminal](https://gitforwindows.org/), clone the VoucherVisionEditor repository from GitHub by running the command:
    <pre><code class="language-python">git clone https://github.com/Gene-Weaver/VoucherVisionEditor.git</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Move into the VoucherVisionEditor directory by running `cd VoucherVisionEditor` in the terminal.
5. To run VoucherVisionEditor we need to install its dependencies inside of a python virtual environmnet. Follow the instructions below for your operating system. 

## About Python Virtual Environments
A virtual environment is a tool to keep the dependencies required by different projects in separate places, by creating isolated python virtual environments for them. This avoids any conflicts between the packages that you have installed for different projects. It makes it easier to maintain different versions of packages for different projects.

For more information about virtual environments, please see [Creation of virtual environments](https://docs.python.org/3/library/venv.html)

---

## Installation - Windows 10+

### Virtual Environment

1. Still inside the VoucherVisionEditor directory, show that a venv is currently not active 
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
2. Then create the virtual environment (venv_VV is the name of our new virtual environment)  
    <pre><code class="language-python">python3 -m venv venv_VV</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Activate the virtual environment  
    <pre><code class="language-python">.\venv_VV\Scripts\activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Confirm that the venv is active (should be different from step 1)  
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
5. If you want to exit the venv later for some reason, deactivate the venv using  
    <pre><code class="language-python">deactivate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

### Installing Packages

1. Install the required dependencies to use VoucherVisionEditor  
    - Option A - If you are using Windows PowerShell:
    <pre><code class="language-python">pip install streamlit streamlit-extras pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity ; if ($?) { pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 } ; if ($?) { pip install numpy -U } ; if ($?) { pip install -U scikit-learn } ; if ($?) { pip install --upgrade streamlit }</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

    - Option B - Install individually:
    <pre><code class="language-python">pip install numpy -U</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    <pre><code class="language-python">pip install -U scikit-learn</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    <pre><code class="language-python">pip install --upgrade streamlit</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    Now the rest of the packages
    <pre><code class="language-python">pip install streamlit streamlit-extras pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>


2. Install PyTorch
    - The LeafMachine2 machine learning algorithm requires PyTorch. If your computer does not have a GPU, then please install a version of PyTorch that is for CPU only. If your computer does have an Nvidia GPU, then please determine which version of PyTorch matches your current CUDA version. Please see [Troubleshooting CUDA](#troubleshooting-cuda) for help. PyTorch is large and will take a bit to install.

    - WITH GPU 
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

> If you need help, please submit an inquiry in the form at [LeafMachine.org](https://LeafMachine.org/)

---

## Troubleshooting CUDA

- If your system already has another version of CUDA (e.g., CUDA 11.7) then it can be complicated to switch to CUDA 11.3. 
- The simplest solution is to install pytorch with CPU only, avoiding the CUDA problem entirely, but that is not recommended given that 
LeafMachine2 is designed to use GPUs. We have not tested LeafMachine2 on systems that lack GPUs.
- Alternatively, you can install the [latest pytorch release](https://pytorch.org/get-started/locally/) for your specific system, either using the cpu only version `pip3 install torch`, `pip3 install torchvision`, `pip3 install torchaudio` or by matching the pythorch version to your CUDA version.
- We have not validated CUDA 11.6 or CUDA 11.7, but our code is likely to work with them too. If you have success with other versions of CUDA/pytorch, let us know and we will update our instructions. 

---


# Setting up API key
VoucherVision requires access to Google Vision OCR and at least one of the following LLMs: OpenAI API, Google PaLM 2, a private instance of OpenAI through Microsoft Azure. On first startup, you will see a page with instructions on how to get these API keys. ***Nothing will work until*** you get at least the Google Vision OCR API key and one LLM API key. 




