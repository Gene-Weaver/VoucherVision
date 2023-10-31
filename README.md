# VoucherVision

[![VoucherVision](https://LeafMachine.org/img/VV_Logo.png "VoucherVision")](https://LeafMachine.org/)

Table of Contents
=================

* [Table of Contents](#table-of-contents)
* [About](#about)
* [Try our public demo!](#try-our-public-demo)
* [Installing VoucherVision](#installing-VoucherVision)
   * [Prerequisites](#prerequisites)
   * [Installation - Cloning the VoucherVision Repository](#installation---cloning-the-VoucherVision-repository)
   * [About Python Virtual Environments](#about-python-virtual-environments)
   * [Installation - Windows 10+](#installation---windows-10)
      * [Virtual Environment](#virtual-environment-1)
      * [Installing Packages](#installing-packages-1)
   * [Troubleshooting CUDA](#troubleshooting-cuda)
* [Run VoucherVision](#run-vouchervision)
    * [Setting up API key](#setting-up-api-key)
    * [Check GPU](#check-gpu)
    * [Run Tests](#run-tests)
    * [Starting VoucherVision](#starting-vouchervision)
    * [Azure Instances of OpenAI](#azure-instances-of-openai)

---

# About
## **VoucherVision** - In Beta Testing Phase 🚀

For inquiries or feedback, [please complete our form](https://docs.google.com/forms/d/e/1FAIpQLSe2E9zU1bPJ1BW4PMakEQFsRmLbQ0WTBI2UXHIMEFm4WbnAVw/viewform?usp=sf_link).

### **Overview:**  
Initiated by the **University of Michigan Herbarium**, VoucherVision harnesses the power of large language models (LLMs) to transform the transcription process of natural history specimen labels. Our workflow is as follows:
- Text extraction from specimen labels with **LeafMachine2**.
- Text interpretation using **Google Vision OCR**.
- LLMs, including ***GPT-3.5***, ***GPT-4***, ***PaLM 2***, and Azure instances of OpenAI models, standardize the OCR output into a consistent spreadsheet format. This data can then be integrated into various databases like Specify, Symbiota, and BRAHMS.
  
For ensuring accuracy and consistency, the [VoucherVisionEditor](https://github.com/Gene-Weaver/VoucherVision) serves as a quality control tool.

### **Package Information:**  
The main VoucherVision tool and the VoucherVisionEditor are packaged separately. This separation ensures that lower-performance computers can still install and utilize the editor. While VoucherVision is optimized to function smoothly on virtually any modern system, maximizing its capabilities (like using LeafMachine2 label collages or running Retrieval Augmented Generation (RAG) prompts) mandates a GPU.

> ***NOTE:*** You can absolutely run VoucherVision on non-GPU systems, but RAG will not be possible (luckily the apparent best prompt--Version2--does not use RAG). Additionally, opting to include LeafMachine2 collages without a GPU will significantly extend processing times.

---

# Try our public demo!
Our public demo, while lacking several quality control and reliability features found in the full VoucherVision module, provides an exciting glimpse into its capabilities. Feel free to upload your herbarium specimen and see what happens! We make frequent updates, so don't forget to revisit!
[VoucherVision Demo](https://vouchervision.azurewebsites.net/)

---


# Installing VoucherVision

## Prerequisites
- Python 3.10 or later 
- Optional: an Nvidia GPU + CUDA for running LeafMachine2

## Installation - Cloning the VoucherVision Repository
1. First, install Python 3.10, or greater, on your machine of choice. We have validated up to Python 3.11.
    - Make sure that you can use `pip` to install packages on your machine, or at least inside of a virtual environment.
    - Simply type `pip` into your terminal or PowerShell. If you see a list of options, you are all set. Otherwise, see
    either this [PIP Documentation](https://pip.pypa.io/en/stable/installation/) or [this help page](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)
2. Open a terminal window and `cd` into the directory where you want to install VoucherVision.
3. In the [Git BASH terminal](https://gitforwindows.org/), clone the VoucherVision repository from GitHub by running the command:
    <pre><code class="language-python">git clone https://github.com/Gene-Weaver/VoucherVision.git</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Move into the VoucherVision directory by running `cd VoucherVision` in the terminal.
5. To run VoucherVision we need to install its dependencies inside of a python virtual environmnet. Follow the instructions below for your operating system. 

## About Python Virtual Environments
A virtual environment is a tool to keep the dependencies required by different projects in separate places, by creating isolated python virtual environments for them. This avoids any conflicts between the packages that you have installed for different projects. It makes it easier to maintain different versions of packages for different projects.

For more information about virtual environments, please see [Creation of virtual environments](https://docs.python.org/3/library/venv.html)

---

## Installation - Windows 10+
Installation should basically be the same for Linux.
### Virtual Environment

1. Still inside the VoucherVision directory, show that a venv is currently not active 
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

1. Install the required dependencies to use VoucherVision  
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

# Run VoucherVision
1. In the terminal, make sure that you `cd` into the `VoucherVision` directory and that your virtual environment is active (you should see venv_VV on the command line). 
2. Type:
    <pre><code class="language-python">python run_VoucherVision.py</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    or depending on your Python installation:
    <pre><code class="language-python">python3 run_VoucherVision.py</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. If you ever see an error that says that a "port is not available", open `run_VoucherVision.py` in a plain text editor and change the `--port` value to something different but close, like 8502.

## Setting up API key
VoucherVision requires access to Google Vision OCR and at least one of the following LLMs: OpenAI API, Google PaLM 2, a private instance of OpenAI through Microsoft Azure. On first startup, you will see a page with instructions on how to get these API keys. ***Nothing will work until*** you get at least the Google Vision OCR API key and one LLM API key. 

## Check GPU
Press the "Check GPU" button to see if you have a GPU available. If you know that your computer has an Nvidia GPU, but the check fails, then you need to install an different version of PyTorch in the virtual environment. 

## Run Tests
Once you have provided API keys, you can test all available prompts and LLMs by pressing the test buttons. Every combination of LLM, prompt, and LeafMachine2 collage will run on the image in the `../VoucherVision/demo/demo_images` folder. A grid will appear letting you know which combinations are working on your system. 

## Starting VoucherVision
1. "Run name" - Set a run name for your project. This will be the name of the new folder that contains the output files.
2. "Output directory" - Paste the full file path of where you would like to save the folder that will be created in step 1.
3. "Input images directory" - Paste the full file path of where the input images are located. This folder can only have JPG or JPEG images inside of it.
4. "Select an LLM" - Pick the LLM you want to use to parse the unstructured OCR text. 
    - As of Nov. 1, 2023 PaLM 2 is free to use. 
5. "Prompt Version" - Pick your prompt version. We recommend "Version 2" for production use, but you can experiment with our other prompts. 
6. "Cropped Components" - Check the box to use LeafMachine2 collage images as the input file. LeafMachine2 can often find small handwritten text that may be missed by Google Vision OCR's text detection algorithm. But, the difference in performance is not that big. You will still get good performance without using the LeafMachine2 collage images.  
7. "Domain Knowledge" is only used for "Version 1" prompts.
8. "Component Detector" sets basic LeafMachine2 parameters, but the default is likely good enough.
9. "Processing Options"
    - The image file name defines the row name in the final output spreadsheet.
    - We provide some basic options to clean/parse the image file name to produce the desired output. 
    - For example, if the input image name is `MICH-V-3819482.jpg` but the desired name is just `3819482` you can add `MICH-V-` to the "Remove prefix from catalog number" input box. Alternatively, you can check the "Require Catalog..." box and achieve the same result. 

10. ***Finally*** you can press the start processing button.

## Azure Instances of OpenAI
If your institution has an enterprise instance of OpenAI's services, [like at the University of Michigan](https://its.umich.edu/computing/ai), you can use Azure instead of the OpenAI servers. Your institution should be able to provide you with the required keys (there are 5 required keys for this service). 






