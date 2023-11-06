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
* [Create a Desktop Shortcut to Launch VoucherVision GUI](#create-a-desktop-shortcut-to-launch-vouchervision-gui)
* [Run VoucherVision](#run-vouchervision)
    * [Setting up API key](#setting-up-api-key)
    * [Check GPU](#check-gpu)
    * [Run Tests](#run-tests)
    * [Starting VoucherVision](#starting-vouchervision)
    * [Azure Instances of OpenAI](#azure-instances-of-openai)
* [Custom Prompt Builder](#custom-prompt-builder)
    * [Load, Build, Edit](#load-build-edit)
    * [Instructions](#instructions)
    * [Defining Column Names Field-Specific Instructions](#defining-column-names-field-specific-instructions)
    * [Prompting Structure](#prompting-structure)
    * [Mapping Columns for VoucherVisionEditor](#mapping-columns-for-vouchervisioneditor)
* [Expense Reporting](#expense-reporting)
    * [Expense Report Dashboard](#expense-report-dashboard)
* [User Interface Images](#user-interface-images)

---

# About
## **VoucherVision** - In Beta Testing Phase 🚀

For inquiries, feedback (or if you want to get involved!) [please complete our form](https://docs.google.com/forms/d/e/1FAIpQLSe2E9zU1bPJ1BW4PMakEQFsRmLbQ0WTBI2UXHIMEFm4WbnAVw/viewform?usp=sf_link).

### **Overview:**  
Initiated by the **University of Michigan Herbarium**, VoucherVision harnesses the power of large language models (LLMs) to transform the transcription process of natural history specimen labels. Our workflow is as follows:
- Text extraction from specimen labels with **LeafMachine2**.
- Text interpretation using **Google Vision OCR**.
- LLMs, including ***GPT-3.5***, ***GPT-4***, ***PaLM 2***, and Azure instances of OpenAI models, standardize the OCR output into a consistent spreadsheet format. This data can then be integrated into various databases like Specify, Symbiota, and BRAHMS.
  
For ensuring accuracy and consistency, the [VoucherVisionEditor](https://github.com/Gene-Weaver/VoucherVisionEditor) serves as a quality control tool.

### **Package Information:**  
The main VoucherVision tool and the VoucherVisionEditor are packaged separately. This separation ensures that lower-performance computers can still install and utilize the editor. While VoucherVision is optimized to function smoothly on virtually any modern system, maximizing its capabilities (like using LeafMachine2 label collages or running Retrieval Augmented Generation (RAG) prompts) mandates a GPU.

> ***NOTE:*** You can absolutely run VoucherVision on non-GPU systems, but RAG will not be possible (luckily the apparent best prompt--Version2--does not use RAG). Additionally, opting to include LeafMachine2 collages without a GPU will significantly extend processing times.

---

# Try our public demo!
Our public demo, while lacking several quality control and reliability features found in the full VoucherVision module, provides an exciting glimpse into its capabilities. Feel free to upload your herbarium specimen and see what happens! We make frequent updates, so don't forget to revisit!
[VoucherVision Demo](https://vouchervision.azurewebsites.net/)
> The web version is quite far behind this GitHub repo...

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
    Or depending on your Python version...
    <pre><code class="language-python">python -m venv venv_VV</code></pre>
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
    <pre><code class="language-python">pip install wheel streamlit streamlit-extras plotly pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity ; if ($?) { pip install numpy -U } ; if ($?) { pip install -U scikit-learn } ; if ($?) { pip install --upgrade streamlit }</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

    - Option B:
    <pre><code class="language-python">pip install wheel streamlit streamlit-extras plotly pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

2. Upgrade important packages. Run this if there is an update to VoucherVision.
    <pre><code class="language-python">pip install --upgrade numpy scikit-learnstreamlit google-generativeai        google-cloud-storage google-cloud-vision azureml-sdk azure-identity openai langchain</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

3. Install PyTorch
    - The LeafMachine2 machine learning algorithm requires PyTorch. If your computer does not have a GPU, then please install a version of PyTorch that is for CPU only. If your computer does have an Nvidia GPU, then please determine which version of PyTorch matches your current CUDA version. Please see [Troubleshooting CUDA](#troubleshooting-cuda) for help. PyTorch is large and will take a bit to install.

    - WITH GPU (or visit [PyTorch.org](https://pytorch.org/get-started/locally/) to find the appropriate version of PyTorch for your CUDA version)
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    - WITHOUT GPU, CPU ONLY
    <pre><code class="language-python">pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    

> If you need help, please submit an inquiry in the form at [LeafMachine.org](https://LeafMachine.org/)

---

## Troubleshooting CUDA

- If your system already has another version of CUDA (e.g., CUDA 11.7) then it can be complicated to switch to CUDA 11.3. 
- The simplest solution is to install pytorch with CPU only, avoiding the CUDA problem entirely.
- Alternatively, you can install the [latest pytorch release](https://pytorch.org/get-started/locally/) for your specific system, either using the cpu only version `pip3 install torch`, `pip3 install torchvision`, `pip3 install torchaudio` or by matching the pythorch version to your CUDA version.
- We have not validated CUDA 11.6 or CUDA 11.7, but our code is likely to work with them too. If you have success with other versions of CUDA/pytorch, let us know and we will update our instructions. 

---

# Create a Desktop Shortcut to Launch VoucherVision GUI
We can create a desktop shortcut to launch VoucherVision. In the `../VoucherVision/` directory is a file called `create_desktop_shortcut.py`. In the terminal, move into the `../VoucherVision/` directory and type:
<pre><code class="language-python">python create_desktop_shortcut.py</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
Or...
<pre><code class="language-python">python3 create_desktop_shortcut.py</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
Follow the instructions, select where you want the shortcut to be created, then where the virtual environment is located. 

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

# Custom Prompt Builder
VoucherVision empowers individual institutions to customize the format of the LLM output. Using our pre-defined prompts you can transcribe the label text into 20 columns, but using our Prompt Builder you can load one of our default prompts and adjust the output to meet your needs. More instructions will come soon, but for now here are a few more details. 

### Load, Build, Edit

The Prompt Builder creates a prompt in the structure that VoucherVision expects. This information is stored as a configuration yaml file in `../VoucherVision/custom_prompts/`. We provide a few versions to get started. You can load one of our examples and then use the Prompt Builder to edit or add new columns. 

![prompt_1](https://LeafMachine.org/img/prompt_1.PNG)

### Instructions

Right now, the prompting instructions are not configurable, but that may change in the future. 

![prompt_2](https://LeafMachine.org/img/prompt_1.PNG)

### Defining Column Names Field-Specific Instructions

The central JSON object shows the structure of the columns that you are requesting the LLM to create and populate with information from the specimen's labels. These will become the rows in the final xlsx file the VoucherVision generates. You can pick formatting instructions, set default values, and give detailed instructions.

> Note: formatting instructions are not always followed precisely by the LLM. For example, GPT-4 is capable of granular instructions like converting ALL CAPS TEXT to sentence-case, but GPT-3.5 and PaLM 2 might not be capable of following that instruction every time (which is why we have the VoucherVisionEditor and are working to link these instructions so that humans editing the output can quickly/easily fix these errors).

![prompt_3](https://LeafMachine.org/img/prompt_3.PNG)

### Prompting Structure

The rightmost JSON object is the entire prompt structure. If you load the `required_structure.yaml` prompt, you will wee the bare-bones version of what VoucherVision expects to see. All of the parts are there for a reason. The Prompt Builder UI may be a little unruly right now thanks to quirks with Streamlit, but we still recommend using the UI to build your own prompts to make sure that all of the required components are present. 

![prompt_4](https://LeafMachine.org/img/prompt_4.PNG)

### Mapping Columns for VoucherVisionEditor

Finally, we need to map columns to a VoucherVisionEditor category.

![prompt_5](https://LeafMachine.org/img/prompt_5.PNG)

# Expense Reporting
VoucherVision logs the number of input and output tokens (using [tiktoken](https://github.com/openai/tiktoken)) from every call. We store the publicly listed prices of the LLM APIs in `../VoucherVision/api_cost/api_cost.yaml`. Then we do some simple math to estimage the cost of run, which is stored inside of your project's output directory `../run_name/Cost/run_name.csv` and all runs are accumulated in a csv file stored in `../VoucherVision/expense_report/expense_report.csv`. VoucherVision only manages `expense_report.csv`, so if you want to split costs by month/quarter then copy and rename `expense_report.csv`. Deleting `expense_report.csv` will let you accumulate more stats.  

> This should be treated as an estimate. The true cost may be slightly different.  

This is an example of the stats that we track:
| run                        | date                     | api_version | total_cost | n_images | tokens_in | tokens_out | rate_in | rate_out | cost_in   | cost_out |
|----------------------------|--------------------------|-------------|------------|----------|-----------|------------|---------|----------|-----------|----------|
| GPT4_test_run1  | 2023_11_05__17-44-31     | GPT_4       | 0.23931    | 2        | 6749      | 614        | 0.03    | 0.06     | 0.20247   | 0.03684  |
| GPT_3_5_test_run  | 2023_11_05__17-48-48     | GPT_3_5     | 0.0189755  | 4        | 12033     | 463        | 0.0015  | 0.002    | 0.0180495 | 0.000926 |
| PALM2_test_run  | 2023_11_05__17-50-35     | PALM2       | 0          | 4        | 13514     | 771        | 0       | 0        | 0         | 0        |
| GPT4_test_run2  | 2023_11_05__18-49-24     | GPT_4       | 0.40962    | 4        | 12032     | 811        | 0.03    | 0.06     | 0.36096   | 0.04866  |

## Expense Report Dashboard
The sidebar in VoucherVision displays summary stats taken from `expense_report.csv`.
![Expense Report Dashboard](https://LeafMachine.org/img/expense_report.PNG)

# User Interface Images
Validation test when the OpenAI key is not provided, but keys for PaLM 2 and Azure OpenAI are present:
![Validation 1](https://LeafMachine.org/img/validation_1.PNG)

---

Validation test when all versions of the OpenAI keys are provided:
![Validation GPT](https://LeafMachine.org/img/validation_gpt.PNG)

---

A successful GPU test:
![Validation GPU](https://LeafMachine.org/img/validation_gpu.PNG)

---

Successful PaLM 2 test:
![Validation PaLM](https://LeafMachine.org/img/validation_palm.PNG)




