python -m venv venv_VV
.\venv_VV\Scripts\activate

1. Install Packages
    - Option A: Install using Windows PowerShell
        * Install all packages:
        <pre><code class="language-python">pip install streamlit pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity ; if ($?) { pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 } ; if ($?) { pip install numpy -U } ; if ($?) { pip install -U scikit-learn } ; if ($?) { pip install --upgrade streamlit }</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
        * If PyTorch failed, or your GPU/CUDA version needs to be different, replace the following with the equivalent [command from the PyTorch website]( https://pytorch.org/get-started/previous-versions/)
        <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>


<pre><code class="language-python">pip install --upgrade streamlit</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>


pip install streamlit pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil cpuinfo py-cpuinfo azureml-sdk azure-identity && pip install numpy -U && pip install -U scikit-learn && pip install --upgrade streamlit

2. Install PyTorch
# CUDA 11.7 https://pytorch.org/get-started/previous-versions/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


pip install streamlit pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil py-cpuinfo azureml-sdk azure-identity ; if ($?) { pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 } ; if ($?) { pip install numpy -U } ; if ($?) { pip install -U scikit-learn } ; if ($?) { pip install --upgrade streamlit }
