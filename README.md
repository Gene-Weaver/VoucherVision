python -m venv venv_VV
.\venv_VV\Scripts\activate

pip install numpy -U
pip install -U scikit-learn
2. Upgrade Streamlit 
    <pre><code class="language-python">pip install --upgrade streamlit</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

# CUDA 11.7 https://pytorch.org/get-started/previous-versions/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install streamlit pyyaml Pillow pandas matplotlib matplotlib-inline tqdm openai langchain tiktoken openpyxl google-generativeai google-cloud-storage google-cloud-vision opencv-python chromadb chroma-migrate InstructorEmbedding transformers sentence-transformers seaborn dask psutil cpuinfo py-cpuinfo azureml-sdk azure-identity
  