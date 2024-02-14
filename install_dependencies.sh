#!/bin/bash

# List of packages to be installed
packages=(
    wheel
    gputil
    streamlit
    streamlit-extras
    streamlit-elements==0.1.*
    plotly
    google-api-python-client
    wikipedia
    PyMuPDF
    craft-text-detector
    pyyaml
    Pillow
    bitsandbytes
    accelerate
    mapboxgl
    pandas
    matplotlib
    matplotlib-inline
    tqdm
    openai
    langchain
    langchain-community
    langchain-core
    langchain_mistralai
    langchain_openai
    langchain_google_genai
    langchain_experimental
    jsonformer
    vertexai
    ctransformers
    google-cloud-aiplatform
    tiktoken
    llama-cpp-python
    openpyxl
    google-generativeai
    google-cloud-storage
    google-cloud-vision
    opencv-python
    chromadb
    chroma-migrate
    InstructorEmbedding
    transformers
    sentence-transformers
    seaborn
    dask
    psutil
    py-cpuinfo
    Levenshtein
    fuzzywuzzy
    opencage
    geocoder
    pycountry_convert
)

# Function to install a single package
install_package() {
    package=$1
    echo "Installing $package..."
    pip3 install $package
    if [ $? -ne 0 ]; then
        echo "Failed to install $package"
        exit 1
    fi
}

# Install each package individually
for package in "${packages[@]}"; do
    install_package $package
done

echo "All packages installed successfully."
echo "Cloning and installing LLaVA..."


cd vouchervision
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA # Assuming you want to run pip install in the LLaVA directory
pip install -e .
git pull
pip install -e .
echo "LLaVA ready"
