def categorize_installation_methods(requirements):
    is_conda = []
    is_pip = []

    # This list can be expanded based on the common knowledge of packages available on Conda
    known_conda_packages = {
        "torch", "torchvision", "torchaudio", "streamlit", "plotly", "pyyaml", "Pillow", 
        "pandas", "matplotlib", "tqdm", "opencv-python", "google-api-python-client", 
        "google-cloud-storage", "google-cloud-vision", "transformers", "sentence-transformers",
        "seaborn", "dask", "psutil", "openpyxl", "matplotlib-inline"
    }
    
    # Simplified check: Assumes specific versions are handled similarly across pip and conda
    # For a more sophisticated script, you might need to handle version-specific logic
    for req in requirements:
        # Strip version specifications for simplicity
        package = req.split("==")[0].split(">")[0].split("<")[0]
        if package in known_conda_packages:
            is_conda.append(req)
        else:
            is_pip.append(req)
    
    return is_conda, is_pip

requirements = [
    "wheel", "torch", "torchvision", "torchaudio", "streamlit==1.31.0", "streamlit-extras",
    "crewai", "duckduckgo-search", "plotly", "pyyaml", "Pillow", "pandas", "matplotlib",
    "matplotlib-inline", "tqdm", "openai", "google-api-python-client", "mapboxgl", "langchain",
    "langchain-community", "langchain-core", "langchain_mistralai", "langchain_openai",
    "langchain_google_genai", "langchain_experimental", "langchain-google-vertexai", "jsonformer",
    "PyMuPDF", "craft-text-detector", "ctransformers", "gputil", "vertexai", "google-cloud-aiplatform",
    "bitsandbytes", "llama-cpp-python", "accelerate", "tiktoken", "wikipedia", "wikibase-rest-api-client",
    "Wikipedia-API", "mediawikiapi", "openpyxl", "google-generativeai", "google-cloud-storage",
    "google-cloud-vision", "opencv-python", "chromadb", "chroma-migrate", "InstructorEmbedding",
    "transformers", "sentence-transformers", "seaborn", "dask", "psutil", "py-cpuinfo", "Levenshtein",
    "fuzzywuzzy", "opencage", "geocoder", "pycountry_convert"
]

if __name__ == '__main__':
    is_conda, is_pip = categorize_installation_methods(requirements)

    print("Install via conda:")
    for pkg in is_conda:
        print(pkg)

    print("\nInstall via pip:")
    for pkg in is_pip:
        print(pkg)