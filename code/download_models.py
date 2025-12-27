import os
import requests
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Detailed file list preserving structure
EMBEDDING_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json", 
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json", # Critical for Pooling/Dense layers
]
LOCAL_EMBED_DIR = "local_embedding_model"

LLM_URL = "https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf"
LOCAL_LLM_DIR = "local_llm"
LLM_FILENAME = "orca-mini-3b-gguf2-q4_0.gguf"

def download_file(url, output_path):
    print(f"Downloading {url}...")
    try:
        # Create parent dir if needed for nested files
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # verify=False bypasses the SSL error
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 
        
        with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        print(f"Saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    # 1. Download Embedding Model
    print("--- Downloading Embedding Model ---")
    if not os.path.exists(LOCAL_EMBED_DIR):
        os.makedirs(LOCAL_EMBED_DIR)
    
    base_url = f"https://huggingface.co/{EMBEDDING_MODEL_ID}/resolve/main"
    
    for filename in EMBEDDING_FILES:
        # Construct full URL and local path
        file_url = f"{base_url}/{filename}"
        output_path = os.path.join(LOCAL_EMBED_DIR, filename)
        
        if not os.path.exists(output_path):
            download_file(file_url, output_path)
        else:
            print(f"{filename} already exists.")

    # 2. Download LLM
    print("\n--- Downloading LLM ---")
    if not os.path.exists(LOCAL_LLM_DIR):
        os.makedirs(LOCAL_LLM_DIR)
        
    llm_path = os.path.join(LOCAL_LLM_DIR, LLM_FILENAME)
    if not os.path.exists(llm_path):
        download_file(LLM_URL, llm_path)
    else:
        print(f"{LLM_FILENAME} already exists.")

    print("\nDownload setup complete. Now you can run rag_pipeline.py")

if __name__ == "__main__":
    # Disable warnings for unverified HTTPS requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
