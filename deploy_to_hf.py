import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub library is not installed.")
    print("Please install it by running: pip install huggingface_hub")
    sys.exit(1)

# Configuration
REPO_ID = "o1-spec/pneumodetect-ai"
REPO_TYPE = "space"

# List of files to upload and their target locations in the HF Space
FILES_TO_UPLOAD = {
    # Fine-tuned Models
    "models/finetuned/resnet50/resnet50_rsna_best.keras": "models/finetuned/resnet50/resnet50_rsna_best.keras",
    "models/finetuned/densenet121/densenet121_rsna_best.keras": "models/finetuned/densenet121/densenet121_rsna_best.keras",
    
    # App code & config
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
}

def main():
    print("=" * 80)
    print(f"Hugging Face Space Deployment Script: {REPO_ID}")
    print("=" * 80)

    # Initialize HF API
    # It will automatically look for token saved by running `huggingface-cli login`
    api = HfApi()

    # Verify if user is logged in
    try:
        user_info = api.whoami()
        print(f"✓ Authenticated as Hugging Face user: {user_info['name']}")
    except Exception:
        print("⚠ Not logged in or invalid token.")
        print("Please log in first using Hugging Face CLI:")
        print("  huggingface-cli login")
        print("\nAlternatively, you can get a token with write access from:")
        print("  https://huggingface.co/settings/tokens")
        token = input("\nEnter your Hugging Face write token: ").strip()
        if not token:
            print("Error: Token is required to upload files.")
            sys.exit(1)
        os.environ["HF_TOKEN"] = token

    print("\nStarting upload sequence...")
    print("-" * 50)

    for local_path, repo_path in FILES_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            print(f"⚠ Skipping {local_path}: File not found.")
            continue
            
        print(f"Uploading {local_path} to HF Space path: {repo_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE
            )
            print(f"✓ Successfully uploaded {local_path}")
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")
            print("Please ensure your token has write access to the Space.")
            sys.exit(1)

    print("-" * 50)
    print("✓ Deployment complete! Hugging Face Space is building.")
    print(f"Check build status at: https://huggingface.co/spaces/{REPO_ID}")
    print("=" * 80)

if __name__ == "__main__":
    main()
