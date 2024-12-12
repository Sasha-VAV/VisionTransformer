from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Sashavav/VisionTransformer", local_dir="dvc-remote", repo_type="dataset"
)
