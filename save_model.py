import subprocess
from pathlib import Path

def download_model(model_name: str, weight_format : str, save_dir : Path):
    command = f"optimum-cli export openvino --model {model_name} --weight-format {weight_format} --trust-remote-code {save_dir}"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    if "error" in result.stdout or "error" in result.stderr:
        raise Exception(f"Unable to download model {model_name}\nStderr: {result.stderr}")
    return result


# MANUAL TESTING:
if __name__ == "__main__":
    save_dir = Path(__file__).parent / "models/TinyLlama2"
    download_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "int4", save_dir)
