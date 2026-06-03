import openvino_genai as ov_genai
from pathlib import Path

class ModelStruct:
    def __init__(self, hf_name : str, local_path : Path, weights_format : str):
        self.hf_name = hf_name
        self.local_path = local_path
        self.weights_format = weights_format


def run_model(model_path, prompt, device = "GPU", max_new_tokens=1000):
    pipe = ov_genai.LLMPipeline(model_path, device)
    return pipe.generate(prompt, max_new_tokens=max_new_tokens)


# MANUAL TESTING:
if __name__ == "__main__":
    CONTEXT = Path(__file__).parent
    print(run_model(CONTEXT / "models/TinyLlama_1_1b_v1_ov", "What is OpenVINO?"))