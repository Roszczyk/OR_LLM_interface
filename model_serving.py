import openvino_genai as ov_genai
from pathlib import Path

CONTEXT = Path(__file__).parent

pipe = ov_genai.LLMPipeline(CONTEXT / "models/TinyLlama_1_1b_v1_ov", "CPU")
print(pipe.generate("What is OpenVINO?", max_new_tokens=100))