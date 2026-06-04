from model_serving import run_model, ModelStruct
from save_model import download_model

from pathlib import Path

CONTEXT_DIR = Path(__file__).parent

models = dict({
    "TinyLlama-1.1B_int4" : ModelStruct("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        CONTEXT_DIR / "models/TinyLlama_1_1b_v1_ov",
                                        "int4"),
    "shit" : ModelStruct("shit", CONTEXT_DIR / "shit", "shit")
})

if __name__ == "__main__":
    for m in models.keys():
        try:
            if not models[m].local_path.exists():
                res = download_model(models[m].hf_name, models[m].weights_format, models[m].local_path)
            prompt = "What is OpenVINO?"
            model_reply = run_model(models[m].local_path, prompt)
            print(f"==> {m} reply to prompt \"{prompt}\":")
            print(model_reply)
        except Exception as e:
            print(f"[ ERROR ] {m} has not work properly. Exception happened: {e}")