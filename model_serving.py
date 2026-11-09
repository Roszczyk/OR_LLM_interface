import openvino_genai
from pathlib import Path

from save_model import download_model

class ModelStruct:
    def __init__(self, hf_name : str, local_path : Path, weights_format : str):
        self.hf_name = hf_name
        self.local_path = local_path
        self.weights_format = weights_format

    def enable_model(self):
        """
        returns True if successful
        """
        if not self.local_path.exists():
            try:
                download_model(self.hf_name, self.weights_format, self.local_path)
            except Exception as e:
                return False
        return True


class Inference:
    def __init__(self, system_prompt : str):
        self.sys_prompt = system_prompt

    def run_inference(self, prompt : str):
        assert False, f"this function should be overwritten by the child class, prompt: '{prompt}'"


class OV_Local_Inference(Inference):
    def __init__(self, system_prompt : str, model_path : Path, weights_format : str, device : str):
        super().__init__(system_prompt)
        self.path = model_path
        self.weigths = weights_format
        self.device = device
        self.pipe = openvino_genai.LLMPipeline(str(self.path), self.device)

    def run_inference(self, prompt : str, max_new_tokens = 2000):
        # messages = [
        #     {"role": "system", "content": self.sys_prompt},
        #     {"role": "user", "content": prompt},
        # ]

        messages = (
            f"<|system|>\n{self.sys_prompt}</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>\n"
        )
        return self.pipe.generate(messages, max_new_tokens=max_new_tokens)


class OpenAI_Inference(Inference):
    def __init__(self, system_prompt : str, model : str, api_key : str):
        from openai import OpenAI
        super().__init__(system_prompt)
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def run_inference(self, prompt : str, max_new_tokens: int = 2000):
        response = self.client.responses.create(
            model=self.model,
            instructions=self.sys_prompt,
            input=prompt,
            max_output_tokens=max_new_tokens,
        )
        return response.output_text



# MANUAL TESTING:
if __name__ == "__main__":
    CONTEXT = Path(__file__).parent
    tiny_llama_1_1b_model_path = CONTEXT / "models/TinyLlama_1_1b_v1_ov"
    tiny_llama_inference = OV_Local_Inference("What is mathematics?", tiny_llama_1_1b_model_path,
                                              "i8", "GPU")
    qwen2_7B_model_path = CONTEXT / "models/Qwen2-7B"
    qwen2_7B_inference = OV_Local_Inference("Talk like a pirate", qwen2_7B_model_path,
                                              "i8", "GPU")
    print(qwen2_7B_inference.run_inference("What is mathematics?"))