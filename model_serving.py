import openvino_genai
from pathlib import Path

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
    tiny_llama_inference = OV_Local_Inference("You are a helpful assistant", tiny_llama_1_1b_model_path,
                                              "i8", "GPU")
    print(tiny_llama_inference.run_inference("What is OpenVINO?"))