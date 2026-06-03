import subprocess
from pathlib import Path

COMMAND = "optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --trust-remote-code TinyLlama_1_1b_v1_ov"
CONTEXT = Path(__file__).parent.resolve()

result = subprocess.run(COMMAND.split(), cwd=CONTEXT, capture_output=True, text=True)

print(result)