from model_serving import ModelStruct
from tasks_library.task_util import Task
from tasks_library.linear_programming import LP_TASKS_COLLECTION
from or_stages import stage_validate_task, stage_define_ampl

from pathlib import Path

CONTEXT_DIR = Path(__file__).parent

models = dict({
    "TinyLlama-1.1B_int4" : ModelStruct("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        CONTEXT_DIR / "models/TinyLlama_1_1b_v1_ov",
                                        "int4"),
    # "Qwen2-7B_fp16" : ModelStruct("Qwen/Qwen2-7B",
    #                                 CONTEXT_DIR / "models/Qwen2-7B", 
    #                                 "fp16")
})

def run(models : dict[str, ModelStruct], tasks : dict[str, Task]):
    # COLLECT ALL MODELS
    print("[ INFO ] Collecting all the models...")
    for m in models.keys():
        if not models[m].enable_model():
            models.pop(m)

    # CHECK ALL TASKS
    for t in tasks.keys():
        print(f"=== TASK {t} ===")
        for m in models.keys():
            task_validation = stage_validate_task(models[m], tasks[t].task)
            if not task_validation["validation"]:
                print(f"==> Model {m} did not validate task {t}")
                print(f"Problems diagnozed with task {t}:\n {task_validation["instruction"]}")
            else:
                print(f"==> Model {m} validated task {t}")
            ampl_model = stage_define_ampl(models[m], tasks[t].task)
            print(f"==> {m} prepared AMPL model for task {t}: ")
            print(ampl_model)


if __name__ == "__main__":
    run(models, LP_TASKS_COLLECTION)