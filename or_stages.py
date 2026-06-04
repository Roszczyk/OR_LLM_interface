from model_serving import run_model, ModelStruct

def stage_validate_task(model : ModelStruct, task : str, device = "GPU", max_new_tokens = 1000):
    prompt = f"""
You are an optimization problem validator.

Your task is to check whether the optimization problem contains enough information to create a complete AMPL model.

Check:
1. Are all decision variables clearly defined?
2. Is the objective function fully specified?
3. Are all constraints fully specified?
4. Are all constants, limits, capacities, costs, demands, or coefficients provided?
5. Is the problem solvable without making assumptions?

OUTPUT RULES:

- If the problem is complete, output EXACTLY:
%DONE%

- If the problem is incomplete, output:
%MISSING%
followed by a short list of missing information.

Do not explain anything else.
Do not solve the problem.
Do not generate AMPL.

### OPTIMIZATION PROBLEM ###
{task}
### END ###
"""
    reply = run_model(model.local_path, prompt, device, max_new_tokens)
    if "%DONE%" in reply:
        return dict({
            "validation" : True
        })
    else:
        return dict({
            "validation" : False, 
            "instruction" : reply
        })


def stage_define_ampl(model : ModelStruct, task : str, device = "GPU", max_new_tokens = 2000):
    prompt = f"""
Task:
Create an AMPL model. In your answer should be ONLY an AMPL model. DO NOT put any additional comments.

Optimization problem:
{task}

Answer:
"""
    ampl_model = run_model(model.local_path, prompt, device, max_new_tokens)
    return ampl_model


# MANUAL TESTING
if __name__ == "__main__":
    from main import models
    from tasks_library.linear_programming import LP_TASKS_COLLECTION

    task = LP_TASKS_COLLECTION["minimize_retail_store_simple"].task
    model = models["TinyLlama-1.1B_int4"]

    print("=== STAGE 1: VALIDATE TASK ===")
    print(stage_validate_task(model, task))
    print("=== STAGE 2: DEFINE AMPL ===")
    print(stage_define_ampl(model, task))
