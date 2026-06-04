from model_serving import run_model, ModelStruct
from main import models

def stage_validate_task(model : ModelStruct, task : str, device = "GPU", max_new_tokens = 1000):
    prompt = f"""
You will be given the optimization problem. Your job is to verify if all the crutual data was \
provided, all the variables explained. Validate if the problem is possible to solve. If there are \
any gaps - explained to the user what needs to be added to the model. If the problem does not\
need redefinition - write "%DONE%"
Current task: {task}
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


def stage_define_ampl(model : ModelStruct, task : str, device = "GPU", max_new_tokens = 1000):
    prompt = f"""
Prepare a valid model in AMPL based on the following optimization problem:
{task}
In your reply you should only include AMPL model without any additional comments.
Declare all the constants and all the constraints in AMPL model. 
"""
    ampl_model = run_model(model.local_path, prompt, device, max_new_tokens)
    return ampl_model

if __name__ == "__main__":

    TASK = """
    A company supplies products from four warehouses to a retail store. Let:

    x1 = units shipped from Warehouse A
    x2 = units shipped from Warehouse B
    x3 = units shipped from Warehouse C
    x4 = units shipped from Warehouse D

    The transportation costs per unit are:

    Warehouse A: $5
    Warehouse B: $4
    Warehouse C: $6
    Warehouse D: $3

    The store requires at least 100 units in total.

    Warehouse capacities are:

    Warehouse A: at most 40 units
    Warehouse B: at most 50 units
    Warehouse C: at most 30 units
    Warehouse D: at most 60 units

    Additionally, due to contractual obligations, at least 20 units must be shipped from Warehouse B and Warehouse D combined.

    Formulate a linear programming model that minimizes the total transportation cost.
    """
    model = models["TinyLlama-1.1B_int4"]

    print("=== STAGE 1: VALIDATE TASK ===")
    print(stage_validate_task(model, TASK))
    print("=== STAGE 2: DEFINE AMPL ===")
    print(stage_define_ampl(model, TASK))
