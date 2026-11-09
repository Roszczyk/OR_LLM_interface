class Solution:
    def __init__(self, variables : dict, objective : float):
        self.variables = variables
        self.objective = objective

class Task:
    def __init__(self, task : str, solution : Solution):
        self.task = task
        self.solution = solution

    def llm_ampl_model(self):
        # preparing AMPL model using LLM for comparison
        pass

    def llm_solve(self):
        # solving the model to get the value at optimum
        pass