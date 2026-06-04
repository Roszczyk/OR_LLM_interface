class Task:
    def __init__(self, task : str, value_at_optimum : float = None, is_complete = True):
        self.task = task
        self.value_at_optimum = value_at_optimum
        self.is_complete = is_complete

    def llm_ampl_model(self):
        # preparing AMPL model using LLM for comparison
        pass

    def llm_solve(self):
        # solving the model to get the value at optimum
        pass