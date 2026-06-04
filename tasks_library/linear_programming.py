from tasks_library.task_util import Task

linear_programming_minimize_retail_store_simple = Task(
    task = """
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
)

LP_TASKS_COLLECTION = dict({
    "minimize_retail_store_simple" : linear_programming_minimize_retail_store_simple
})