from tasks_library.task_util import Task, Solution

from pathlib import Path
import json

NLP4LP_path = Path(__file__).parent / "NLP4LP"

def get_easy_NLP4LP():
    assert NLP4LP_path.exists(), "Submodule NLP4LP not initiated. Run `git submodule update --init --recursive`"
    data_dir = NLP4LP_path / "data"

    def read_problem_description(problem_dir : Path):
        with open(problem_dir / "description.txt", "r") as f:
            description = f.read()
        return description

    def read_problem_solution(problem_dir : Path):
        with open(problem_dir / "solution.json", "r") as f:
            solution_read = json.load(f)
        return Solution(solution_read["variables"], solution_read["objective"])

    NLP4LP_tasks_list = {}

    for x in [x for x in data_dir.iterdir() if x.is_dir()]:
        try:
            NLP4LP_tasks_list[x.stem] = Task(read_problem_description(x), read_problem_solution(x))
        except Exception as e:
            continue

    return NLP4LP_tasks_list

