"""Implementation of a space that starts with a function name and then a dictionary of arguments.

For example:
action_space = FuncConditional(
    {
        "swap": MultiDiscrete(np.array([[self.num_rows, self.num_cols], [self.num_rows, self.num_cols]]))
        "reorder": Permutation(self.num_rows * self.num_cols)
    }
)

each time it samples a function name, then it samples the arguments from the corresponding space.
"""

from gymnasium.spaces import Space, Discrete
from typing import Dict, Any, List
import numpy as np
import random

class FuncConditional(Space):
    def __init__(self, func_arg_spaces: Dict[str, Space]):
        """
        func_arg_spaces: a dictionary mapping function names to their corresponding argument spaces
        """
        assert len(func_arg_spaces) > 0, "You must provide at least one function and its arg space."

        self.func_arg_spaces = func_arg_spaces
        self.function_names = list(func_arg_spaces.keys())
        self.function_space = Discrete(len(self.function_names))  # sampling index, not string

        super().__init__(None, None)  # shape/dtype are not used in composite spaces
    
    def get_function_names(self):
        return self.function_names

    def sample(self, exclude_actions: List[str] = [], use_string_repr: bool = True) -> Dict[str, Any]:
        valid_actions = [name for name in self.function_names if name not in exclude_actions]
        idx = random.randint(0, len(valid_actions) - 1)
        func_name = valid_actions[idx]

        # Sample its corresponding argument space
        arg_sample = self.func_arg_spaces[func_name].sample()

        if use_string_repr:
            arg_sample = arg_sample.tolist() if isinstance(arg_sample, np.ndarray) else arg_sample
            return f"('{func_name}', {arg_sample})"
        else:
            return (func_name, arg_sample)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if "function" not in x or "args" not in x:
            return False
        if x["function"] not in self.func_arg_spaces:
            return False
        return self.func_arg_spaces[x["function"]].contains(x["args"])

    def __repr__(self):
        return f"FuncConditional({list(self.func_arg_spaces.keys())})"

    def __eq__(self, other):
        return isinstance(other, FuncConditional) and \
               self.func_arg_spaces == other.func_arg_spaces
    
    def __getitem__(self, key: str) -> Space:
        return self.func_arg_spaces[key]
    
if __name__ == "__main__":
    from gymnasium.spaces import MultiDiscrete, Permutation, Text

    # Example usage for a 3x3 jigsaw puzzle
    num_rows = 3
    num_cols = 3
    func_conditional = FuncConditional({
        "swap": MultiDiscrete(np.array([[num_rows, num_cols], [num_rows, num_cols]])),
        "reorder": Permutation(num_rows * num_cols),
        "stop": Text(1)
    })

    for _ in range(10):
        print(func_conditional.sample())
