from typing import Callable, List
import aerosandbox.numpy as np
from aerosandbox.modeling.surrogate_model import SurrogateModel
import casadi as _cas


# def blackbox(
#         function,
#         gradient,
# ):

class BlackBoxModel(SurrogateModel):

    def __init__(self,
                 function: Callable,
                 gradient: Callable = None,
                 input_shape=(1,),
                 output_shape=(1,),

                 ):
        self.function = function
        self.gradient = gradient
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, *args, **kwargs):
        outer_self = self

        ### Create the class
        class BlackBoxInternalModel(_cas.Callback):
            def __init__(self, model_name="BlackBoxInternalModel"):
                super().__init__()

                if outer_self.gradient is None:
                    self.construct(model_name, {"enable_fd": True})
                else:
                    self.construct(model_name)

            def eval(self, eval_args: List[_cas.DM]) -> List[_cas.DM]:
                output = outer_self.function(*eval_args, **kwargs)
                return [output]

        ### Store the class and instantiated model in a semi-private place for external access during debugging.
        self._casadi_model_class = BlackBoxInternalModel
        self._casadi_model = BlackBoxInternalModel()

        return self._casadi_model(*args)
