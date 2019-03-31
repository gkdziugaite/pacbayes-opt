import os
import sys
sys.path.insert(0, "/snn/")
sys.path.insert(0, "/")

from snn.core import package_path
from snn.core.parse_args import BasicParser, Interpreter
from snn.core.utils import serialize


def run_sgd(model, epochs):
    """
        Runs SGD for a predefined number of epochs and saves the resulting model.
    """
    print("Training full network")
    weights_rand_init = model.optimize(epochs=epochs)
    print("Model optimized!!!")

    return [model.get_model_weights(), weights_rand_init]


if __name__ == '__main__':
    basic_args = BasicParser().parse()
    model, test_set, save_path = Interpreter(basic_args).interpret()
    saved_entities = run_sgd(model, basic_args["sgd_epochs"])
    serialization_path = os.path.join(package_path, "experiments", save_path)
    print("Saving run in ", serialization_path)
    serialize(saved_entities, serialization_path, basic_args["overwrite"])
    print("SGD run complete!")
