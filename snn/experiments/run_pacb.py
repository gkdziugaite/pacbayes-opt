import os
import sys
sys.path.insert(0, "/snn/")
sys.path.insert(0, "/")

from snn.core import package_path
from snn.core.parse_args import CompleteParser, Interpreter
from snn.core.utils import deserialize


def run_pacb(weights_rand_init, model, test_set, epochs, learning_rate, drop_lr, lr_factor, seed, trainw):
    testX, testY = test_set

    save_dict = {"log_post_all": True, "PACB_weights": True, "L2_PACB": False, "diff": False, "iter": 500*50,
                 "w*": model.get_model_weights(), "mean_weights": True, "var_weights": True, "PACBound": True,
                 "B_val": True, "KL_val": True, "test_acc": True, "train_acc": True, "log_prior_std": True}
    # Optimize the pac bayes bound with this newly trained prior
    model.PACB_init
    # Checkpoint optimization runs periodically (absolute),
    model.optimize_PACB(weights_rand_init, epochs, learning_rate=learning_rate, drop_lr=drop_lr, lr_factor=lr_factor,
                        save_dict=save_dict, trainWeights=trainw)
    model.evaluate_SNN_accuracy(testX, testY, weights_rand_init, N_SNN_samples=1, save_dict=save_dict)

    path = os.path.join(package_path, "experiments", "cifar",
                        ("model_mean_opt{}_LR{}_seed{}.pickle".format(trainw, learning_rate, seed)))
    model.save_output(path=path)


if __name__ == '__main__':
    complete_args = CompleteParser().parse()
    _, _, save_path = Interpreter(complete_args).interpret()
    deserialization_path = os.path.join(package_path, "experiments", save_path)
    print("Loading model weights saved in ", deserialization_path)
    model_weights, weights_rand_init = deserialize(deserialization_path)
    print("Model weights loaded!")
    model, test_set, _ = Interpreter(complete_args).interpret(model_weights)
    run_pacb(weights_rand_init, model, test_set, complete_args["pacb_epochs"], complete_args["lr"],
             complete_args["drop_lr"], complete_args["lr_factor"], complete_args["seed"], complete_args["trainw"])
    print("PAC-Bayes run complete!")
