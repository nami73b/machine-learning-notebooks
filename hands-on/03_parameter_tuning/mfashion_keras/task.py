import argparse
import json
import os

try:
    import model
except ImportError:
    from . import model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="GCS location to write checkpoints and export models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--filter_size_1", type=int, default=32)
    parser.add_argument("--filter_size_2", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--dropout_rate_2", type=float, default=0.4)
    parser.add_argument("--batch_norm", dest="batch_norm", action="store_true")
    parser.set_defaults(batch_norm = False)

    model_names = [name.replace("_model","") for name in dir(model) if name.endswith("_model")]
    parser.add_argument("--model", required=True, help="Type of model. Supported types are {}".format(model_names))

    args = parser.parse_args()
    hparams = args.__dict__

    output_dir = hparams.pop("output_dir")
    output_dir = os.path.join(
        output_dir,
        os.environ.get("CLOUD_ML_TRIAL_ID", "")
    )
    model.train_and_evaluate(output_dir, hparams)
