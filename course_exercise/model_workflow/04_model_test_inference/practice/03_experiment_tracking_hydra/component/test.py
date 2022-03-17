import argparse
import logging
import math

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init()

    # print training name
    logger.info(f"Running with a={args.a} and b={args.b}")
    run.config.update(args)


    # Fake training loop
    for i in range(100):
        run.log(
            {
                "c": args.a / math.log(i+2) + args.b,
                "epoch": i*5
            }
        )

    # Create a fake artifact
    with open("my_artifact.txt", "w+") as fp:
        fp.write(f"{args.a} {args.b}")

    artifact = wandb.Artifact(name="my_artifact.txt", type="fake_artifact", description='A fake artifact')
    artifact.add_file("my_artifact.txt")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A noop component",
                                     fromfile_prefix_chars="@")

    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    args = parser.parse_args()

    go(args)