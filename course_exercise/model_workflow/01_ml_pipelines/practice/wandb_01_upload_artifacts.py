import wandb


if __name__ == "__main__":

    # create an artifact
    with open("my_artifact.txt", "w+") as fp:
        fp.write("This is an example of an artifact")

    run = wandb.init(project='demo_artifact_1',
                     group='experiment_1')


    artifact = wandb.Artifact(
        name="my_artifact.txt",
        type="data",
        description="This is an example of an artifact",
        metadata={
            "key_1": "value_1"
        }
    )

    artifact.add_file("my_artifact.txt")

    run.log_artifact(artifact)
    run.finish()























## notes
# python file modes
# https://tutorial.eyehunts.com/python/python-file-modes-open-write-append-r-r-w-w-x-etc/