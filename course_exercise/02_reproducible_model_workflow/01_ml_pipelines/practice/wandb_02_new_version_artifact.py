import wandb

if __name__ == "__main__":

    with open("my_artifact.txt", "w+") as fp:
        fp.write("This is an example of an artifact changed")

    run = wandb.init(project="demo_artifact_1",
                     group="experiment_1")

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