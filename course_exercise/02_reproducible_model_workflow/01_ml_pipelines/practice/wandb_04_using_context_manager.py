import wandb

if __name__ == "__main__":


    with wandb.init(project="multiple_runs") as run:
        artifact = wandb.Artifact(
            name="my_artifact.txt",
            type="data",
            description="This is an example of an artifact",
            metadata={
                "key_1": "value_1"
            }
        )
        artifact.add_file("my_artifact.txt")

    with wandb.init(project="multiple_runs") as run:
        artifact = wandb.Artifact(
            name="my_artifact.txt",
            type="data",
            description="This is an example of an artifact",
            metadata={
                "key_1": "value_1"
            }
        )
        artifact.add_file("my_artifact.txt")


    # Note
    ## This shows the artifact upload via context manager
    ## But in this code, version will not change since the artfact has no change.