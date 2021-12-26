# Tracking Data Locally with DVC

## Direction
- track data using a local "remote" 
- explore the fundamentals of DVC.

### 1. Set up
- Create a conda environment with DVC installed. 
- Initialize DVC and Git using dvc init and git init, respectively.

### 2. Create Local Remote
- Create a folder outside of your current working directory and set it as your local "remote" storage.

### 3. Track files in DVC
- In Python, write a (simple) function that creates a list of identifiers (e.g. 1 to 10) and save it as a csv.
- Track that file using DVC and push it to your local remote.
- Re-run your code but now double the number of identifiers. Now track and push the newly updated csv file.
- Optionally, explore the created .dvc file and what see how it changes when you update the tracked csv.



## Solution

### Setup
- Set up repository and local remote

```bash
git init
dvc init
mkdir ../local_remote
dvc remote add -d localremote ../local_remote
```

### Write simple code and run it

```python
# ex_func.py

import sys
import pandas as pd


def create_ids(id_count: str) -> None:
    """ Generate a list of IDs and save it as a csv."""
    ids = [i for i in range(int(id_count))]
    df = pd.DataFrame(ids)
    df.to_csv("./id.csv", index=False)


if __name__ == "__main__":
    create_ids(sys.argv[1])
```

```bash
python ./ex_func.py 10
```

### Add and push data.
```bash
python ./ex_func.py 10
dvc add id.csv
git add .gitignore id.csv.dvc
git commit -m "Initial commit of tracked sample.csv"
dvc push
```





