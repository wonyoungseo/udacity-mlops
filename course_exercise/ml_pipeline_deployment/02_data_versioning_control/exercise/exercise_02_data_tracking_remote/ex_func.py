import sys
import pandas as pd


def create_ids(id_count: str) -> None:
    """ Generate a list of IDs and save it as a csv."""

    df = pd.DataFrame(columns=['ids'])
    for i in range(int(id_count)):
        df.loc[i, 'ids'] = i

    df.to_csv("./id.csv", encoding='utf-8', sep='\t', index=False)

if __name__ == "__main__":
    create_ids(sys.argv[1])