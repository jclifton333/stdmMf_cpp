import argparse
import pandas

def process_rep(rep_data):
    pass

def main(data_file):
    df = pandas.read_csv(data_file)

    print df.describe()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", type=str, required=True)

    args = ap.parse_args()

    main(args.data_file)
