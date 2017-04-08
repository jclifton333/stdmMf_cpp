"""Sequential Q-Learning using Neural Networks"""

import pandas
import tensorflow as tf
import os
import argparse
from tqdm import tqdm

def format_obs(data):
    data.sort_values(["rep", "time", "node"])
    assert len(data["rep"].unique()) == 1
    assert len(data["time"].unique()) == 1

    ## pull data out of pandas into tuples and lists
    s1 = (data["curr_inf"].tolist(), data["curr_shield"].tolist())
    trt = data["trt"].tolist()
    s2 = (data["next_inf"].tolist(), data["next_shield"].tolist())
    r = - float(sum(s2[0])) / float(len(s2[0]))

    ## convert to tf features
    s1 = (tf.train.Feature(int64_list=tf.train.Int64List(value=s1[0])),
          tf.train.Feature(float_list=tf.train.FloatList(value=s1[0])))

    trt = tf.train.Feature(int64_list=tf.train.Int64List(value=trt))

    s2 = (tf.train.Feature(int64_list=tf.train.Int64List(value=s2[0])),
          tf.train.Feature(float_list=tf.train.FloatList(value=s2[0])))

    r = tf.train.Feature(float_list=tf.train.FloatList(value=[r]))

    return s1, trt, s2, r

def format_and_write_data(input_file, output_file):
    print("Reading csv data")
    raw_data = pandas.read_csv(input_file)

    print("Sorting csv data")
    raw_data = raw_data.sort_values(["rep", "time", "node"])

    print("Grouping data")
    grouped_data = raw_data.groupby(["rep", "time"])

    print("Writing data")
    writer = tf.python_io.TFRecordWriter(output_file)

    with tqdm(total = len(grouped_data)) as pbar:
        for (rep, time), obs in grouped_data:
            fmt_obs = format_obs(obs)

            features = tf.train.Features(
                feature = {
                    "curr_inf": fmt_obs[0][0],
                    "curr_shield": fmt_obs[0][1],
                    "trt": fmt_obs[1],
                    "next_inf": fmt_obs[2][0],
                    "next_shield": fmt_obs[2][1],
                    "outcome": fmt_obs[3]
                    })

            example = tf.train.Example(features = features)

            writer.write(example.SerializeToString())

            pbar.update()

    writer.close()


def main(input_file):
    output_file = os.path.splitext(input_file)[0] + ".tfrecords"
    if os.path.exists(output_file):
        raise RuntimeError("Output file %s already exists." % output_file)
    data = format_and_write_data(input_file, output_file)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file",
                    type = str,
                    required = True)

    args = ap.parse_args()

    main(args.input_file)
