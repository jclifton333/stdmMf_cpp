import argparse
import pandas

import numpy as np

import multiprocessing

def process(time, time_data):
    g = list(time_data.groupby("from_sim"))
    assert len(g) == 2
    obs_data = g[0][1] if g[0][0] == False else g[1][1]
    sim_data = g[0][1] if g[0][0] == True else g[1][1]

    assert len(obs_data.index) == 1
    obs_sample = obs_data.loc[obs_data.index[0]]

    obs_inf = obs_sample["next_inf"]

    obs_diff = 0

    num_nodes = len(obs_inf)

    sample_diff = [0.] * len(sim_data.index)

    for sample_ind, sample_index in enumerate(sim_data.index):
         sample = sim_data.loc[sample_index]

         sample_inf = sample["next_inf"]

         diff = sum(i != j for i,j in zip(obs_inf, sample_inf))
         diff /= float(num_nodes)

         obs_diff += diff

         for sample_comp_index in sim_data.index:
             if sample_index != sample_comp_index:
                 sample_comp = sim_data.loc[sample_comp_index]

                 sample_comp_inf = sample_comp["next_inf"]

                 diff = sum(i != j for i,j in zip(sample_inf, sample_comp_inf))
                 diff /= float(num_nodes)

                 sample_diff[sample_ind] += diff

    obs_diff /= float(len(sim_data.index))

    sample_diff = [i / float(len(sim_data.index) - 1) for i in sample_diff]

    return obs_diff, sample_diff



def process_time(rep, time_data):
    vals = []

    pool = multiprocessing.Pool()
    for g in time_data.groupby("rep"):

        pool.apply_async(process, args = g,
                         callback = (lambda res :
                                     vals.append(
                                         np.searchsorted(
                                             sorted(res[1]), res[0]))))

        # obs_diff, sample_diff = process(*g)

        # vals.append(np.searchsorted(sample_diff, obs_diff))

    pool.close()
    pool.join()

    return vals



def main(data_file):
    df = pandas.read_csv(data_file)
    df["from_sim"] = df["sample"].apply(lambda x : x >= 0)

    for g in df.groupby("time"):
        res = process_time(*g)

        print "time:", g[0]

        print res
        print np.mean(res)
        print np.median(res)
        print np.std(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", type=str, required=True)

    args = ap.parse_args()

    main(args.data_file)
