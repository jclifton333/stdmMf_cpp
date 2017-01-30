import argparse
import pandas

import numpy as np

import multiprocessing

import gmpy

def process(time, time_data):
    g = list(time_data.groupby("from_sim"))
    assert len(g) == 2
    obs_data = g[0][1] if g[0][0] == False else g[1][1]
    sim_data = g[0][1] if g[0][0] == True else g[1][1]

    assert len(obs_data.index) == 1
    obs_sample = obs_data.loc[obs_data.index[0]]

    obs_inf_bin = int(obs_sample["next_inf"], 2)

    trt_bin = int(obs_sample["trt"], 2)

    sample_inf_bin_all = [int(i,2) for i in sim_data["next_inf"]]

    obs_diff = 0

    num_nodes = len(obs_sample["next_inf"])
    num_trt = gmpy.popcount(trt_bin)

    sample_diff = [0.] * len(sim_data.index)

    for sample_ind in range(len(sim_data.index)):
         sample_inf_bin = sample_inf_bin_all[sample_ind]

         diff = gmpy.popcount((sample_inf_bin ^ obs_inf_bin) & trt_bin)
         diff /= float(num_trt)

         obs_diff += diff

         for sample_comp_ind in range(len(sim_data.index)):
             if sample_comp_ind > sample_ind:
                 sample_comp_inf_bin = sample_inf_bin_all[sample_comp_ind]

                 diff = gmpy.popcount((sample_comp_inf_bin ^ sample_inf_bin)
                                      & trt_bin)
                 diff /= float(num_trt)

                 sample_diff[sample_ind] += diff

                 sample_diff[sample_comp_ind] += diff

    obs_diff /= float(len(sim_data.index))

    sample_diff = [i / float(len(sim_data.index) - 1) for i in sample_diff]

    return obs_diff, sample_diff



def process_time(rep, time_data):
    vals = []

    pool = multiprocessing.Pool()
    for g in time_data.groupby("rep"):

        def callback(res):
            vals.append(np.searchsorted(sorted(res[1]), res[0]))

        # pool.apply_async(process, args = g, callback = callback)
        callback(apply(process, g))


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
