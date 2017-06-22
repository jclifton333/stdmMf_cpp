rm(list = ls(all = TRUE))

library(xtable)
library(readr)
library(reshape2)

filename = "../data/2017-06-21_13-36-31/ebola_Gravity-Gravity_raw.txt"
raw_data = read.csv(filename)


proc_data = raw_data
proc_data$value_mean_ssd = paste(round(raw_data$value_mean, 2), " (",
                                 round(raw_data$value_ssd, 2), ")", sep="")
proc_data = dcast(proc_data, network ~ agent, value.var = "value_mean_ssd")
proc_data$network = NULL
proc_data = proc_data[,c("none", "random", "proximal", "myopic", "vfn")]
names(proc_data) = c("None", "Random", "Proximal", "Myopic", "Model based")

sink("../data/figures/ebola_sim_results.tex")
print(xtable(proc_data, align = "cccccc"), include.rownames = FALSE, floating = FALSE)
sink()
