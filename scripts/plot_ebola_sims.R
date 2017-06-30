rm(list = ls(all = TRUE))

library(xtable)
library(readr)
library(reshape2)

filename = "../data/2017-06-21_13-36-31/ebola_Gravity-Gravity_history.txt"
raw_data = read_delim(filename, ",", trim_ws = TRUE,
                      col_types = cols(col_character(), ## agent
                                       col_integer(), ## rep
                                       col_integer(), ## time
                                       col_integer(), ## node
                                       col_integer(), ## inf
                                       col_integer() ## trt
                                       ),
                      progress = FALSE)

se = function(x) sqrt(var(x)/length(x))

agg_data = raw_data[which(raw_data$time == max(raw_data$time)),]
agg_data = summarise(group_by(agg_data, agent),
                     value_mean = mean(inf),
                     value_sse = se(inf))

agg_data$network = "Ebola"


proc_data = agg_data
proc_data$value_mean_sse = paste(round(agg_data$value_mean, 4), " (",
                                 round(agg_data$value_sse, 4), ")", sep="")
proc_data = dcast(proc_data, network ~ agent, value.var = "value_mean_sse")
proc_data$network = NULL
proc_data = proc_data[,c("none", "random", "proximal", "myopic", "vfn")]
names(proc_data) = c("None", "Random", "Proximal", "Myopic", "Model based")

sink("../data/figures/ebola_sim_results.tex")
print(xtable(proc_data, align = "cccccc"), include.rownames = FALSE, floating = FALSE)
sink()