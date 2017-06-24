rm(list = ls(all = TRUE))

library(ggplot2)
library(readr)
library(viridis)
library(plyr)
library(dplyr)
library(progress)

data_dir = "../data/2017-05-28_16-13-13"

nets = data.frame(
         type = c("barabasi", "barabasi", "barabasi",
                  "grid", "grid", "grid",
                  "random", "random", "random"),
         size = c(100, 500, 1000,
                  100, 500, 1000,
                  100, 500, 1000),
         label = c("barabasi-100", "barabasi-500", "barabasi-1000",
                   "grid-10x10", "grid-20x25", "grid-40x25",
                   "random-100", "random-500", "random-1000")
       )

mods_miss <- data.frame(
              miss_prop = c(0.0, 0.25, 0.5, 0.75, 1.0),
              mod_system = c("NoImNoSo", "Mixture-NoImNoSo-75-PosImNoSo-25",
                             "Mixture-NoImNoSo-50-PosImNoSo-50",
                             "Mixture-NoImNoSo-25-PosImNoSo-75", "PosImNoSo"),
              mod_agent = c("NoImNoSo", "NoImNoSo", "NoImNoSo", "NoImNoSo", "NoImNoSo")
             )

se = function(x) sqrt(var(x)/length(x))


pb = progress_bar$new(total = nrow(nets) * nrow(mods_miss),
                      format = "loading data [:bar]",
                      clear = TRUE, width = 60)

sim_data = NULL
for (net_index in 1:nrow(nets)) {
  for (mod_index in 1:nrow(mods_miss)) {
    ## read file
    file_name = paste(data_dir, "/",
                      nets$label[net_index], "_",
                      mods_miss$mod_system[mod_index], "-",
                      mods_miss$mod_agent[mod_index], "_",
                      "history.txt", sep="")

    raw_data = read_delim(file_name, ",", trim_ws = TRUE,
                          col_types = cols(col_character(), ## agent
                                           col_integer(), ## rep
                                           col_integer(), ## time
                                           col_integer(), ## node
                                           col_integer(), ## inf
                                           col_double(), ## shield
                                           col_integer() ## trt
                                           ),
                          progress = FALSE)

    ## isolate last time point
    agg_data = raw_data[which(raw_data$time == max(raw_data$time)),]
    agg_data = summarise(group_by(agg_data, agent),
                         value_mean = mean(inf),
                         value_sse = se(inf))

    ## add data for plotting
    agg_data$network_type = nets$type[net_index]
    agg_data$network_size = nets$size[net_index]
    agg_data$miss_prop = mods_miss$miss_prop[mod_index]

    sim_data = rbind(sim_data, agg_data)

    pb$tick()
  }
}

sim_data$agent = mapvalues(sim_data$agent,
                           from = c("none", "proximal", "random", "myopic",
                                    "vfn_finite_q", "br_finite_q"),
                           to = c("No treatment", "Proximal", "Random", "Myopic",
                                      "Model based", "Model free"))

sim_data$network_type = mapvalues(sim_data$network_type,
                                  from = c("barabasi", "grid", "random"),
                                  to = c("Scale-free", "Lattice", "Random"))

p = ggplot(data = sim_data)
p = p + geom_line(aes(x = miss_prop, y = value_mean,
                      color = agent, lty = agent), size = 1.0)
p = p + facet_grid(network_type ~ network_size)
p = p + scale_color_viridis("Treatment Strategy",
                           breaks = c("No treatment", "Proximal", "Random", "Myopic",
                                      "Model based", "Model free"),
                           discrete = TRUE)
p = p + scale_linetype_manual("Treatment Strategy",
                              breaks = c("No treatment", "Proximal", "Random", "Myopic",
                                         "Model based", "Model free"),
                              values = c("No treatment" = 4,
                                         "Proximal" = 4,
                                         "Random" = 4,
                                         "Myopic" = 4,
                                         "Model based" = 1,
                                         "Model free" = 1))
p = p + ylab("Estimated mean value")
p = p + xlab("Misspecification proportion")
p = p + theme(panel.spacing = unit(1.5, "lines"),
              legend.key.width=unit(3,"line"))
print(p)

ggsave("../data/figures/toy_sim_results.pdf", p)
