rm(list = ls(all = TRUE))

library(ggplot2)
library(readr)
library(viridis)
library(plyr)

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

mods_miss = data.frame(
              miss_prop = c(0.0, 0.25, 0.5, 0.75, 1.0),
              mod_system = c("NoImNoSo", "Mixture-NoImNoSo-75-PosImNoSo-25",
                          "Mixture-NoImNoSo-50-PosImNoSo-50",
                          "Mixture-NoImNoSo-25-PosImNoSo-75", "PosImNoSo"),
              mod_agent = c("NoImNoSo", "NoImNoSo", "NoImNoSo", "NoImNoSo", "NoImNoSo")
         )

sim_data = NULL
for (net_index in 1:nrow(nets)) {
  for (mod_index in 1:nrow(mods_miss)) {
    ## read file
    file_name = paste(data_dir, "/",
                      nets$label[net_index], "_",
                      mods_miss$mod_system[mod_index], "-",
                      mods_miss$mod_agent[mod_index], "_",
                      "raw.txt", sep="")

    raw_data = read_delim(file_name, ",",
                          col_types = c(col_character(), ## network
                                        col_character(), ## model
                                        col_character(), ## agent
                                        col_double(), ## value_mean
                                        col_double(), ## value_ssd
                                        col_double() ## time_mean
                                        ))

    ## add data for plotting
    raw_data$network_type = nets$type[net_index]
    raw_data$network_size = nets$size[net_index]
    raw_data$miss_prop = mods_miss$miss_prop[mod_index]

    sim_data = rbind(sim_data, raw_data)
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
                      color = agent), size = 1.5)
p = p + facet_grid(network_type ~ network_size)
p = p + scale_color_manual("Treatment Strategy",
                           breaks = c("No treatment", "Proximal", "Random", "Myopic",
                                      "Model based", "Model free"),
                           values = c("#E69F00", "#009E73", "#F0E442",
                                      "#0072B2", "#D55E00", "#CC79A7"))
p = p + ylab("Estimated mean value")
p = p + xlab("Misspecification proportion")
p = p + theme(panel.spacing = unit(1.5, "lines"))
print(p)

ggsave("../data/figures/toy_sim_results.pdf", p)
