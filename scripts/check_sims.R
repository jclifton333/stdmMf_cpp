library(ggplot2)
library(readr)

data_dir = "../data/2017-06-15_01-17-57"
history_file = paste(data_dir, "ebola_Gravity-Gravity_history.txt", sep="/")

## history = read.table(text = history, sep = ",", header = TRUE)

history = read_delim(history_file, ",", trim_ws = TRUE, col_names = TRUE)

mean_inf = aggregate(inf ~ agent + time, data = history, FUN = mean)

p = ggplot(data = mean_inf)
p = p + geom_line(aes(x = time, y = inf, col = agent))
print(p)
