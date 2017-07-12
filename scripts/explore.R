rm(list = ls(all = TRUE))

library(readr)
library(ggplot2)
library(viridis)

data_dir = "../data/2017-07-11_15-37-15"

coefs_emf = read_csv(sprintf("%s/%s", data_dir, "coefs_emf.txt"))

coefs_etpf = read_csv(sprintf("%s/%s", data_dir, "coefs_etpf.txt"))


## p_emf = ggplot() +
##   geom_line(data = coefs_emf,
##             mapping = aes(x = time, y = coef, lty = as.factor(index), group = index)) +
##   facet_wrap(~ rep)

## print(p_emf)



p_etpf = ggplot() +
  geom_line(data = subset(coefs_etpf, coefs_etpf$index == 1),
            mapping = aes(x = time, y = coef, col = rep, group = rep)) +
  scale_color_viridis() +
  facet_wrap(~ rep)

print(p_etpf)
