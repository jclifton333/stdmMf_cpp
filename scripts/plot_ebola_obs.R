rm(list=ls(all=TRUE))

require("rgdal")
require("rgeos")
require("maptools")
require("ggplot2")
require("plyr")
require("gpclib")
require("readr")
require("gganimate")
require("viridis")

gpclibPermit()


polygons = readRDS("~/Downloads/DataAndCode/DataAndCode/WestAfricaCountyPolygons.rds")
polygons.data = polygons@data
polygons.data$id = rownames(polygons@data)
polygons.data$node = 0:289
polygons = gSimplify(polygons, tol=0.01, topologyPreserve=TRUE)
attributes(polygons)$data = polygons.data
polygons.points = fortify(polygons)
polygons.df = join(polygons.points, polygons@data, by="id")


outbreaks = read_csv("../src/data/ebola_outbreaks.txt", col_names = FALSE)
names(outbreaks) = c("day")
outbreaks$day = ifelse(outbreaks$day < 0, NA, outbreaks$day)
outbreaks$node = 0:289

obs_polygons_data = join(outbreaks, polygons.df, by = "node")

p = ggplot(obs_polygons_data) +
  aes(long,lat,group=group,fill=day) +
  geom_polygon() +
  geom_path(color="gray") +
  scale_fill_viridis("Day of outbreak") +
  coord_equal() +
  theme(legend.position="right",
        panel.background = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        axis.title = element_blank())

print(p)

ggsave("../data/figures/ebola_obs_outbreaks.pdf", p)

## p = ggplot(polygons.df) +
##   aes(long,lat,group=group,fill=id) +
##   geom_polygon() +<
##   geom_path(color="gray") +
##   coord_equal() +
##   theme(legend.position="none")
## print(p)
