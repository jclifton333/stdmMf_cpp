rm(list=ls(all=TRUE))

require("rgdal") # requires sp, will use proj.4 if installed
require("rgeos")
require("maptools")
require("ggplot2")
require("plyr")
require("gpclib")
require("readr")
require("gganimate")

gpclibPermit()


polygons = readRDS("~/Downloads/DataAndCode/DataAndCode/WestAfricaCountyPolygons.rds")
polygons.data = polygons@data
polygons.data$id = rownames(polygons@data)
polygons.data$node = 0:289
polygons = gSimplify(polygons, tol=0.1, topologyPreserve=TRUE)
attributes(polygons)$data = polygons.data
polygons.points = fortify(polygons)
polygons.df = join(polygons.points, polygons@data, by="id")


sim_run_data = "../data/2017-07-01_17-33-00"
sim_filename = paste(sim_run_data,
                     "ebola_Gravity-Gravity_history.txt", sep="/")
sim_data_all = as.data.frame(read_delim(sim_filename, trim_ws = TRUE, delim = ","))

sim_data = subset(sim_data_all,
                  sim_data_all$rep == 0
                  & (sim_data_all$agent == "vfn_finite_q_g"
                    | sim_data_all$agent == "myopic"
                    | sim_data_all$agent == "none"))

sim_polygons_data = join(sim_data, polygons.df, by = "node")

p = ggplot(sim_polygons_data) +
  aes(long,lat,group=group,fill=interaction(inf, trt),frame=time) +
  geom_polygon() +
  geom_path(color="gray") +
  coord_equal() +
  facet_wrap(~agent) +
  theme(legend.position="none")

gganimate(p)


## p = ggplot(polygons.df) +
##   aes(long,lat,group=group,fill=id) +
##   geom_polygon() +<
##   geom_path(color="gray") +
##   coord_equal() +
##   theme(legend.position="none")
## print(p)
