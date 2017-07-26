rm(list=ls(all=TRUE))

require(maptools)
require(PBSmapping)
require(rgeos)
require(rgdal)

## download data and code for Ebola paper from
## http://datadryad.org/resource/doi:10.5061/dryad.k95j3.2

## directory to unziped data
data_dir = "~/Downloads/DataAndCode/DataAndcode"


## polygon data set
polygons = readRDS(paste(data_dir, "WestAfricaCountyPolygons.rds", sep="/"))
centroids = calcCentroid(SpatialPolygons2PolySet(polygons), rollup = 1)
region = as.integer(as.factor(polygons$ISO))

## outbreaks data set
outbreaks = readRDS(paste(data_dir,
                          "OutbreakDateByCounty_Summer_AllCountries.rds",
                          sep="/"))

## get adjacency matrix
adj = gTouches(polygons, byid = TRUE)
stopifnot(all(adj == t(adj)))

## find islands
connected = c(1)
for(i in 1:length(polygons)) {
  for(c in connected) {
    connected = c(connected, which(adj[c,]))
    connected = sort(unique(connected))
  }
  if (length(connected) == length(polygons)) {
    break;
  }
}

hung = which(!(1:length(polygons) %in% connected))

## Right now only know how to deal with a lone island.  If the next
## two lines fail, need to think about connecting subgroups.
stopifnot(length(hung) == 1)
stopifnot(hung[1] == 158)

## connect to closest centroid
centroid_dist = as.matrix(dist(centroids))
diag(centroid_dist) = Inf
for(h in hung) {
  neigh = which.min(centroid_dist[h,])
  adj[h, neigh] = TRUE
  adj[neigh, h] = TRUE
}

## run checks on new adjacency matrix
for(h in hung) {
  stopifnot(sum(adj[h,]) == 1)
  stopifnot(sum(adj[,h]) == 1)
}

connected = c(1)
for(i in 1:length(polygons)) {
  for(c in connected) {
    connected = c(connected, which(adj[c,]))
    connected = sort(unique(connected))
  }
  if (length(connected) == length(polygons)) {
    break;
  }
}
stopifnot(length(connected) == length(polygons))

## save edges
edges = which(adj, arr.ind = TRUE)
edges = edges[which(edges[, 1] < edges[, 2]), ]
edges = edges - 1 ## convert to zero based indexing
write.table(edges, file = "ebola_edges.txt", sep = " ",
            row.names = FALSE, col.names = FALSE, quote = FALSE)


## read in supplemental data to get x and y coordinates
admn = read.csv(paste(data_dir, "AdmUnits_WBtwn.csv", sep="/"), as.is=TRUE)


## combine necessary data for simulations
ebola = data.frame(county = polygons$county_names,
                   country = polygons$ISO,
                   region = region,
                   outbreaks = outbreaks$infection_date,
                   population = polygons$pop.size,
                   x = centroids$X,
                   y = centroids$Y
                   )

## fix names
ebola$county = gsub("\\s+", "_", ebola$county)
ebola$country = gsub("\\s+", "_", ebola$country)

## need to change date of outbreak to days
first_date = min(ebola$outbreaks, na.rm = TRUE)
ebola$outbreaks = ifelse(is.na(ebola$outbreaks),
                        -1,
                        ebola$outbreaks - first_date)
## ebola paper cuts off outbreaks past 157 days
ebola$outbreaks = ifelse(ebola$outbreaks > 157, -1, ebola$outbreaks)


## write to files
for(n in names(ebola)) {
  write.table(ebola[[n]], file = paste("ebola_", n, ".txt", sep=""),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}
