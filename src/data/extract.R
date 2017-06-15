rm(list=ls(all=TRUE))

require(maptools)
require(PBSmapping)

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


## id data set
ids = readRDS(paste(data_dir, "MobilityDataIDs.rds", sep="/"))
## fix special character issues
ids$county[which(ids$county=="Gbapolu")] = "Gbarpolu"
ids$county[which(ids$county=="BafatÃ¡")] = "Bafatá"
ids$county[which(ids$county=="GabÃº")] = "Gabú"


## read in supplemental data to get x and y coordinates
admn = read.csv(paste(data_dir, "AdmUnits_WBtwn.csv", sep="/"), as.is=TRUE)


## combine necessary data for simulations
ebola = data.frame(county = ids$county,
                   country = ids$country,
                   region = region,
                   loc = ids$loc,
                   outbreaks = outbreaks$infection_date,
                   population = polygons$pop.size / mean(polygons$pop.size),
                   x = centroids$X,
                   y = centroids$Y
                   )

## fix names
ebola$county = gsub("\\s+", "_", ebola$county)
ebola$country = gsub("\\s+", "_", ebola$country)
ebola$loc = gsub("\\s+", "_", ebola$loc)

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
