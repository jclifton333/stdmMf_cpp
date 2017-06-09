rm(list=ls(all=TRUE))

## download data and code for Ebola paper from
## http://datadryad.org/resource/doi:10.5061/dryad.k95j3.2

data_dir = "~/Downloads/DataAndCode/DataAndcode"

polygons = readRDS(paste(data_dir, "WestAfricaCountyPolygons.rds", sep="/"))

outbreaks = readRDS(paste(data_dir,
                          "OutbreakDateByCounty_Summer_AllCountries.rds",
                          sep="/"))
outbreaks = outbreaks[order(as.character(outbreaks$county_names)),]

ids = readRDS(paste(data_dir, "MobilityDataIDs.rds", sep="/"))
ids$county[which(ids$county=="Gbapolu")] = "Gbarpolu"
ids$county[which(ids$county=="BafatÃ¡")] = "Bafatá"
ids$county[which(ids$county=="GabÃº")] = "Gabú"
ids = ids[order(ids$county),]

admn = read.csv(paste(data_dir, "AdmUnits_WBtwn.csv", sep="/"), as.is=TRUE)


## pull out necessary data
ebola = data.frame(county = ids$county,
                   country = ids$country,
                   loc = ids$loc,
                   outbreak = outbreaks$infection_date,
                   population = polygons$pop.size,
                   x = 0,
                   y = 0
                   )

## get x and y from admn
for(i in 1:nrow(ebola)) {
  ## extract to and from that matches ebola location code
  from_vals = admn[which(admn$from_loc == ebola$loc[i]), ]
  to_vals = admn[which(admn$to_loc == ebola$loc[i]), ]
  ## pull out x and y
  x_vals = c(from_vals$from_x, to_vals$to_x)
  y_vals = c(from_vals$from_y, to_vals$to_y)
  ## make sure they are unique
  stopifnot(length(unique(x_vals)) == 1)
  stopifnot(length(unique(y_vals)) == 1)
  ## assign to ebola data set
  ebola$x[i] = unique(x_vals)
  ebola$y[i] = unique(y_vals)
}

## need to change date of outbreak to days
first_date = min(ebola$outbreak, na.rm = TRUE)
ebola$outbreak = ifelse(is.na(ebola$outbreak),
                        -1,
                        ebola$outbreak - first_date)

## write to files
for(n in names(ebola)) {
  write.table(ebola[[n]], file = paste("ebola_", n, ".txt", sep=""))
}
