## Pulling the Raw Data

In the process of initially working with this data I discovered that the `subset` function 
in the `copernicusmarine` package does not work the way one would want it to. Rather than 
having a memory footprint equivalent to the subset of data you are pulling down, even if you
pull something as a small as a few MB's it will go ahead and carve up all of the available 
memory on my machine and then crash whatever process is running. This leaves us with using 
the `get` function which will download full files from the Copernicus Marine Service. 

The issue with this is that there is a *lot* of such data. For the geo-physics data, for 
example, each day is a 1.34GB file and therefore 10years of data works out to around ~5TB. 
This both takes a while to download and far exceeds the amount of storage I had available
on my machine. So I had to pull the data down and move it over to an external hard drive as I 
did so. 

Annoyingly I found that using the `get` function to directly push data to an external drive
would sometimes cause crashes as well and that the only rememdy was to download the data 
locally and then move it over to the external drive using the `mv` command. But this in turn
was only efficient if gets and mv's were done in parallel. 

So all in all we get the `nicolaus` script in this directory which pulls however many processes
you have of files down at a time to a local directory, moves them to an external drive,
deletes the data on the local drive, and then repeats the process until all the data has been
moved to the external disk. For ten years of data this takes about 24 hours. So I also decided
to just pull down the data once and store it so that if I want to make changes to the 
summarized data I'll put in AWS I can do so without having to pull down the data again.

### Some Examples of Nicolaus

The idea is pretty simple. You just need to specify the start and end date of the data you want
to pull, the local directory you want to stage the data in, the external directory you want to
store the data in, the dataset id you want to pull, the number of workers you want to use, and
the pattern of the files you want to pull. That pattern designates how the dates are indicated 
in the file names on the Copernicus Marine Service for that dataset in question.

Note that the `local-output-dir` has to not exist or be empty. 

```bash
python nicolaus.py \
    --start-date 2020-01-01 \
    --end-date 2022-12-31 \
    --local-output-dir test_input \
    --external-output-dir /Volumes/Copernicus1  \
    --dataset-id cmems_mod_glo_bgc_my_0.25deg_P1D-m \
    --num-workers 7 \
    --pattern "_{year}{month}{day}.nc"
```

```bash
python nicolaus.py \
    --start-date 2020-01-01 \
    --end-date 2022-12-31 \
    --local-output-dir test_input \
    --external-output-dir /Volumes/Copernicus1  \
    --dataset-id cmems_mod_glo_phy_myint_0.083deg_P1D-m  \
    --num-workers 7 \
    --pattern ".*_{year}{month}{day}_.*"
```

