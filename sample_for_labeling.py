import pandas as pd

start_date = "2019-09-01"
end_date = "2019-11-09"
date_range = pd.date_range(start_date, end_date)
sample_n = 9000

filenames = ["output/daily/day_"+date.date().isoformat()+".csv" for date in date_range]
files = [pd.read_csv(filename, index_col=0) for filename in filenames]
data = pd.concat(files)
sample = data.sample(n=sample_n)
sample.to_csv("output/sample_for_labeling.csv")
