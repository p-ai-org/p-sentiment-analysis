import pandas as pd

def main():
    print(get_input())

def get_input():
    start_date = "2019-09-01"
    end_date = "2019-11-09"
    date_range = pd.date_range(start_date, end_date)

    filenames = ["output/daily/day_"+date.date().isoformat()+".csv" for date in date_range]
    files = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return pd.concat(files).reset_index()

if __name__ == "__main__":
    main()