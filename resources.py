import pandas as pd

def main():
    # print(get_input()
    print(get_third_party_train())

def get_input():
    start_date = "2019-09-01"
    end_date = "2019-11-09"
    date_range = pd.date_range(start_date, end_date)

    filenames = ["output/daily/day_"+date.date().isoformat()+".csv" for date in date_range]
    files = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return pd.concat(files).reset_index()

def get_third_party_train():
    return get_third_party_dataset("thirdparty/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")

def get_third_party_test():
    return get_third_party_dataset("thirdparty/testdata.manual.2009.06.14.csv")

def get_third_party_dataset(filename, encoding=None):
    data = pd.read_csv(filename, encoding=encoding)
    data.columns = ["sentiment", "id", "date", "query", "username", "text"]
    return data


if __name__ == "__main__":
    main()