


# normalize data
def normalise_zero_base(df):
    return df / df.iloc[0] - 1

# normalize data
def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())

