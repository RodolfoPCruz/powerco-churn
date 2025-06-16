def merge_data(X):

    df1, df2 = X
    df1_preprocessed = pipeline_client_data.fit_transform(df1)
    df2_preprocessed = pipeline_price_series.fit_transform(df2)

    #Convert to dataframe if necessary
    if not isinstance(df1_preprocessed, pd.DataFrame):
        df1_preprocessed = pd.DataFrame(df1_preprocessed, index = df1.index)

    if not isinstance(df2_preprocessed, pd.DataFrame):
        df2_preprocessed = pd.DataFrame(df2_preprocessed, index = df2.index)
    
    logging.info("Merging dataframes...")
    merged = pd.merge(
        df1_preprocessed,
        df2_preprocessed,
        on='id',
        how='inner'
    )

    return merged
