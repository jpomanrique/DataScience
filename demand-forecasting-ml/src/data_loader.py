import pandas as pd

def load_data(path):
    # carregar dataset bruto
    df = pd.read_csv(path, sep=';', index_col=0)

    # converter index para datetime
    df.index = pd.to_datetime(df.index)

    # ordenar por data
    df = df.sort_index()

    # pegar apenas 1 série
    df = df.iloc[:, 0].to_frame(name='value')

    # corrigir formato numérico
    df['value'] = (
        df['value']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    # agregação diária
    df = df.resample('D').mean()

    # remover nulos
    df = df.dropna()

    print(f"Dataset carregado: {df.shape[0]} registros")

    return df