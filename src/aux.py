import pyspark.pandas as ps
from IPython.display import display
from pathlib import Path

def read_parquet(path_file: str, get_info: bool = True):
    """
    Lê um arquivo Parquet usando pyspark.pandas (pandas-on-Spark)
    e opcionalmente exibe shape e informações básicas.
    """
    df = ps.read_parquet(path_file)

    if get_info:
        print(f"shape: {df.shape}\n")
        print(df.info())

    return df

def get_gender_offer_props(df_offer, df):
    import pandas as pd
    df_base = df.copy()
    df_offers = df_offer.copy()

    for offer in sorted(df_offers['label_offer'].unique().tolist()):
        dist_por_evento = {}

        for evento in ['offer_received', 'offer_viewed', 'offer_completed', 'transaction']:
            if evento == 'transaction':
                df_evt = df_base[df_base['event'] == evento]
            else:
                df_evt = df_base[(df_base['event'] == evento) & (df_base['label_offer'] == offer)]
            df_evt = df_evt.to_pandas()

            # Conta por gênero e normaliza (já converte em pandas Series)
            dist = (df_evt['gender'].value_counts(normalize=True) * 100).round(1)
            dist_por_evento[evento] = dist

        # Junta em DataFrame pandas normal
        df_final = pd.DataFrame(dist_por_evento).fillna(0)

        print(f'\n📦 Oferta: {offer}')
        display(df_final.sort_values('offer_received', ascending=False))


def save_parquet(df: ps.DataFrame, relative_dir: str, filename: str):
    """
    Salva um DataFrame como Parquet dentro da pasta do projeto (raiz).

    Parâmetros
    ----------
    df : pyspark.pandas.DataFrame
        DataFrame a ser salvo.
    relative_dir : str
        Caminho relativo a partir da raiz do projeto, ex.: "data/processed/bronze"
    filename : str
        Nome do arquivo (com extensão .parquet)
    """
    # raiz do projeto = sobe 1 nível a partir da pasta notebooks/
    project_root = Path.cwd().parent
    output_dir = project_root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / filename)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Arquivo {filename} salvo com sucesso.")


