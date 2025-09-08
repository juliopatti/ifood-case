
import pandas as pd  # para notebook 2

def get_channels_df(df_ch):
    df = df_ch.copy()
    unique_ch = df['channels'].apply(list).astype(str).unique().tolist()
    channels = []
    for ch in unique_ch:
        channels+=eval(ch)
    channels = list(set(channels))
    try:
        channels.remove('NA')
    except:
        True
    return channels

def one_hot_encode(df_, cols, drop_first=False):
    """
    One Hot Encoding com pyspark.pandas,
    mantendo o nome da categoria como nome da coluna.
    """
    df = df_.copy()
    return pd.get_dummies(
        df,
        columns=cols,
        prefix=None,      # <<-- NÃO usar string
        prefix_sep="_",    # sem separador
        drop_first=drop_first,
        dtype=int 
    )

def get_offer_formated():
    # Leitura
    path_offers = "../data/processed/silver/offers.parquet"
    df_offers_silver = pd.read_parquet(path_offers)
    df_offers_silver['offer_id'] = df_offers_silver['label_offer'].copy()

    # Channels explode
    df_offers_silver['channels'] = df_offers_silver['channels'].apply(list)
    channels = get_channels_df(df_offers_silver)
    df_offers_silver.channels = df_offers_silver.channels.astype(str)
    for ch in channels:
        df_offers_silver[ch] = df_offers_silver.channels.str.contains(ch).astype(int)

    # one_hot
    df_offers_silver = one_hot_encode(df_offers_silver, cols=['offer_type', 'label_offer'])
    df_offers_silver = df_offers_silver.drop(
        columns=['channels']).rename(columns={'offer_id': 'label_offer'})
    return df_offers_silver



def build_df_agg_synthetic_scenario(df_profiles, df_offers, offer_list):
    """
    Constrói df_agg supondo que TODOS os perfis receberam exatamente as ofertas em offer_list.
    
    Parâmetros:
    -----------
    df_profiles : DataFrame
        Contém account_id e variáveis fixas do cliente.
    df_offers : DataFrame
        Contém as características de cada oferta (com 'label_offer').
    offer_list : list
        Lista de ofertas simuladas (ex: [4,5,8]).
    
    Retorna:
    --------
    df_agg : DataFrame
        Agregado por account_id, com soma/média/max conforme regras.
    """

    # Seleciona apenas as ofertas que queremos simular
    offers_selected = df_offers[df_offers["label_offer"].isin(offer_list)].copy()

    # Produto cartesiano: cada cliente × cada oferta da lista
    df_acc = df_profiles.copy().assign(key=1)
    df_off = offers_selected.assign(key=1)
    df_features = df_acc.merge(df_off, on="key").drop("key", axis=1)

    # -------------------------------
    # Agregação por conta
    # -------------------------------
    columns_to_keep_first = ['credit_card_limit', 'gender', 'dias_de_registro', 'anos_de_ifood', 'age']
    columns_to_get_mean = ['max_discount', 'discount_value', 'min_value', 'duration']
    columns_to_get_sum = ['social', 'mobile', 'web','offer_type_bogo', 'offer_type_discount', 
                          'offer_type_informational',"label_offer_1","label_offer_2","label_offer_3",
                          "label_offer_4", "label_offer_5","label_offer_6","label_offer_7",
                          "label_offer_8", "label_offer_9","label_offer_10"]

    # soma
    df_sum_agg = df_features.groupby("account_id")[columns_to_get_sum].sum()
    # média
    df_mean_agg = df_features.groupby("account_id")[columns_to_get_mean].mean()
    # atributos fixos
    df_remain_agg = df_features.groupby("account_id")[columns_to_keep_first].max()

    # juntar tudo
    df_agg = pd.concat([df_sum_agg, df_mean_agg, df_remain_agg], axis=1).dropna().reset_index()
    df_agg["n_off"] = int(len(offer_list))

    return df_agg

