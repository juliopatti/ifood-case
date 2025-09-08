import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import math
from IPython.display import display
import numpy as np


def scatter_amount(df, plot_low_region=False):
    df = df.copy()
    # Agrupar por usuário
    df_user = df.groupby("account_id").agg({
        "amount": "sum",
        "credit_card_limit": "first",
        "anos_de_ifood": "first"
    }).reset_index()

    # Adiciona a coluna com a reta desejada
    df_user["linha_reta"] = (1/170) * df_user["credit_card_limit"]
    df_user = df_user[df_user['credit_card_limit'] > 0].copy()

    if plot_low_region:
        df_user = df_user[df_user['amount'] < df_user["linha_reta"]]

    # Converter para pandas (se vier de pyspark.pandas)
    try:
        df_user = df_user.to_pandas()
    except AttributeError:
        pass

    # Ordenar anos pela contagem de exemplares
    anos_order = (
        df_user["anos_de_ifood"]
        .value_counts()
        .sort_index(ascending=False)
        .index.tolist()
    )

    # Correlação
    pearson_corr, p_pearson = pearsonr(df_user["amount"].tolist(), df_user["credit_card_limit"].tolist())
    spearman_corr, p_spearman = spearmanr(df_user["amount"].tolist(), df_user["credit_card_limit"].tolist())

    print(f"Correlação de Pearson:  {pearson_corr:.3f} (p={p_pearson:.4f})")
    print(f"Correlação de Spearman: {spearman_corr:.3f} (p={p_spearman:.4f})")

    # Figura maior
    plt.figure(figsize=(12, 8))

    # Paleta categórica mais distinta
    palette = sns.color_palette("tab20", n_colors=len(anos_order))

    sns.scatterplot(
        data=df_user,
        x="credit_card_limit",
        y="amount",
        hue="anos_de_ifood",
        hue_order=anos_order,
        palette=palette,
        alpha=0.7,
        s=50  # aumenta tamanho dos pontos
    )
    sns.regplot(
        data=df_user,
        x="credit_card_limit",
        y="amount",
        scatter=False,
        color="red",
        label="Regressão"
    )
    if not plot_low_region:
        plt.plot(
            df_user["credit_card_limit"],
            df_user["linha_reta"],
            color="green",
            linestyle="--",
            label="(1/170)*limite"
        )

    plt.title("Relação: Limite do Cartão vs. Gasto Total no Mês", fontsize=14)
    plt.xlabel("Limite do Cartão", fontsize=12)
    plt.ylabel("Amount (soma por usuário)", fontsize=12)

    # Legenda fora do gráfico, com fonte maior
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.,
        title="Anos de iFood",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()


def scatter_amount_by_year(df, plot_low_region=False, hue_col=None, fixed_palette=None):
    """
    Plota scatters de amount vs credit_card_limit separados por anos_de_ifood.
    Opcionalmente colore os pontos de acordo com outra coluna (ex.: gender).
    As cores podem ser fixadas via fixed_palette (dict).
    """
    df = df.copy()
    # Agrupar por usuário
    df_user = df.groupby("account_id").agg({
        "amount": "sum",
        "credit_card_limit": "first",
        "anos_de_ifood": "first",
        "gender": "first",
        "dias_de_registro": "first",
        "age": "first"
    }).reset_index()

    # Linha de referência
    df_user["linha_reta"] = (1/170) * df_user["credit_card_limit"]
    df_user = df_user[df_user['credit_card_limit'] > 0].copy()
    if plot_low_region:
        df_user = df_user[df_user['amount'] < df_user["linha_reta"]]

    # Converter para pandas se for pyspark.pandas
    try:
        df_user = df_user.to_pandas()
    except AttributeError:
        pass

    anos = sorted(df_user["anos_de_ifood"].unique())
    n = len(anos)

    # Grid fixo de 3 colunas
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), sharey=True)
    axes = axes.flatten()

    for ax, ano in zip(axes, anos):
        subset = df_user[df_user["anos_de_ifood"] == ano]

        # Scatter com cor fixa
        sns.scatterplot(
            data=subset,
            x="credit_card_limit",
            y="amount",
            hue=hue_col,
            palette=fixed_palette,
            alpha=0.6,
            ax=ax
        )
        sns.regplot(
            data=subset,
            x="credit_card_limit",
            y="amount",
            scatter=False,
            color="red",
            ax=ax
        )
        if not plot_low_region:
            ax.plot(
                subset["credit_card_limit"],
                subset["linha_reta"],
                color="green",
                linestyle="--"
            )

        # Correlações
        if len(subset) > 1:
            pearson_corr, _ = pearsonr(subset["amount"], subset["credit_card_limit"])
            spearman_corr, _ = spearmanr(subset["amount"], subset["credit_card_limit"])
            corr_text = f"Pearson={pearson_corr:.2f}, Spearman={spearman_corr:.2f}"
        else:
            corr_text = "Poucos dados"

        ax.set_title(f"Ano {ano} – {corr_text}")
        ax.set_xlabel("Limite do Cartão")
        ax.set_ylabel("Amount")

    # Remove eixos extras se sobrar
    for j in range(len(anos), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def scatter_plot(df, x: str, y: str, hue: str = None, add_regression: bool = False, alpha: float = 0.6):
    """
    Gera um scatterplot genérico entre duas colunas de um DataFrame.

    Parâmetros
    ----------
    df : DataFrame
        DataFrame contendo os dados.
    x : str
        Nome da coluna para o eixo X.
    y : str
        Nome da coluna para o eixo Y.
    hue : str, opcional
        Coluna para colorir os pontos (ex.: 'anos_de_ifood').
    add_regression : bool, default=False
        Se True, adiciona linha de regressão linear.
    alpha : float, default=0.6
        Transparência dos pontos.
    """
    try:
        df = df.to_pandas()  # se for pyspark.pandas
    except AttributeError:
        pass

    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=alpha)

    if add_regression:
        sns.regplot(data=df, x=x, y=y, scatter=False, color="red", label="Regressão")

    plt.title(f"Scatterplot: {x} vs {y}", fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.show()


def boxplot_years_cc(df, dropzero=True):
    df = df.copy()
    if dropzero:
        df = df[df.credit_card_limit>0].copy()
    df_user = df.drop_duplicates(subset=["account_id"]).copy().to_pandas()

    plt.figure(figsize=(10,3))
    sns.boxplot(
        data=df_user,
        x="anos_de_ifood",
        y="credit_card_limit"
    )

    plt.title("Distribuição do limite de crédito por anos de iFood")
    plt.xlabel("Anos de iFood")
    plt.ylabel("Limite do cartão")
    plt.tight_layout()
    plt.show()


def get_freq_offered(df):
    df =  df.copy()
    df_received = df[df['event'] == 'offer_received'].copy()
    display(df_received['account_id'].value_counts().value_counts())
    vc_dist = df_received['account_id'].value_counts().value_counts().sort_index()
    vc_dist = vc_dist.to_pandas() 
    vc_dist.plot(kind='bar')
    plt.xlabel('Número de registros por usuário')
    plt.ylabel('Quantidade de usuários')
    plt.title('Distribuição de frequência por account_id')
    plt.tight_layout()
    plt.show()

    return vc_dist, df_received


def plot_metrics_vs_offers(df_ps, include_lucro = False):
    """
    Prepara df_resultado, agrega métricas por 'offer_received'
    e plota em múltiplos eixos. 
    Mantém a legenda em figura separada.

    Parâmetros
    ----------
    df : DataFrame
        Deve ter colunas: account_id, amount, reward, event, offer_id.
    include_lucro : bool
        Se True, calcula métricas de lucro e adiciona ao gráfico.

    Retorna
    -------
    df_resultado : DataFrame
    agg_df : DataFrame
    """
    # Essa eu só consegui no pandas
    df = df_ps.copy().to_pandas()

    # ----------------------------------------
    # calcula lucro se solicitado
    # ----------------------------------------
    if include_lucro:
        df["lucro"] = (df["amount"] + df["reward"]) * 0.12 - df["reward"]

    # ----------------------------------------
    # df_resultado
    # ----------------------------------------
    event_counts = df.pivot_table(
        index="account_id",
        columns="event",
        values="label_offer",
        aggfunc="count",
        fill_value=0
    )
    sums_cols = ["amount", "reward"] + (["lucro"] if include_lucro else [])
    sums = df.groupby("account_id")[sums_cols].sum()

    df_resultado = event_counts.join(sums).reset_index()
    den = df_resultado["amount"] + df_resultado["reward"]
    df_resultado["eff_factor"] = np.where(den > 0, df_resultado["amount"] / den, np.nan)

    # ----------------------------------------
    # agregação
    # ----------------------------------------
    agg_ops = dict(
        clientes=("account_id", "count"),
        amount_medio=("amount", "mean"),
        eff_factor_medio=("eff_factor", "mean"),
        reward_medio=("reward", "mean"),
    )
    if include_lucro:
        agg_ops["lucro_medio"] = ("lucro", "mean")
        agg_ops["lucro_total"] = ("lucro", "sum")

    agg_df = (
        df_resultado
        .groupby("offer_received", as_index=False)
        .agg(**agg_ops)
        .rename(columns={"offer_received": "Total offer_received"})
    )

    assert not agg_df.empty, "agg_df está vazio."
    x = agg_df["Total offer_received"]

    # ----------------------------------------
    # legenda em figura separada
    # ----------------------------------------
    color_map = {
        "Clientes": "tab:blue",
        "Amount Médio": "tab:orange",
        "Eff Factor Médio": "tab:red",
        "Reward Médio": "tab:green",
        "Lucro Médio": "tab:purple",
        "Lucro Total": "tab:brown",
    }

    handles = [
        Line2D([0], [0], marker="o", color=color_map["Clientes"], label="Clientes"),
        Line2D([0], [0], marker="o", color=color_map["Amount Médio"], label="Amount Médio"),
        Line2D([0], [0], marker="o", color=color_map["Eff Factor Médio"], label="Eff Factor Médio"),
        Line2D([0], [0], marker="o", color=color_map["Reward Médio"], label="Reward Médio"),
    ]
    if include_lucro:
        handles.append(Line2D([0], [0], marker="o", color=color_map["Lucro Médio"], label="Lucro Médio"))
        handles.append(Line2D([0], [0], marker="o", color=color_map["Lucro Total"], label="Lucro Total"))

    fig_leg = plt.figure(figsize=(8, 1.2))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=len(handles))
    fig_leg.tight_layout()
    plt.show()

    # ----------------------------------------
    # gráfico principal
    # ----------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax4.spines["right"].set_position(("outward", 120))

    l1, = ax1.plot(x, agg_df["clientes"], marker="o", color=color_map["Clientes"])
    l2, = ax2.plot(x, agg_df["amount_medio"], marker="o", color=color_map["Amount Médio"])
    l3, = ax3.plot(x, agg_df["eff_factor_medio"], marker="o", color=color_map["Eff Factor Médio"])
    l4, = ax4.plot(x, agg_df["reward_medio"], marker="o", color=color_map["Reward Médio"])

    ax1.set_xlabel("Total offer_received")
    ax1.set_ylabel("Clientes", color=color_map["Clientes"])
    ax2.set_ylabel("Amount Médio", color=color_map["Amount Médio"])
    ax3.set_ylabel("Eff Factor Médio", color=color_map["Eff Factor Médio"])
    ax4.set_ylabel("Reward Médio", color=color_map["Reward Médio"])
    ax3.set_ylim(0, 1.1)
    ax1.set_xticks(sorted(x.unique()))

    if include_lucro:
        ax5 = ax1.twinx()
        ax6 = ax1.twinx()
        ax5.spines["right"].set_position(("outward", 180))
        ax6.spines["right"].set_position(("outward", 240))
        ax5.plot(x, agg_df["lucro_medio"], marker="o", color=color_map["Lucro Médio"])
        ax6.plot(x, agg_df["lucro_total"], marker="o", color=color_map["Lucro Total"])
        ax5.set_ylabel("Lucro Médio", color=color_map["Lucro Médio"])
        ax6.set_ylabel("Lucro Total", color=color_map["Lucro Total"])

    fig.suptitle("Métricas em função do Total offer_received", fontsize=12)
    fig.tight_layout()
    plt.show()

    return df_resultado, agg_df

def plot_lucro_por_genero(df, user_col="account_id", time_col="time_since_test_start",
                          lucro_col="lucro", genero_col="gender", event_col="event"):
    """
    Plota a evolução do lucro médio por usuário (geral e por gênero),
    incluindo linhas verticais nos momentos de 'offer_received'.
    """
    df_plot = df.copy().to_pandas()

    # Número total de usuários
    n_users_total = df_plot[user_col].nunique()
    # Número de usuários por gênero
    n_users_gender = df_plot.groupby(genero_col)[user_col].nunique()

    # --- Séries ---
    serie_total = (
        df_plot.groupby(time_col)[lucro_col].sum() / n_users_total
    ).sort_index()

    serie_gender = (
        df_plot.groupby([time_col, genero_col])[lucro_col].sum()
          .unstack()
          .fillna(0)
          .sort_index()
    )
    # Normaliza pelo nº de usuários de cada gênero
    serie_gender = serie_gender.div(n_users_gender)

    # --- Paleta fixa para gêneros conhecidos
    cores_fixas = {"M": "blue", "F": "deeppink", "O": "orange", "NI": "gray"}

    # --- Plot ---
    plt.figure(figsize=(14, 6))

    # Curva geral
    plt.plot(serie_total.index, serie_total.values, label="Geral", linewidth=2, color="black")

    # Curvas por gênero (mantendo consistência de cor)
    for g in serie_gender.columns:
        cor = cores_fixas.get(g, None)  # usa cor fixa se existir
        plt.plot(serie_gender.index, serie_gender[g].values, label=f"Gênero {g}", color=cor)

    plt.title("Lucro médio por usuário ao longo do tempo")
    plt.xlabel(time_col)
    plt.ylabel("Lucro médio por usuário")
    plt.grid(True)

    # Linhas verticais: horários de promoções
    offers = (
        df_plot.loc[df_plot[event_col] == "offer_received", time_col]
        .unique()
        .tolist()
    )
    offers = sorted(offers)

    for t in offers:
        plt.axvline(x=t, color="red", linestyle="--", alpha=0.5)

    # Legenda enxuta
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))
    labels.append("Horário de promoção")
    plt.legend(handles, labels, loc="best")

    plt.show()
