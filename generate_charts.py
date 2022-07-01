from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def read_path(dir_path):
    filename = "experiment_gens_bests.csv"
    dir_path = Path(dir_path)
    sub_dirs = list(dir_path.glob("*"))
    df_all = pd.DataFrame()
    for dir_path in tqdm(sub_dirs):
        f = dir_path.joinpath(filename)
        if not f.exists():
            print(f"{f} not found")
            continue
        df = pd.read_csv(f, usecols=["fit", "gen"])
        # fix the fitness format
        df["Cosine similarity"] = df["fit"].apply(lambda x: float(x.split(",")[0][1:]) if isinstance(x, str) else x)
        df["Iteration (%)"] = 100 * df["gen"] / df["gen"].max()
        # print(df["Iteration (%)"])
        df_all = pd.concat((df_all, df[["Iteration (%)", "Cosine similarity"]]))
    return df_all


def generate(dir_cmaes, dir_hybrid, dir_adam):
    df_cmaes = read_path(dir_cmaes)
    df_cmaes["optimizer"] = "CMA-ES"
    df_hybrid = read_path(dir_hybrid)
    df_hybrid["optimizer"] = "Hybrid"
    df_adam = read_path(dir_adam)
    df_adam["optimizer"] = "Adam"
    df = pd.concat((df_adam, df_hybrid, df_cmaes))
    plt.figure()
    sns.lineplot(data=df, x="Iteration (%)", y="Cosine similarity", hue="optimizer", style="optimizer", ci=95)
    plt.legend(title='')
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    prefixes = ["new_exp", "vader_exp", "fox_exp"]
    for prefix in prefixes:
        print(f"generating {prefix}")
        generate(f"output/{prefix}_evo_local0_sigma_0.2/", f"output/{prefix}_evo_local1_sigma_0.2/",
                 f"output/{prefix}_adam_local1_1000/")
        plt.savefig(f'output/{prefix}_fitness.pdf')
    # generate("output/test/", "output/test/", "output/test/")
