import pandas as pd


def selection():
    df = pd.read_excel('reports/metrics.xlsx')
    metrics_df = df.sort_values(by='Accuracy', ascending=False)
    best_metrics = metrics_df['Model'].values[0]
    best_df = open("reports/best_model.txt", "w+")
    best_df.write(str(best_metrics))
    best_df.close()


if __name__ == "__main__":
    selection()
