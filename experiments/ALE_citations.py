# add imports
if __name__ == '__main__':
    # read model and data
    for n in range(3,11):
        for max_bin_size in range(3,11):
            for _ in range(5):
                ale_exact, t_exact = accumulated_local_effects_exact(model,train_data, 0, 5, 2**max_bin_size, k=2**n)
                ale_approximate, t_approximate = accumulated_local_effects(model,train_data, 0, 5, 2**max_bin_size, k=2**n)
                results = pd.concat([results, pd.DataFrame({'k': 2**n, 'max_bin_size': 2**max_bin_size, 'ALE exact': ale_exact, 'time_exact': t_exact, 'ALE approximate': ale_approximate, 'time_approximate': t_approximate})])
                # think how to save without overwriting previous results. Especially taking into account 30min timelimit
                results.to_csv('/content/drive/MyDrive/ALE_GNN/exact_results_22_05_twitch.csv')#, 'time_approximate': t_approximate'ALE approximate': ale_approximate,