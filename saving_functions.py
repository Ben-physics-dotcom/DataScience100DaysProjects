import pickle


def saving_results(results: dict) -> None:
    with open('../Results/regressions.pkl', 'wb') as f:
        pickle.dump(results, f)


def loading_results() -> dict:
    with open('../Results/regressions.pkl', 'rb') as f:
        results = pickle.load(f)
    return results
