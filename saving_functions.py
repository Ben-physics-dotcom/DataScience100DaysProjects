import pickle
import os


def saving_results(results: dict) -> None:
    with open('../Results/regressions.pkl', 'wb') as f:
        pickle.dump(results, f)


def loading_results() -> dict:
    with open('../Results/regressions.pkl', 'rb') as f:
        results = pickle.load(f)
    return results


def saving_classification_results(results: dict) -> None:
    os.makedirs('../Results', exist_ok=True)
    with open('../Results/classifications.pkl', 'wb') as f:
        pickle.dump(results, f)


def loading_classification_results() -> dict:
    with open('../Results/classifications.pkl', 'rb') as f:
        results = pickle.load(f)
    return results
