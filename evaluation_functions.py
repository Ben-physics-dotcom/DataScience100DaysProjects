from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score


def add_extra_metrics(results: dict) -> dict:
    """
    Extends the results dict produced by classification_results_func() with
    two metrics it doesn't compute by default: Cohen's Kappa and MCC.

    Requires 'ytest' and 'ypred' to already be present in each results[col_name]
    entry -- classification_results_func() stores both, so this is safe to run
    on anything saved by classification_saving_functions.py.

    Also adds macro-averaged F1 alongside the existing weighted F1, because
    weighted F1 alone hides minority-class failure (a model can score well on
    weighted F1 by nailing the majority class and ignoring everything else).
    """
    for col_name, r in results.items():
        if 'ytest' not in r or 'ypred' not in r:
            raise KeyError(
                f"'{col_name}' has no stored ytest/ypred -- was this produced by "
                f"classification_results_func()? Can't compute Kappa/MCC without "
                f"actual predictions."
            )
        r['cohen_kappa'] = cohen_kappa_score(r['ytest'], r['ypred'])
        r['mcc'] = matthews_corrcoef(r['ytest'], r['ypred'])
        r['f1_macro'] = f1_score(r['ytest'], r['ypred'], average='macro', zero_division=0)
    return results
