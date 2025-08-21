from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

def report_str(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names)