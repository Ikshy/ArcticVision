"""Torch-free copy of compute_metrics for use in tests."""
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae  = float(np.mean(np.abs(residuals)))
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(residuals[mask]/y_true[mask]))*100) if mask.sum()>0 else float("nan")
    ss_res = np.sum(residuals**2); ss_tot = np.sum((y_true-y_true.mean())**2)
    r2 = float(1-ss_res/ss_tot) if ss_tot>0 else float("nan")
    return {"rmse":round(rmse,6),"mae":round(mae,6),"mape":round(mape,4),"r2":round(r2,4)}
