"""Monitoring and retraining: score forecasts, detect drift, alert, retrain behind a gate.

* :mod:`~nem_forecast.monitor.scoring`: realised error of logged forecasts once actuals land
* :mod:`~nem_forecast.monitor.drift`: data, prediction and target drift with Evidently
* :mod:`~nem_forecast.monitor.alerts`: thresholds, and GitHub issues when one is breached
* :mod:`~nem_forecast.monitor.retrain`: champion/challenger comparison before any promotion
* :mod:`~nem_forecast.monitor.site`: the report published to GitHub Pages
"""
