from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    return model.fit()