import os
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.features import create_features
from src.models import train_random_forest, train_arima
from src.evaluate import evaluate

# Garantir pasta de imagens
os.makedirs("images", exist_ok=True)

# 1. Carregar dados
print("Carregando dados...")
df = load_data("data/LD2011_2014.txt")

# 2. Criar features
df_feat = create_features(df)

# 3. Split treino/teste
train_size = int(len(df_feat) * 0.8)

train = df_feat[:train_size]
test = df_feat[train_size:]

X_train = train.drop('value', axis=1)
y_train = train['value']

X_test = test.drop('value', axis=1)
y_test = test['value']

# 4. Treinar Random Forest
print("Treinando Random Forest...")
rf_model = train_random_forest(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 5. Treinar ARIMA
print("Treinando ARIMA...")
arima_model = train_arima(df['value'][:train_size])
arima_preds = arima_model.forecast(steps=len(test))

# 6. Avaliação
rf_mae, rf_rmse = evaluate(y_test, rf_preds)
arima_mae, arima_rmse = evaluate(y_test, arima_preds)

print("\nResultados:")
print("Random Forest -> MAE:", rf_mae, "RMSE:", rf_rmse)
print("ARIMA -> MAE:", arima_mae, "RMSE:", arima_rmse)

# 7. Forecast Plot
plt.figure(figsize=(10,5))

plt.plot(y_test.index, y_test, label="Real")
plt.plot(y_test.index, rf_preds, linestyle='--', label="Random Forest")
plt.plot(y_test.index, arima_preds, linestyle=':', label="ARIMA")

plt.xlabel("Time")
plt.ylabel("Electricity Demand")
plt.title("Forecast vs Actual Demand")

plt.legend()
plt.grid()

plt.savefig("images/forecast_plot.png", dpi=300)
plt.show()

# 8. Residuals
errors = y_test - rf_preds

plt.figure()
plt.plot(y_test.index, errors)

plt.xlabel("Time")
plt.ylabel("Residual")
plt.title("Residuals Over Time")

plt.grid()

plt.savefig("images/error_time_series.png", dpi=300)
plt.show()

# 9. Error Distribution
plt.figure()
plt.hist(errors, bins=30)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

plt.grid()

plt.savefig("images/error_distribution.png", dpi=300)
plt.show()

# 10. Pipeline Diagram
plt.figure()

plt.text(0.1, 0.5, "Raw Data")
plt.text(0.3, 0.5, "Preprocessing")
plt.text(0.5, 0.5, "Feature Engineering")
plt.text(0.7, 0.5, "Model")
plt.text(0.9, 0.5, "Evaluation")

plt.axis('off')

plt.savefig("images/pipeline_diagram.png", dpi=300, bbox_inches='tight')
plt.show()