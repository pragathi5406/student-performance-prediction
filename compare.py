import matplotlib.pyplot as plt

models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mae = [3.57, 3.52, 3.41]   # lower is better
r2 = [0.024, 0.035, 0.091] # higher is better

x = range(len(models))

plt.figure(figsize=(8,5))
plt.bar(x, mae, width=0.4, label='MAE', color='skyblue')
plt.bar([i + 0.4 for i in x], r2, width=0.4, label='R²', color='lightgreen')
plt.xticks([i + 0.2 for i in x], models)
plt.ylabel('Metric Value')
plt.title('Model Comparison: MAE and R²')
plt.legend()
plt.show()
