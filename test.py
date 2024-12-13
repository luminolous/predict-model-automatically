import pandas as pd
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = pd.read_csv('updated_pollution_dataset.csv')
x = data.drop('Air Quality', axis=1)
y = data['Air Quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1)

model = model.fit(x_train, y_train)

print(f"Akurasi pada data test: {model.score(x_test, y_test):.2f}")

model.export('best_pipeline.py')
