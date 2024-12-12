from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Model TPOT
model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1)

# Melatih model
model.fit(X_train, y_train)

# Evaluasi
print(f"Akurasi pada data test: {model.score(X_test, y_test):.2f}")

# Ekspor pipeline terbaik
model.export('best_pipeline.py')
