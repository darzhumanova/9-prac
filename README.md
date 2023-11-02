from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка датасета Iris
iris = load_iris()
X = iris.data
y = iris.target

# Разбиваем датасет на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Создаем модель градиентного бустинга для классификации
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)

# Обучаем модель на обучающих данных
gb_classifier.fit(X_train, y_train)

# Делаем прогноз на тестовых данных
y_pred = gb_classifier.predict(X_test)

# Оцениваем производительность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
