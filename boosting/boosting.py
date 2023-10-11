# Импортируем необходимые библиотеки
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Загружаем данные и преобразуем их, как вы сделали ранее
with open('lenses.txt', 'r', encoding='utf-8') as fr:
    lenses = [inst.strip().split('-') for inst in fr.readlines()]

lenses_target = []
for each in lenses:
    lenses_target.append(each[-1])

lensesLabels = ['возраст', 'рецепт', 'астигматизм', 'скорость разрыва слез']
lenses_dict = {}

for each_label in lensesLabels:
    le = LabelEncoder()
    lenses_dict[each_label] = le.fit_transform([row[lensesLabels.index(each_label)] for row in lenses])

lenses_pd = pd.DataFrame(lenses_dict)

# Применяем Label Encoding к целевым данным (lenses_target)
le_target = LabelEncoder()
lenses_target = le_target.fit_transform(lenses_target)

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(lenses_pd, lenses_target, test_size=0.2, random_state=42)

# Создаем объект DMatrix для xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Определяем параметры для xgboost
params = {
    'objective': 'multi:softmax',  # Многоклассовая классификация
    'num_class': len(set(y_train)),  # Количество классов
    'max_depth': 10,  # Глубина деревьев
    'eta': 0.1,  # Шаг обучения
    'eval_metric': 'mlogloss'  # Метрика для оценки качества
}

# Обучаем модель XGBoost
model = xgb.train(params, dtrain, num_boost_round=50)

# Предсказываем на тестовом наборе
y_pred = model.predict(dtest)

# Оцениваем качество модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
