from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('lenses.txt', 'r', encoding='utf-8') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['возраст', 'рецепт', 'астигматизм', 'скорость разрыва слез']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []

    lenses_pd = pd.DataFrame(lenses_dict)

    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)

    plt.figure(figsize=(16, 12))
    tree.plot_tree(clf, feature_names=lenses_pd.keys(), class_names=clf.classes_, filled=True, rounded=True,
                   fontsize=10)
    plt.savefig('tree.png', dpi=300)  # Сохранить изображение с высоким разрешением
    plt.show()

    prediction = clf.predict([[1, 0, 1, 0]])
    print(prediction)
