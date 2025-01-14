from src.data_exploration import describe, describe_numerical_data, describe_categorical_data, compare_data
from src.feature_engineering import combine_data, process_cabin_feature, process_ticket_feature, process_name_feature
from src.data_processing import process_data_feature, process_null_values, normalize_values, encode_values, scale_data, split_data
from src.model_building import evaluate_model
from src.model_tuning import perform_clf
from src.survival_predication import prediction

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main(): 
    train_path = "/home/lelou1505/DS/titanic/data/raw/train.csv"
    test_path = "/home/lelou1505/DS/titanic/data/raw/test.csv"
    figure_path = "/home/lelou1505/DS/titanic/reports/figures"

    # Exploration
    data, infos, description = describe(train_path)
    describe_numerical_data(train_path, ['Age','SibSp','Parch','Fare'], figure_path)
    describe_categorical_data(train_path, ['Survived','Pclass','Sex','Ticket','Cabin','Embarked'], figure_path)
    print(compare_data(data, "Survived", column="Pclass", values="Ticket", aggfunc="count"))
    print(compare_data(data, "Survived", column="Sex", values="Ticket", aggfunc="count"))
    print(compare_data(data, "Survived", column="Embarked", values="Ticket", aggfunc="count"))


    # Feature engineering
    train, test, all_data = combine_data(train_path, test_path)
    train, _, _, _, _ = process_cabin_feature(train)
    train, _, _, _, _ = process_ticket_feature(train)
    train, _ = process_name_feature(train)


    # Data processing 
    data = process_data_feature(all_data)
    data = process_null_values(data, train)
    data = normalize_values(data)
    dummies_data = encode_values(data)
    scaled_data = scale_data(dummies_data)
    X_train, X_test, y_train, y_test = split_data(dummies_data, data)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_data(scaled_data, data)


    # Model building
    gaussian_cv = evaluate_model(GaussianNB(), X_train_scaled, y_train)
    print("GaussianNB score: ", gaussian_cv)

    logistic_regression_cv = evaluate_model(LogisticRegression(max_iter=2000), X_train_scaled, y_train)
    print("Logistic regression score: ", logistic_regression_cv)

    decision_tree_cv = evaluate_model(tree.DecisionTreeClassifier(random_state=1), X_train_scaled, y_train)
    print("Decision tree score: ", decision_tree_cv)

    kneighbors_cv = evaluate_model(KNeighborsClassifier(), X_train_scaled, y_train)
    print("KNeighbors score", kneighbors_cv)

    svc_cv = evaluate_model(SVC(probability=True), X_train_scaled, y_train)
    print("SVC score", svc_cv)


    # Model tuning 
    svc = SVC(probability = True)
    param = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]

    best_model = perform_clf(svc, "SVC", param, X_train_scaled, y_train)


    # Survival prediction
    print(prediction(best_model, expected_features=X_train_scaled.columns.tolist()))



if __name__ == "__main__":
    main()