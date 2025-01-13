from src.data_exploration import describe, describe_numerical_data, describe_categorical_data, compare_data

def main(): 
    raw_path = "/home/lelou1505/DS/Titanic/data/raw/train.csv"
    figure_path = "/home/lelou1505/DS/Titanic/reports/figures"
    print(describe(raw_path))
    describe_numerical_data(raw_path, ['Age','SibSp','Parch','Fare'], figure_path)
    describe_categorical_data(raw_path, ['Survived','Pclass','Sex','Ticket','Cabin','Embarked'], figure_path)
    print(compare_data(raw_path, "Survived", column="Pclass", values="Ticket", aggfunc="count"))
    print(compare_data(raw_path, "Survived", column="Sex", values="Ticket", aggfunc="count"))
    print(compare_data(raw_path, "Survived", column="Embarked", values="Ticket", aggfunc="count"))


if __name__ == "__main__":
    main()