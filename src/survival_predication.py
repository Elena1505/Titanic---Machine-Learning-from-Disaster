import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


def prediction(model, expected_features):
    passengerid = input("Entrez votre id passager ")
    pclass = input("Entrez votre classe (1, 2 ou 3) ")
    name = input("Entrez votre nom (de la forme : nom de famille, Mr ou Mrs prénom) ")
    sex = input("Entrez votre sex (male ou female) ")
    age = input("Entrez votre age ")
    sibsp = input("Entrez le nombre de vos frèes et soeurs à bord ")
    parch = input("Entrez le nombre de vos parents et enfants à bord ")
    ticket = input("Entrez votre numero de ticket (numéro à 6 chiffres) ")
    fare = input("Entrez le tarif payé pour le billet ")
    cabin = input("Entrez votre ou vos cabines (une lettre en majuscule suivie de 2 à 3 chiffres) ")
    embarked = input("Entrez votre porte d'embarquement (C, Q ou S) ")

    # Création du DataFrame
    passenger_data = {
        'PassengerId': passengerid,
        'Pclass': pclass,
        'Name': name,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': ticket,
        'Fare': fare,
        'Cabin': cabin,
        'Embarked': embarked
    }
    passenger_data = pd.DataFrame([passenger_data])

    # Transformation des données
    passenger_data['Cabin_multiple'] = passenger_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    passenger_data['Cabin_adv'] = passenger_data.Cabin.apply(lambda x: str(x)[0])
    passenger_data['Numeric_ticket'] = passenger_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    passenger_data['Ticket_letters'] = passenger_data.Ticket.apply(
        lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
    passenger_data['Name_title'] = passenger_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    passenger_data['Fare'] = pd.to_numeric(passenger_data['Fare'], errors='coerce')
    passenger_data['Norm_fare'] = np.log(passenger_data.Fare + 1)
    passenger_data.Pclass = passenger_data.Pclass.astype(str)

    # Encodage avec get_dummies
    dummies_data = pd.get_dummies(passenger_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Norm_fare',
                                                  'Embarked', 'Cabin_adv', 'Cabin_multiple', 'Numeric_ticket',
                                                  'Name_title']])

    # Ajouter les colonnes manquantes
    for col in expected_features:
        if col not in dummies_data.columns:
            dummies_data[col] = 0

    # Assurer l'ordre des colonnes
    dummies_data = dummies_data[expected_features]

    # Standardisation
    scale = StandardScaler()
    dummies_data[['Age', 'SibSp', 'Parch', 'Norm_fare']] = scale.fit_transform(
        dummies_data[['Age', 'SibSp', 'Parch', 'Norm_fare']])

    # Prédiction
    predicted_survival = model.predict(dummies_data)
    predicted_probability = model.predict_proba(dummies_data)
    print(f"Le passager a une probabilité de survie de : {predicted_probability[0][1]:.2f}")
    print(f"Prédiction de survie : {'Survécu' if predicted_survival[0] == 1 else 'Décédé'}")
