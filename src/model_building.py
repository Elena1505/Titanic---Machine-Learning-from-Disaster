from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train):
    cv = cross_val_score(model, X_train, y_train, cv=5)
    return cv.mean()