from sklearn.model_selection import GridSearchCV

def perform_clf(classifier, model_name, param, X_train, y_train):
    clf_svc = GridSearchCV(classifier, param_grid = param, cv = 5, verbose = True, n_jobs = -1)
    best_clf_svc = classifier.fit(X_train,y_train)
    print(model_name)
    print('Best Score: ' + str(best_clf_svc.best_score_))
    print('Best Parameters: ' + str(best_clf_svc.best_params_))
    return best_clf_svc