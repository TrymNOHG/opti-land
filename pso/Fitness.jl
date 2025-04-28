using ScikitLearn

module Fitness

function h_p(individual::BitArray)
    return sum(individual)
end

function get_fitness(column_vec::BitArray, X_train, X_test, y_train, y_test; epsilon=0.0)
    # drop_cols = []
    # for i, col in enumerate(X_train.columns):
    #     if individual[i] == 0:
    #         drop_cols.append(col)

    avg_accuracy = 0
    for _ in 1:30
        model = RandomForestClassifier(n_estimators=30, max_depth=None, criterion="gini", max_features=None, random_state=456)
        

        new_X_train = X_train.drop(columns=drop_cols)
        new_X_test = X_test.drop(columns=drop_cols)

        model = model.fit(new_X_train, y_train)

        y_pred = model.predict(new_X_test)

        avg_accuracy += accuracy_score(y_test, y_pred)
    end
    avg_accuracy /= 30
    classification_error = 1 - avg_accuracy
    fitness = classification_error + epsilon * h_p(column_vec)
    return fitness
    
end

end