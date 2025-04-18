X = one_hot_df[features]
y = one_hot_df["Q13_Part_10"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2)
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("Classifier Accuracy: ", accuracy_score(y_pred, y_test))