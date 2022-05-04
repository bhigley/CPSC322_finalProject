import importlib
from tokenize import Double

import mysklearn.myutils
import mysklearn.myutils as myutils

import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import numpy as np
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyRandomForestClassifier
import mysklearn.myevaluation

import os
import pickle
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
my_forest = MyRandomForestClassifier()

team_values = [[10, 2, 10, 10, 1, 4, 3, 4, 4, 6, 3, 10, 9, 1.0], \
[10, 4, 10, 7, 4, 6, 4, 8, 6, 4, 3, 7, 10, 1.0], [9, 3, 10, 6, 5, 6, 8, 8, 4, 4, 2, 6, 10, 1.0], \
[9, 4, 10, 8, 2, 6, 4, 8, 7, 4, 5, 10, 10, 1.0], [6, 4, 9, 6, 4, 6, 4, 4, 4, 4, 1, 7, 6, 10.0], \
[5, 4, 9, 8, 4, 6, 6, 5, 6, 6, 4, 3, 6, 10.0], [8, 8, 8, 8, 7, 3, 3, 3, 6, 6, 6, 3, 6, 10.0], \
[7, 8, 8, 7, 10, 2, 6, 9, 4, 4, 8, 6, 6, 10.0], [8, 5, 9, 7, 6, 5, 5, 6, 1, 7, 4, 1, 5, 11.0], \
[8, 6, 9, 5, 7, 5, 1, 4, 4, 3, 6, 4, 5, 11.0], [6, 6, 8, 7, 6, 3, 1, 5, 3, 6, 4, 4, 6, 11.0], \
[3, 3, 8, 4, 6, 8, 10, 8, 3, 2, 2, 4, 6, 11.0], [4, 5, 7, 4, 5, 5, 4, 7, 3, 3, 7, 3, 5, 11.0], \
[5, 3, 9, 5, 3, 5, 3, 5, 6, 3, 5, 5, 5, 12.0], [6, 7, 8, 6, 5, 4, 7, 7, 4, 6, 5, 7, 5, 12.0], \
[5, 6, 8, 5, 4, 4, 2, 5, 8, 2, 3, 4, 6, 12.0], [4, 6, 7, 5, 3, 8, 3, 6, 8, 2, 3, 4, 6, 12.0], \
[5, 7, 7, 5, 8, 2, 5, 7, 5, 3, 7, 5, 4, 12.0], [6, 6, 8, 9, 5, 3, 3, 1, 2, 5, 7, 3, 5, 13.0], \
[8, 9, 7, 10, 9, 3, 2, 5, 7, 10, 8, 8, 6, 13.0], \
[5, 7, 7, 6, 7, 5, 4, 6, 3, 3, 3, 3, 5, 13.0], [4, 8, 4, 6, 6, 5, 4, 6, 10, 4, 7, 1, 3, 13.0], \
[4, 9, 5, 8, 6, 5, 2, 5, 3, 8, 6, 5, 2, 14.0], [4, 9, 4, 5, 8, 6, 6, 5, 8, 6, 4, 4, 4, 14.0], \
[3, 8, 4, 7, 6, 6, 4, 4, 10, 5, 5, 5, 4, 14.0], [2, 6, 4, 4, 6, 6, 4, 7, 6, 2, 3, 7, 2, 14.0], \
[1, 5, 5, 2, 1, 8, 7, 8, 9, 4, 2, 4, 2, 15.0], [4, 9, 4, 6, 6, 6, 4, 10, 7, 4, 7, 5, 2, 15.0], \
[4, 8, 4, 7, 5, 8, 3, 6, 6, 6, 5, 3, 2, 15.0], [3, 9, 3, 3, 8, 6, 6, 8, 8, 3, 10, 4, 2, 15.0], \
[2, 6, 4, 1, 7, 5, 8, 10, 5, 2, 10, 5, 2, 16.0], [3, 9, 3, 4, 5, 7, 3, 10, 6, 1, 5, 10, 2, 16.0], \
[2, 8, 3, 5, 2, 8, 4, 7, 10, 4, 2, 5, 3, 16.0], [1, 7, 2, 3, 2, 9, 3, 8, 6, 2, 2, 6, 1, 16.0], \
[4, 10, 2, 5, 8, 6, 4, 9, 5, 2, 6, 7, 1, 16.0], [1, 9, 1, 2, 6, 8, 8, 8, 9, 1, 4, 7, 1, 16.0], \
[9, 5, 10, 6, 4, 4, 3, 5, 3, 4, 3, 6, 9, 2.0], [10, 5, 10, 8, 4, 3, 2, 8, 4, 5, 4, 6, 9, 2.0], \
[9, 4, 10, 5, 5, 3, 5, 8, 5, 5, 4, 1, 9, 2.0], [7, 3, 10, 4, 1, 4, 6, 9, 5, 2, 4, 8, 9, 2.0], \
[5, 1, 10, 5, 3, 8, 9, 6, 8, 1, 4, 5, 8, 3.0], [6, 1, 10, 4, 3, 6, 8, 7, 4, 4, 4, 5, 9, 3.0], \
[10, 6, 10, 9, 7, 5, 1, 4, 8, 7, 6, 4, 9, 3.0], \
[6, 4, 9, 3, 7, 1, 3, 5, 6, 1, 5, 5, 9, 3.0], [8, 3, 10, 4, 5, 1, 5, 5, 4, 4, 5, 4, 8, 4.0], \
[7, 4, 9, 6, 4, 5, 2, 6, 5, 5, 5, 5, 8, 4.0], [6, 3, 9, 3, 5, 5, 6, 6, 9, 1, 5, 9, 8, 4.0], \
[6, 6, 8, 4, 4, 5, 2, 8, 9, 3, 4, 4, 8, 4.0], [8, 2, 10, 6, 1, 5, 7, 7, 4, 3, 2, 2, 8, 5.0], \
[10, 6, 10, 6, 7, 1, 5, 9, 5, 5, 5, 8, 8, 5.0], [5, 3, 9, 6, 5, 5, 5, 3, 1, 4, 7, 2, 7, 5.0], \
[7, 5, 9, 4, 3, 5, 4, 7, 5, 4, 7, 3, 7, 5.0], [6, 3, 9, 4, 4, 6, 9, 9, 6, 2, 4, 2, 7, 6.0], \
[4, 2, 9, 4, 2, 8, 10, 10, 6, 2, 1, 7, 6, 6.0], [8, 7, 9, 5, 7, 7, 4, 10, 7, 1, 6, 9, 6, 6.0], \
[7, 6, 8, 7, 7, 3, 4, 5, 5, 4, 5, 4, 8, 6.0], [8, 7, 9, 7, 5, 4, 2, 8, 6, 6, 7, 3, 6, 7.0], \
[6, 5, 8, 5, 5, 7, 2, 7, 5, 6, 4, 6, 7, 7.0], [6, 5, 8, 6, 5, 5, 7, 6, 5, 4, 3, 4, 7, 7.0], \
[5, 6, 8, 5, 2, 5, 1, 7, 5, 4, 6, 4, 7, 7.0], [3, 1, 9, 3, 1, 7, 7, 7, 5, 4, 3, 4, 7, 8.0], \
[5, 4, 9, 5, 5, 6, 6, 3, 7, 3, 5, 3, 7, 8.0], \
[7, 6, 9, 5, 7, 4, 1, 2, 4, 5, 7, 8, 7, 8.0], [5, 4, 8, 2, 3, 5, 4, 9, 5, 3, 4, 6, 6, 8.0], \
[6, 4, 9, 6, 4, 10, 7, 10, 9, 5, 5, 8, 6, 9.0], [5, 4, 8, 3, 5, 9, 4, 6, 5, 1, 4, 4, 6, 9.0], \
[5, 5, 8, 6, 4, 5, 5, 10, 3, 4, 4, 9, 5, 9.0], [3, 3, 8, 4, 2, 8, 2, 8, 2, 1, 4, 5, 6, 9.0]]

@app.route("/predict", methods=["Get"])
def predict():
    team = request.args.get("team", "")
    my_forest.X_test = [team_values[int(team)]]
    prediction = my_forest.predict()
    # if prediction fails returns None
    if prediction is not None:
        final = ""
        if prediction[0] == 0:
            if np.random.randint(0, 9) < 3:
                final = "Round of 32"
            else:
                final = "Round of 64"
        elif prediction[0] == 1:
            final = "Elite 8"
        else:
            final = "Final 4"
        result = {"prediction": final}
        return jsonify(result), 200
    return "Error making prediction", 400

@app.route("/", methods=["Get"])
def home():
    fname = os.path.join("input_data", "cbb.csv")
    bball_table = MyPyTable()
    bball_table.load_from_file(fname)

    stats_header = ['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
    'DRB','FTR','3P_O','3P_D','ADJ_T','WAB']
    stats_cols = []
    stats_cols_inner = []
    stats_col = []
    # Grabbing all the rows we want to use
    for stat in stats_header:
        stats_col.append(myutils.discretize(myutils.normalize(bball_table.get_column(stat))))
    stats_col.append(bball_table.get_column('SEED'))

    # Creating a new table with the rows based on the appropriate columns
    for index in range(len(bball_table.data)):
        for stat_col in stats_col:
            stats_cols_inner.append(stat_col[index])
        stats_cols.append(stats_cols_inner)
        stats_cols_inner = []

    y_train_bball = [val for val in bball_table.get_column('POSTSEASON')]
    X_train_bball = stats_cols.copy()
    my_forest.fit(X_train_bball, myutils.discretizeY(y_train_bball), 20, 3, 7)
    return render_template('predict.html')


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, port=port, host="0.0.0.0") # TODO: make sure you turn off debug when you deploy