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

@app.route("/predict", methods=["Get"])
def predict():
    print(my_forest.learners)
    adjoe = request.args.get("ADJOE") # "" default value
    adjde = request.args.get("ADJDE")
    barthag = request.args.get("BARTHAG")
    efg_o = request.args.get("EFG_O")
    efg_d = request.args.get("EFG_D")
    tor = request.args.get("TOR")
    tord = request.args.get("TORD")
    drb = request.args.get("DRB")
    ftr = request.args.get("FTR")
    p_o = request.args.get("3P_O")
    p_d = request.args.get("3P_D")
    adj_t = request.args.get("ADJ_T")
    wab = request.args.get("WAB")
    seed = request.args.get("SEED")
    # predict?ADJOE=3&ADJDE=3&BARTHAG=8&EFG_O=4&EFG_D=2&_TOR=8&TORD=2&DRB=8&FTR=2&3p_O=1&3P_D=4&ADJ_T=5&WAB=6&SEED=9.0
    my_forest.X_test = [[int(adjoe), int(adjde), int(barthag), int(efg_o), int(efg_d), int(tor), int(tord), int(drb), int(ftr), int(p_o), int(p_d), int(adj_t), int(wab), int(float(seed))]]
    my_forest.X_test = [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1.0]]
    print(my_forest.X_test)
    my_forest.print_trees_rules()
    prediction = my_forest.predict()
    print(prediction, "test")
    # if prediction fails returns None
    if prediction is not None:
        result = {"prediction": prediction}
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
    my_forest.fit(X_train_bball, myutils.discretizeY(y_train_bball), 4, 2, 7, 0)
    print(my_forest.learners)
    # return "<h1>Hello catrld!</h1>"
    return render_template('predict.html')

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port, host="0.0.0.0") # TODO: make sure you turn off debug when you deploy