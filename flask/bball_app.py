import os
import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello World!</h1>"

@app.route("/predict", methods=["Get"])
def predict():
    level = request.args.get("level") # "" default value
    lang = request.args.get("lang")
    tweets = request.args.get("Tweets")
    phd = request.args.get("phd")
    # print("level", level, lang, tweets, phd)
    # TODO: fix the hardcoding
    prediction = predict_interview_well([level, lang, tweets, phd])
    # prediction = predict_interview_well(["junior", "Java", "yes", "no"])
    # # if anything goes wrong, predict_interview_well returns none
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def predict_interview_well(instance):
    # traverse interview tree
    # make a prediction for instance
    # how do we get the tree here?
    # save a trained ML mode from another process for use later
    # enter pickling
    # unpicle tree.p
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    header[2] = "Tweets"
    print("tree:", tree)
    infile.close()

    # print("header:", header)
    # print("tree:", tree)
    # prediction time!!
    try:
        prediction = tdidt_predict(header, tree, instance)
        return prediction
    except:
        print("error")
        return None

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    # print(info_type)
    if info_type == "Leaf":
        return tree[1] # label
    # we are at an attribute
    # find attribute value match for instance
    # for loop
    # print("Ben")
    print(header, tree[1])
    att_index = header.index(tree[1])
    # print("cat")
    # print(att_index)
    for i in range(2, len(tree)):
        # print("dog")
        value_list = tree[i]
        print("cat", value_list[1], instance[att_index])
        if value_list[1] == instance[att_index]:
            print("test", value_list[1], instance[att_index])
            # we have a match, recurse
            prediction = tdidt_predict(header, value_list[2], instance)
            return prediction

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port, host="0.0.0.0") # TODO: make sure you turn off debug when you deploy
    # prediction = predict_interview_well(["Junior", "java", "yes", "no"])
    # print(prediction)