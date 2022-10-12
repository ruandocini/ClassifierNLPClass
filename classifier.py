import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from gensim.models import KeyedVectors
import numpy as np
from xgboost import XGBClassifier
# from transformers import AutoModel
# from transformers import AutoTokenizer
# import torch


def categorical_to_numerical(data: pd.DataFrame):
    data["req_type"] = data["req_type"].map(
        {"ambiente": 0, "industria": 1, "justica": 2}
    )
    return data


def categorical_to_numerical_multi_class(data: pd.DataFrame):
    data["req_type"] = data["req_type"].map(
        {
            "ambiente": np.array([1, 0, 0]),
            "industria": np.array([0, 1, 0]),
            "justica": np.array([0, 0, 1]),
        }
    )
    return data

def avg_document_vector(word2vec_model, doc):
    doc = [
        word for word in doc.lower().split() if word in word2vec_model.vocab
    ]
    if doc:
        return np.mean(word2vec_model[doc], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def avg_document_vector_bertimbau(sentences):
    model = AutoModel.from_pretrained("neuralmind/bert-large-portuguese-cased")
    tokenizer = AutoTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased", do_lower_case=False
    )

    input_ids = [
        tokenizer.encode(sentence, return_tensors="pt")
        for sentence in sentences
    ]

    with torch.no_grad():
        outs = [model(input_id) for input_id in input_ids]
        encoded = [out[0][0, 1:-1] for out in outs]
        final_representation = np.array(
            [np.mean(sent.cpu().detach().numpy(), axis=0) for sent in encoded]
        )
        return final_representation


def gridSearchLogisticRegression():
    solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    multi_class = ["auto", "ovr", "multinomial"]
    max_iter = np.arange(100, 1000, 100)
    grid = np.array(np.meshgrid(solvers, multi_class, max_iter)).T.reshape(
        -1, 2
    )
    return grid


def gridSearchXGBoost():
    max_depth = range(1, 10, 1)
    learning_rate = np.arange(0.1, 0.3, 0.1)
    n_estimators = range(50, 500, 50)
    min_child_weight = [1, 5, 10]
    gamma = [0.5, 1, 1.5, 2, 5]
    subsample = [0.6, 0.8, 1.0]
    colsample_bytree = [0.6, 0.8, 1.0]
    grid = np.array(
        np.meshgrid(
            max_depth,
            learning_rate,
            n_estimators,
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
        )
    ).T.reshape(-1, 7)
    return grid


def gridSearchMLP():
    hidden_layer_sizes = np.arange(10, 200, 10)
    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    grid = np.array(
        np.meshgrid(hidden_layer_sizes, activation, solver)
    ).T.reshape(-1, 3)
    return grid


def modelSelection(selectedModel: str):
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            multi_class="auto",
            solver="newton-cg",
            max_iter=1000,
        ),
        "MultiLayerPerceptron": MLPClassifier(
            hidden_layer_sizes=(100, 50, 50, 50, 50, 50, 50, 50, 5),
            max_iter=1000,
            activation="tanh",
            solver="adam",
            random_state=42,
            learning_rate="constant",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            min_child_weight=1,
            gamma=0.5,
            subsample=0.6,
            colsample_bytree=0.6,
        ),
    }

    return models[selectedModel]


raw_data = pd.read_excel("data/data.xlsx")

word2vec = KeyedVectors.load_word2vec_format("glove_s1000.txt")

X = pd.DataFrame(
    [avg_document_vector(word2vec, doc) for doc in raw_data["text"]]
)

Y = categorical_to_numerical(raw_data)["req_type"].values.reshape(1, -1)[0]

print(
    "Baseline: "
    + str(
        round(
            max(raw_data["req_type"].value_counts() / len(raw_data)) * 100,
            2,
        )
    )
    + "%"
)

finalClassifier = modelSelection("LogisticRegression")

result = cross_val_score(
    finalClassifier,
    X,
    Y,
    scoring="accuracy",
    cv=10,
    error_score="raise",
)

print("Logistic Classifier: " + str(round(result.mean() * 100, 2)) + "%")