from curses import raw
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def categorical_to_numerical(data: pd.DataFrame):
    data["req_type"] = data["req_type"].map(
        {"ambiente": 1, "industria": 2, "justica": 3}
    )
    return data


raw_data = pd.read_excel("data/data.xlsx")

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigrams = bigram_vectorizer.fit_transform(raw_data["text"]).toarray()

X = pd.DataFrame(bigrams, columns=bigram_vectorizer.get_feature_names_out())

Y = categorical_to_numerical(raw_data)["req_type"].values.reshape(1, -1)[0]

logistic_classifier = LogisticRegression(random_state=42, multi_class="auto")

print(
    "Baseline: "
    + str(
        round(
            max(raw_data["req_type"].value_counts() / len(raw_data)) * 100, 2
        )
    )
    + "%"
)

result = cross_val_score(
    logistic_classifier, X, Y, scoring="accuracy", cv=10
).mean()

print("Logistic Classifier: " + str(round(result * 100, 2)) + "%")
