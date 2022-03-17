import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


# Example dataframe from the sklearn docs
df = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw',
              'London', 'London', 'Paris', 'Sallisaw',
              'Seoul', 'Newyork', 'Sydney', 'Beijing',
              'Seoul', 'Newyork', 'Sydney', 'Beijing'],
     'title': ["His Last Bow", "How Watson Learned the Trick", "A Moveable Feast", "The Grapes of Wrath",
               "Harry Potter", "Up", "EA Sport Season Game", "The great game",
               "Last Emperor", "Scikit learn", "Pycharm is the best", "Google is the best",
               "Social Network", "Pytorch Machine learning", "Feast on the way", "Downsizing"],
     'expert_rating': [5, 3, 4, 5, 5, 3, 4, 5, 5, 3, 4, 5, 5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3],
     'click': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']})
y = df.pop("click")
X = df

# Build a Column transformer
categorical_preproc = OneHotEncoder()
text_preproc = TfidfVectorizer()
numerical_preprocessing = make_pipeline(SimpleImputer(), StandardScaler())

preproc = ColumnTransformer(
    transformers=[
        ("cat_transform", categorical_preproc, ['city']),
        ("text_transform", text_preproc, 'title'),
        ("num_transform", numerical_preprocessing, ['expert_rating', 'user_rating'])
    ],
    remainder='drop'
)
pipe = make_pipeline(preproc, LogisticRegression())

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)

# train
pipe.fit(X, y)

# inference
pred = pipe.predict(X_test)
pred_proba = pipe.predict_proba(X_test)

print("Prediction result:")
print(f"class: {pred}")
print(f"proba: {pred_proba}")