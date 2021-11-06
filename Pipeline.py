import numpy as np
import warnings

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import MeanMedianImputer
from Feature_engineering import FeatureEngPipe, dtype_fix, get_feature_name, Cabin_code

#VARIABLES
warnings.filterwarnings("ignore")
seed = 42

#### algorithyms
lr = LogisticRegression(random_state=seed)
rf = RandomForestClassifier(random_state=seed)
gbc = GradientBoostingClassifier(random_state=seed)

#pipeline for columns transformations on features
cat_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan,
                                                 strategy='most_frequent'),
                                    OneHotEncoder(handle_unknown='ignore')) #Só vai fazer no test data o q já fez no train ou em inf no test)
num_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan,
                                                 strategy='median'))
pipe_preprocessing = ColumnTransformer( [("numeric_transf",
                                          num_preprocessing,
                                          make_column_selector(dtype_exclude=object)),    # NOME-PROCESSSO  $$$$$ TRANFORMACAO A SER APLCIADA $$$$$ COLUNAS QUE VAO SOFRER A TRANF.
                                        ("categorical_transf",
                                         cat_preprocessing,
                                         make_column_selector(dtype_include=object))])


vt_final_sklearn = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprocessing),
                          ("Voting", VotingClassifier(voting='hard',
                                                      estimators=[('lr', lr),
                                                                  ('rf', rf),
                                                                  ('gbc', gbc)]))
                      ])

# pipeline for imputing missing values
impute_pipe = Pipeline([
   ('Fix dtypes',dtype_fix()),
   ('Cabin imputer',CategoricalImputer(variables=['Cabin'], fill_value='C')),
   ('Embarked imputer',CategoricalImputer(variables=['Embarked'], imputation_method='frequent')),
   ('Numeric imputer', MeanMedianImputer(variables=['Age','Fare'],imputation_method='median'))
                    ])
# pipeline for extracting features
ext_pp = make_pipeline(get_feature_name(),
                        Cabin_code())
# OneHotEncode categorical features
cat_pp = OneHotEncoderFE(variables=['Pclass',
                                    'Embarked',
                                    'Name_title',
                                    'Name_family',
                                    'Cabin_code'],
                         drop_last=False)

# final pipeline
vt_final_fe = Pipeline([
    ('Impute missing values',impute_pipe),
    ('Extract features from Name and Cabin',ext_pp),
    ('Drop bad features',DropFeatures(features_to_drop=['Name',
                                                        'Cabin',
                                                        'Ticket'])),
    ('Drop numeric features',DropFeatures(features_to_drop=['Cabin_Size',
                                                            'Age',
                                                            'Parch',
                                                            'SibSp',
                                                            'Sex',
                                                            'Fare'])),
    ('Categorical Preprocess', cat_pp),
    ('Voting Classifier Estimator',VotingClassifier(voting='hard',
                                                    estimators=[('lr', lr),
                                                                ('rf', rf),
                                                                ('gbc', gbc)]))
    ])