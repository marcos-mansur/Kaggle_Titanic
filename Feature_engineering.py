# transformadores customizados para feature engineering e pré processamento

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class get_feature_name(BaseEstimator, TransformerMixin):
    """create column with family name  and title of each passenger

    title: bool => if True, create a column with the title extracted from Name
    family: bool => if True, create a column with the family name extracted from Name

    """

    def __init__(self, title: bool = True, family: bool = True):
        self.title = title
        self.family = family
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        df = x.copy()
        try:
            if self.title:
                df['Name_title'] = df['Name'].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split(".")[0])

            if self.family:
                df['Name_family'] = df['Name'].apply(lambda x: x.split(",")[0])

        except:
            print('erro em get_name')
        return df


class dtype_fix(BaseEstimator, TransformerMixin):
    """ corrects dtype of all initial features

    Pclass_type: bool => if True, transforms the column Pclass in dtype 'object'

    """

    def __init__(self, Pclass_type=True):
        self.Pclass_type = Pclass_type
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        df = x.copy()
        try:
            if self.Pclass_type:
                df['Pclass'] = df['Pclass'].astype('O')
            df['Fare'] = df['Fare'].astype('float')
            df['Sex'] = df['Sex'].astype('object')
            df['Age'] = df['Age'].astype('float')
            df['Cabin'] = df['Cabin'].astype('object')
            df['Embarked'] = df['Embarked'].astype('object')
        except:
            print('erro em dtype_fix')
        return df


class Mapper(BaseEstimator, TransformerMixin):
    """Create column with first character of the first Cabin and how many cabins

    features: list => list of features to apply mapping
    map_dicts: list => list of dictionarys with mappings to transform the features, with the same order

    """

    def __init__(self, features: list = ['Sex'], map_dicts=[{'male': 1, 'female': 0}]):

        if not isinstance(map_dicts, list):
            raise ValueError('map_dicts should be a list of dictionarys')

        self.features = features
        self.map_dicts = map_dicts
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame):
        x = x.copy()

        for i, feature in enumerate(self.features):
            x[feature] = x[feature].map(self.map_dicts[i])
        return x


class Cabin_code(BaseEstimator, TransformerMixin):
    """Create column with first character of the first Cabin and how many cabins

        code: bool => if True, create a column with the cabin first letter extracted from Cabin
        size: bool => if True, create a column with the number of cabins extracted from Cabin

    """

    def __init__(self, code=True, size=True):
        self.code = code
        self.size = size
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x = x.copy()
        try:
            if self.code:
                x['Cabin_code'] = x['Cabin'].apply(lambda x: x[0])
            if self.size:
                x['Cabin_Size'] = x['Cabin'].apply(lambda x: len(x.split(" ")))
        except Exception as err:
            print('erro no Cabin_code')
            print(err)
        return x

#====================== Funções do Thiago =======================

def preprosseging_dtypes(df2):
    df = df2.copy()
    df.Pclass = df.Pclass.astype("object")
    return df

def name_information(df2):
    df = df2.copy()
    df["Name_aux"] = df["Name"]
    df["Name_aux"] = df["Name_aux"].str.replace(",|\.|\(|\)|\"|Mrs|Mr|Miss", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]\W", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]$", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("  ", " ", regex=True)
    df["Name_List"] = df["Name_aux"].apply(lambda x: x.split(" "))
    df["Last_Name"] = df["Name_List"].apply(lambda x: x[-1])
    df.drop(["Name","Name_List"], axis=1, inplace=True)
    return df

def cabin_information(df2):
    df = df2.copy()
    df['Cabin'].fillna("S",inplace=True)
    df['Category_Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Size_Cabin'] = df['Cabin'].apply(lambda x: len(x.split(" ")))
    df.drop("Cabin",axis=1,inplace=True)
    return df
def ticket_information(df2):
    df = df2.copy()
    #df.drop("Ticket",axis=1,inplace=True)
    return df



class FeatureEngPipe(BaseEstimator):

    def __init__(self,name=True,cabin=True,ticket=True, preprop=True):
        self.name = name
        self.cabin = cabin
        self.ticket = ticket
        self.preprop = preprop
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        if self.preprop:
            x_dataset= preprosseging_dtypes(x_dataset)

        if self.name:
            x_dataset = name_information(x_dataset)

        if self.cabin:
            x_dataset = cabin_information(x_dataset)

        if self.ticket:
            x_dataset = ticket_information(x_dataset)
        return x_dataset