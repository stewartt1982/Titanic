import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,accuracy_score 
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    Allows colomns to be selected by name from a pandas dataframe

    Parameters: list of str, names of the dataframe columns to select
    Default: []
    """
    def __init__(self, columns=[]):
        self.columns = columns
        
    """ Selects columns of a dataframe

    Parameters: X: pandas dataframe
    
    Returns: trans: pandas dataframe containing selected columns from X
    """
    def transform(self, X):
        trans = X[self.columns].copy()
        return trans

    """ Does nothing defined as it is needed
    Parameters: X : pandas dataframe
    y: default None

    Returns: self
    """
    def fit(self, X, y=None):
        return self

class MultiColumnLabelBinarizer:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        MyLabelBinarizer(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = MyLabelBinarizer().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = MyLabelBinarizer().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
))


    
def create_train_test(data=None,split_size=0.2):
    #default opt=1 setting uses train_test_split
    #if opt=2 use tatified sampling
    data_cp = data.copy()
    train_set, test_set = train_test_split(data_cp,test_size=split_size,random_state=42)
    #split = StratifiedShuffleSplit(n_splits=1,test_size=split_size,random_state=13)
    #for train_index_strat, test_index_strat in split.split(data_cp, data_cp["Sex"]):
    #    train_set = data_cp.loc[train_index_strat]
    #    test_set = data_cp.loc[test_index_strat]
    return train_set, test_set

def create_title(data):
    #extract title
    data_cp = data.copy()
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data_cp['Title'] = data_cp.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    #Grouped together as they are rare and high(er) status
    data_cp['Title'] = data_cp['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                             'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_cp['Title'] = data_cp['Title'].replace('Mlle', 'Miss')
    data_cp['Title'] = data_cp['Title'].replace('Ms', 'Miss')
    data_cp['Title'] = data_cp['Title'].replace('Mme', 'Mrs')
    return data_cp['Title']

def fillna_embarked(data):
    data_cp = data.copy()
    data_cp["Embarked2"] = data_cp["Embarked"].fillna('S')
    return data_cp["Embarked2"]

def create_total_relatives(data):
    data_cp = data.copy()
    data_cp['Family_size'] = data_cp["Parch"]+data_cp["SibSp"]+1
    return data_cp['Family_size']

def create_name_length(data):
    data_cp = data.copy()
    data_cp['Name_length'] = data_cp["Name"].apply(lambda x: len(x))
    return data_cp['Name_length']

def create_has_cabin(data):
    data_cp = data.copy()
    data_cp['Has_cabin'] = 1
    data_cp.loc[data_cp['Cabin'].isnull(),'Has_cabin'] =0
    return data_cp['Has_cabin']

    
class FindBestModel:
    def __init__(self):
        self.models = []
        self.params = []
        self.names = []
    def register_model_params(self,model,params,name):
        self.models.append(model)
        self.params.append(params)
        self.names.append(name)
    def evaluate_models(self,data,target):
        best_acc = 0
        best_model = None
        best_params = None
        model_results = []
        for model,param in zip(self.models,self.params):
            print model,param
            model_results.append(self.run_model(data,target,model,param))
        for param,score,best_estimator in model_results:
            acc=score
            if acc > best_acc:
                best_acc = acc
                best_model = best_estimator
                best_params = param
        #return the best bestimator, and a list of
        #all estimators
        return best_estimator,model_results
    
    def run_model(self,data,target,model,params):
        results = GridSearchCV(model, params, cv=10, scoring="accuracy", n_jobs=1)    
        results.fit(data,target)
        return [str(results.best_params_), results.best_score_, results.best_estimator_]


    
if __name__=='__main__':
    #Read in data
    data = pd.read_csv("input/train.csv")
    test_data = pd.read_csv("input/test.csv")
    
    #get training set and validation set
    train_set,validation_set = create_train_test(data=data,split_size=0.2)

    #feature creation
    datasets = [train_set,validation_set,test_data]

    num_data = ["Fare"]#,"Total_relatives","Parch","SibSp"]
    cat_num_data = ["Pclass","Has_cabin"]
    cat_data = ["Sex","Title","Embarked2"]
    for dataset in datasets:
        dataset['Embarked2'] = fillna_embarked(dataset)
        dataset['Family_size'] = create_total_relatives(dataset)
        dataset['Title'] = create_title(dataset)
        dataset['Name_length'] = create_name_length(dataset)
        dataset['Has_cabin'] = create_has_cabin(dataset)
        #dataset.drop("Name",axis=1,inplace=True)
        #dataset.drop("Ticket",axis=1,inplace=True)
        #dataset.drop("Cabin",axis=1,inplace=True)
        #dataset.drop("PassengerId",axis=1,inplace=True)
        #dataset.drop("Embarked",axis=1,inplace=True)
        dataset[num_data] = dataset[num_data].apply(lambda x: x.astype('float64'))
        dataset[cat_num_data] = dataset[cat_num_data].apply(lambda x: x.astype('category'))
        dataset[cat_data] = dataset[cat_data].apply(lambda x: x.astype('category'))
        
    #before doing a fit to predict the missing Age values we shall
    #drop the columns not needed, scale the data and one hot encode
    #categorical data
    no_transform_pipe = Pipeline([
        ('select_columns',SelectColumnsTransformer(columns=["Family_size","Parch","SibSp","Name_length"]))
        ])
    num_data_pipeline =  Pipeline([
        ('select_columns',SelectColumnsTransformer(columns=num_data)),
        ('imputer',Imputer(strategy='median')),
        ('log',FunctionTransformer(func=np.log1p)),
        ('std_scaler',StandardScaler())
        ])
    #Imputer for the test set which has one NaN in the Faare column. 3rd class pasenger
    cat_num_data_pipeline = Pipeline([
        ('select_columns',SelectColumnsTransformer(columns=cat_num_data)),
        ('cat_encoder', OneHotEncoder(sparse=False))
        ])
    cat_data_pipeline = Pipeline([
        ('select_columns',SelectColumnsTransformer(columns=cat_data)),
        ('stringindexer',StringIndexer()),
        ("onehot", OneHotEncoder(sparse=False))
        ])

    FullPipeline = FeatureUnion(transformer_list=[
        ('no_transform',no_transform_pipe),
        ("num_pipeline", num_data_pipeline),
        ("cat_num_pipeline", cat_num_data_pipeline),
        ("cat_pipeline", cat_data_pipeline)
    ])

    #Imputing age with a random forest
    #convert to pandas to make life easier for imputing Age
    pipe_out_train = pd.DataFrame(FullPipeline.fit_transform(datasets[0]))
    pipe_out_validation = pd.DataFrame(FullPipeline.transform(datasets[1]))
    pipe_out_test = pd.DataFrame(FullPipeline.transform(datasets[2]))
    
    #But we have still not dealt with the missing values in age
    tempRF = GridSearchCV(estimator=RandomForestRegressor(),
                          param_grid=[{"n_estimators": [10, 50, 100]}],
                          scoring="neg_mean_squared_error")
 
    #Add Age back to each dataframe
    pipe_out_train["Age"]=datasets[0]["Age"].values
    pipe_out_validation["Age"]=datasets[1]["Age"].values
    pipe_out_test["Age"]=datasets[2]["Age"].values
    
    #Get data with no nan
    #for training
    pipe_out_train_notnull  = pipe_out_train.loc[ (pipe_out_train["Age"].notnull()) ]# known Age values
    pipe_out_validation_notnull  = pipe_out_validation.loc[ (pipe_out_validation["Age"].notnull()) ]# known Age values
    pipe_out_test_notnull  = pipe_out_test.loc[ (pipe_out_test["Age"].notnull()) ]# known Age values
    #need to impute these values
    pipe_out_train_null = pipe_out_train.loc[ (pipe_out_train["Age"].isnull()) ]      # null Ages
    pipe_out_validation_null = pipe_out_validation.loc[ (pipe_out_validation["Age"].isnull()) ]      # null Ages
    pipe_out_test_null = pipe_out_test.loc[ (pipe_out_test["Age"].isnull()) ]      # null Ages
    #fit the model
    tempRF.fit(pipe_out_train_notnull.drop("Age",axis=1), pipe_out_train_notnull["Age"])
    RFreg = tempRF.best_estimator_
#    RFreg.fit(pipe_out_train_notnull.drop("Age",axis=1), pipe_out_train_notnull["Age"])
    #Now predict the missing values
    pred_train = RFreg.predict(pipe_out_train_null.drop("Age",axis=1))
    pred_validation =  RFreg.predict(pipe_out_validation_null.drop("Age",axis=1))
    pred_test =  RFreg.predict(pipe_out_test_null.drop("Age",axis=1))
    #now replace the Age nan's with the predicted values
    pipe_out_train.loc[ (pipe_out_train["Age"].isnull()), 'Age' ] = pred_train
    pipe_out_validation.loc[ (pipe_out_validation["Age"].isnull()), 'Age' ] = pred_validation
    pipe_out_test.loc[ (pipe_out_test["Age"].isnull()), 'Age' ] = pred_test

    #Now we must scale the Age variable to prepare for 
    scale_age_pipe = Pipeline([
        ('select_columns',SelectColumnsTransformer(columns=["Age"])),
        ('std_scaler',StandardScaler())
    ])
    
    age_scaled_train = pd.DataFrame(scale_age_pipe.fit_transform(pipe_out_train))
    age_scaled_validation = scale_age_pipe.transform(pipe_out_validation)
    age_scaled_test = scale_age_pipe.transform(pipe_out_test)
    pipe_out_train["Age"]=age_scaled_train
    pipe_out_validation["Age"]=age_scaled_validation
    pipe_out_test["Age"]=age_scaled_test


    #Now we have our data read to be feed to our algorithms
    #Let's try ensemble or stacking learning
    best_model = FindBestModel()
    best_model.register_model_params(KNeighborsClassifier(),[{"n_neighbors":[2,3,4,5,6,7,8,9,10]}],"KNN classifier Estimator")
    # best_model.register_model_params(LogisticRegression(),[{"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                                                         "penalty": ["l1","l2"]}],"Logistic Regression Estimator")
    # best_model.register_model_params(SVC(),[{"C": [0.1, 1, 10, 100, 1000],
    #                                          "kernel": ['linear']},
    #                                         {"C": [0.1, 1, 10, 100, 1000],"kernel":['rbf'],
    #                                          "gamma": [0.001,0.0001]}],"SVC Estimator")
    #best_model.register_model_params(RandomForestClassifier(),[{"n_estimators": [10, 25, 50, 100,500,2000],
    #                                                            "max_features": ['sqrt', 'log2'],
    #                                                            "max_depth": [2,4,6]}],"Random Forest classifier Estimator")
    # best_model.register_model_params(AdaBoostClassifier(),[{"n_estimators": [10, 50, 100, 200],
    #                                                         "learning_rate": [0.1,1.0,10]}],"AdaBoost classifier Estimator")
    # best_model.register_model_params(GradientBoostingClassifier(),[{"n_estimators": [10, 50, 100, 200],
    #                                                                 "learning_rate": [0.01,0.1,1.0]}],"GradientBoosting classifier Estimator")
    
    best_estimator,best_estimators = best_model.evaluate_models(pipe_out_train,datasets[0]["Survived"])
    for param,score,estimator in best_estimators:
        print param,score
        pred_train=estimator.predict(pipe_out_train)
        pred_val=estimator.predict(pipe_out_validation)
        print "Accuracy on test data: ",accuracy_score(datasets[0]["Survived"],pred_train)
        print "Accuracy on validation data: ",accuracy_score(datasets[1]["Survived"],pred_val)
    #GridSearchCV gives best parameter for
    #Parameters taken from GridSearchCV
    votingclass = VotingClassifier(
        estimators=[('knn', KNeighborsClassifier(n_neighbors = 5)),
                    ('lr', LogisticRegression(penalty="l2",C=1)),
                    ('rf', RandomForestClassifier(n_estimators=10,max_features="sqrt",max_depth=6)),
                    ('lsvc', SVC(kernel='linear',C=1.0,probability=True)),
                    ('ada',AdaBoostClassifier(n_estimators=200,learning_rate=0.1)),
                    ('gradboost',GradientBoostingClassifier(n_estimators=100,learning_rate=0.01))
        ], voting='soft')
    

    votingclass.fit(pipe_out_train,datasets[0]["Survived"])
    #now predict
    test_pred = votingclass.predict(pipe_out_train)
    test_val = votingclass.predict(pipe_out_validation)
    test_test = votingclass.predict(pipe_out_test)
    print "Accuracy on test data: ",accuracy_score(datasets[0]["Survived"],test_pred)
    print "Accuracy on validation data: ",accuracy_score(datasets[1]["Survived"],test_val)

    #Submission file
        
    submission = pd.DataFrame({
        "PassengerId": datasets[2]["PassengerId"],
        "Survived": test_test})
    
    submission.to_csv('submission.csv', index=False)
