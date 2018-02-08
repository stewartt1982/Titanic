# Imports needed for the script
import numpy as np
import pandas as pd
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

#extact title info, like Mr., Ms., etc
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""


# Define function to calculate Gini Impurity
def get_gini_impurity(survived_count, total_count):
    survival_prob = survived_count/total_count
    not_survival_prob = (1 - survival_prob)
    random_observation_survived_prob = survival_prob
    random_observation_not_survived_prob = (1 - random_observation_survived_prob)
    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob
    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob
    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob
    return gini_impurity

# Loading the data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

# Showing overview of the train dataset
#print train.head(5)

#copy the data frame and we will modify the new one
original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values
original_test = test.copy()

#The data engineering gould be done to both the train and test dta sets
full_data=[train,test]

#covert the Cabin variable to a binary 'Has_cabin' variable
#NaN is a float

#train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
#test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

#group SibSp and Parch together into family size
for dataset in full_data:
    #deal with the cabin information
    #Can we be more clever with cabin?
    #Replace those with no cabin with a placeholder U0 (undefined 0)
    #We will extract deck letter and cabin number  Encode cabin number
    #as either fore, aft or middle
    #For U0 use the average value for each class
    dataset['Cabin'] = dataset['Cabin'].fillna('U0')
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    #Cabin letters
    #A,B,C,D,E,F,G,U for undefined
    dataset['Deck'] = dataset["Deck"].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}).astype(int)
    #No assign those with Deck == 0 to the average for their class
    dataset.loc[(dataset['Deck'] == 0),'Deck'] = np.nan
    dataset['Deck'] = dataset['Deck'].fillna(dataset.groupby('Pclass')['Deck'].transform('mean')).astype(int)
    
    dataset["FamilySize"] = dataset["SibSp"]+dataset["Parch"] + 1
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1,"IsAlone"] = 1

    #now deal with missing data
    dataset["Embarked"] = dataset["Embarked"].fillna('S')
    #Let us predict fare based on class, as fares will vary depending on
    #1st,2nd or 3rd class
    dataset['Fare'] = dataset['Fare'].fillna(dataset.groupby('Pclass')['Fare'].transform('median'))
    

    #Nulls in the age
    #get the mean age, and the std. deviation and randomly assign
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby('Pclass')['Age'].transform('mean'))
    # avg_age = dataset['Age'].mean()
    # avg_age_std = dataset['Age'].std()
    # age_null_count = dataset['Age'].isnull().sum()
    # age_null_random_list = np.random.randint(avg_age - avg_age_std,avg_age+avg_age_std,size = age_null_count)
    # #relace the NaNs with the random list
    # dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset["Age"] = dataset["Age"].astype(int)

    #encode sex
    dataset["Sex"] = dataset["Sex"].map({'female':0, 'male':1}).astype(int)

    #create title data
    dataset["Title"] = dataset["Name"].apply(get_title)

    #None common title like Lord, Lady get put into a single title 'Rare'
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Dona','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

    #Convert Rare into FemaleRare and MaleRare
    dataset.loc[(dataset['Sex'] == 0) & (dataset['Title'] == 'Rare'),'Title'] = 'FemaleRare'
    dataset.loc[(dataset['Sex'] == 1) & (dataset['Title'] == 'Rare'),'Title'] = 'MaleRare'
    
    #Convert less common representations of titles with Miss, or Mrs
    dataset["Title"] = dataset["Title"].replace("Mlle","Miss")
    dataset["Title"] = dataset["Title"].replace("Ms","Miss")
    dataset["Title"] = dataset["Title"].replace("Mme","Mrs")

    #Now do the men and women, convert to numeric
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "MaleRare": 5, "FemaleRare": 6}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    #encode the embarkation as a number
    dataset['Embarked'] = dataset["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)

    # #convert Fare into 4 categories
    # dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']         = 0
    # dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    # dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    # dataset.loc[ dataset['Fare'] > 31, 'Fare']         = 3
    # dataset['Fare'] = dataset['Fare'].astype(int)

    # # Mapping Age
    # dataset.loc[ dataset['Age'] <= 16, 'Age']        = 0
    # dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    # dataset.loc[ dataset['Age'] > 64, 'Age'] ;

# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)


#check that everything is numeric
#print train.head(32)
#

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


# Gini Impurity of starting node
gini_impurity_starting_node = get_gini_impurity(342, 891)
#print gini_impurity_starting_node


##################
#train a model in which hyperparameters also need to be optimized at the sme time
#
#Using a RandomForestClassifier
#
#
# # Number of random trials
# NUM_TRIALS = 10

# # Set up possible values of parameters to optimize over
# # Use a grid over parameters of interest
# param_grid = {
#     "n_estimators" : [10, 60, 110, 200, 300, 500],
#     "max_depth" : [1, 5, 10, 15, 20, 25, 30]}

# # Arrays to store scores
# non_nested_scores = np.zeros(NUM_TRIALS)
# nested_scores = np.zeros(NUM_TRIALS)

# #create the random forest classifier
# # RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
# rfc = ensemble.RandomForestClassifier(n_jobs=-1) 

# # Loop for each trial
# for i in range(NUM_TRIALS):
#     print "Trial i:",i
#     # Choose cross-validation techniques for the inner and outer loops,
#     # independently of the dataset.
#     # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
#     inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
#     outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    
#     # Non_nested parameter search and scoring
#     clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=inner_cv)
#     clf.fit(X = train.drop(['Survived'],axis=1), y = train['Survived'])
#     non_nested_scores[i] = clf.best_score_
#     # Nested CV with parameter optimization
#     nested_score = cross_val_score(clf, X=train.drop(['Survived'],axis=1), y=train['Survived'], cv=outer_cv)
#     nested_scores[i] = nested_score.mean()
#     print non_nested_scores[i],nested_scores[i]
#     print clf.best_params_
    
# score_difference = non_nested_scores - nested_scores
    
# print("Average difference of {0:6f} with std. dev. of {1:6f}."
#       .format(score_difference.mean(), score_difference.std()))

############
rffinal = ensemble.RandomForestClassifier(n_jobs=-1,n_estimators=200,max_depth=10)
rffinal.fit(X = train.drop(['Survived'],axis=1), y = train['Survived'])

prediction = rffinal.predict(test)

fobj = open("submission.csv", 'w')
fobj.write("PassengerId,Survived\n")
for passid,survived in zip(PassengerId.values,prediction.astype(int)):
    fobj.write('{},{}\n'.format(passid,survived))
    #fobj.write"%s,%i"%(jobid,salary)
fobj.close()


# cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
# accuracies = list()
# max_attributes = len(list(test))
# depth_range = range(1, max_attributes + 1)

# # Testing max_depths from 1 to max attributes
# # Uncomment prints for details about each Cross Validation pass

# for depth in depth_range:
#     fold_accuracy = []
#     tree_model = tree.DecisionTreeClassifier(max_depth = depth)
#     # print("Current max depth: ", depth, "\n")
#     for train_fold, valid_fold in cv.split(train):
#         f_train = train.loc[train_fold] # Extract train data with cv indices
#         f_valid = train.loc[valid_fold] # Extract valid data with cv indices
        
#         model = tree_model.fit(X = f_train.drop(['Survived'], axis=1),
#                                y = f_train["Survived"])
#         # We fit the model with the fold train data
#         valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1),
#                                 y = f_valid["Survived"])
#         # We calculate accuracy with the fold validation data
#         fold_accuracy.append(valid_acc)
        
        
#     avg = sum(fold_accuracy)/len(fold_accuracy)
#     accuracies.append(avg)
        
# df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
# df = df[["Max Depth", "Average Accuracy"]]
# print(df.to_string(index=False))



# # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
# y_train = train['Survived']
# x_train = train.drop(['Survived'], axis=1).values
# x_test = test.values

# # Create Decision Tree with max_depth = 3
# decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
# decision_tree.fit(x_train, y_train)

# # Predicting results for test dataset
# y_pred = decision_tree.predict(x_test)
# submission = pd.DataFrame({
#     "PassengerId": PassengerId,
#     "Survived": y_pred})
# submission.to_csv('submission.csv', index=False)


# # Export our trained model as a .dot file
# # with open("tree1.dot", 'w') as f:
# #     f = tree.export_graphviz(decision_tree,
# #                              out_file=f,
# #                              max_depth=3,
# #                              impurity=True,
# #                              feature_names = list(train.drop(['Survived'], axis=1)),
# #                              class_names = ['Died', 'Survived'],
# #                              rounded = True,
# #                              filled = True)
    
# # #Convert .dot to .png to allow display in web notebook
# # check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# # # Annotating chart with PIL
# # img = Image.open("tree1.png")
# # draw = ImageDraw.Draw(img)
# # font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
# # draw.text((10, 0), # Drawing offset (position)
# #           '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
# #           (0,0,255), # RGB desired color
# #           font=font) # ImageFont object with desired font
# # img.save('sample-out.png')
# # PImage("sample-out.png")


# acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
# print acc_decision_tree
