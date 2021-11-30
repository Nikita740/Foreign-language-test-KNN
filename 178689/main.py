import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv("train.csv")
df["Medu"].fillna(-1, inplace = True)
df["Fedu"].fillna(-1, inplace = True)

def sex(sex):
    if sex == 'F':
        return 0
    else:
        return 1

def edu(edu):
    if edu == 'none' or edu == -1:
        return 0
    elif edu == 'primary education (4th grade)':
        return 1
    elif edu == '5th to 9th grade':
        return 2
    elif edu == 'secondary education':
        return 3
    elif edu == 'higher education':
        return 4
def famsize(famsize):
    if famsize == '3 persons or less':
        return 0
    else:
        return 1
def nursery(nursery):
    if nursery == 'yes':
        return 1
    else:
        return 0
def activities(activities):
    if activities == 'yes':
        return 1
    else:
        return 0
def internet(internet):
    if internet == 'yes':
        return 1
    else:
        return 0
def famrel(famrel):
    if famrel == 'very bad':
        return 0
    elif famrel == 'normal':
        return 1
    elif famrel == 'good':
        return 2
    elif famrel == 'excellent':
        return 3
def freetime(freetime):
    if freetime == "very low":
        return 0
    elif freetime == 'low':
        return 1
    elif freetime == 'medium':
        return 2
    elif freetime == 'high':
        return 3
    elif freetime == "very high":
        return 4
def higher(higher):
    if higher == "yes":
        return 1
    else:
        return 0
def famsup(famsup):
    if famsup == "yes":
        return 1
    else:
        return 0
def paid(paid):
    if paid == "yes":
        return 1
    else:
        return 0
def Mjob(Mjob):
    if Mjob == 'teacher':
        return 1
    else:
        return 0
def Fjob(Fjob):
    if Fjob == 'teacher':
        return 1
    else:
        return 0
df.fillna(-1, inplace=True)
df['activities'] = df['activities'].apply(Mjob)
df['Mjob'] = df['Mjob'].apply(Mjob)
df['Fjob'] = df['Fjob'].apply(Fjob)
df['paid'] = df['paid'].apply(paid)
df['famsup'] = df['famsup'].apply(famsup)
df['higher'] = df['higher'].apply(higher)
df['internet'] = df['internet'].apply(internet)
df['famrel'] = df['famrel'].apply(famrel)
df['freetime'] = df['freetime'].apply(freetime)
df['nursery'] = df['nursery'].apply(nursery)
df['famsize'] = df["famsize"].apply(famsize)
df['Medu'] = df["Medu"].apply(edu)
df['Fedu'] = df["Fedu"].apply(edu)
df["sex"] = df["sex"].apply(sex)

df.drop(['reason', 'guardian', 'schoolsup', 'address', 'traveltime', 'studytime'], axis = 1, inplace = True)

med_famrel = df['famrel'].median()
df.fillna(med_famrel, inplace = True)

#df[list(pd.get_dummies(df['sex']).columns)] = pd.get_dummies(df['sex'])
df.info()
print(df['activities'].value_counts())
X = df.drop('result', axis = 1)
Y = df['result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred) * 100
confusion_matrix(Y_test, Y_pred)
print(Y_test, Y_pred)
print(accuracy_score(Y_test, Y_pred) * 100)
print(confusion_matrix(Y_test, Y_pred))

# Здесь должен быть твой код