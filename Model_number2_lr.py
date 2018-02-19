import csv
import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#data cleaning and preparation

#read training data
df = pd.read_csv("exercise_04_train.csv")

#remove samples with missing data
df_s = df[~df.isnull().any(axis=1)]

#show object data
df_s.select_dtypes(include=['object']).head()

#price data is converted to float
df_s.x41 = df_s.x41.str.replace("$","")
df_s.x41 = df_s.x41.convert_objects(convert_numeric=True)

#replace brand name, date, and area percentage to integer
cleanup_nums = {"x35": {"monday": 1, "tuesday": 2, "wed": 3, "wednesday": 3, "thur": 4, "thurday": 4, "friday": 5, "fri": 5 },
                "x68": {"July": 7, "Jun": 6, "Aug": 8, "May": 5, "sept.": 9, "Apr": 4, "Oct": 10, "Mar": 3, "Nov": 11, "Feb": 2, "Dev": 12, "January": 1 },
                "x93": {"asia": 0, "america": 1, "euorpe": 2},
                "x34": {"volkswagon": 0, "Toyota": 1, "bmw": 2, "Honda": 3, "tesla": 4, "chrystler": 5, "nissan": 6, "ford": 7, "mercades": 8, "chevrolet": 9 },
                "x45": {"0.01%": 0, "-0.01%": 1, "0.0%": 2, "-0.0%": 3, "-0.02%": 4, "0.02%": 5, "0.03%": 6, "-0.03%": 7, "-0.04%": 8, "0.04%": 9 }  
                }
df_s.replace(cleanup_nums, inplace=True)

####################################################

#Modeling

#independent variable is y and dependet variables are x
xtrain = df_s.iloc[:,0:df_s.shape[1]-1]
ytrain = df_s.iloc[:,df_s.shape[1]-1]



#use pipeline to do feature selection and build support vector machine classifier
pipeline = Pipeline([
    ('select', RFE(LogisticRegression(), 10)),
    ('classifier', LogisticRegression())
])

#tume model by select the best parameters
grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [1.0, 0.8],
    'classifier__class_weight': [None, 'balanced'],
    'classifier__n_jobs': [-1]
}

grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy', n_jobs=-1, cv=5)
grid_search.fit(X=xtrain, y=ytrain)

#build model using optimized parameters
model = pipeline.fit(xtrain, ytrain)

####################################################

#predict on testing data

#read testing data
df_test = pd.read_csv("exercise_04_test.csv")

#remove samples with missing data
df_st = df_test[~df_test.isnull().any(axis=1)]

#show object data
df_st.select_dtypes(include=['object']).head()

#price data is converted to float
df_st.x41 = df_st.x41.str.replace("$","")
df_st.x41 = df_st.x41.convert_objects(convert_numeric=True)

#replace date to integer
cleanup_nums = {"x35": {"monday": 1, "tuesday": 2, "wed": 3, "wednesday": 3, "thur": 4, "thurday": 4, "friday": 5, "fri": 5 },
                "x68": {"July": 7, "Jun": 6, "Aug": 8, "May": 5, "sept.": 9, "Apr": 4, "Oct": 10, "Mar": 3, "Nov": 11, "Feb": 2, "Dev": 12, "January": 1 },
                "x93": {"asia": 0, "america": 1, "euorpe": 2},
                "x34": {"volkswagon": 0, "Toyota": 1, "bmw": 2, "Honda": 3, "tesla": 4, "chrystler": 5, "nissan": 6, "ford": 7, "mercades": 8, "chevrolet": 9 },
                "x45": {"0.01%": 0, "-0.01%": 1, "0.0%": 2, "-0.0%": 3, "-0.02%": 4, "0.02%": 5, "0.03%": 6, "-0.03%": 7, "-0.04%": 8, "0.04%": 9 }  
                }

df_st.replace(cleanup_nums, inplace=True)

#test and store predicted probability
predicted = model.predict(df_st)
pp = model.predict_proba(df_st)
np.savetxt("p0_lr.csv",predicted, delimiter=",")
np.savetxt("p1_lr.csv",pp, delimiter=",")

