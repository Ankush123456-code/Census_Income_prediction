import pickle
import pandas as pd
import numpy as np

import streamlit as st
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

dataset = pd.read_csv('train.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset['wage_class'] = le.fit_transform(dataset['wage_class'])

dataset = dataset.replace('?', np.nan)

columns_with_nan = ['workclass', 'occupation', 'native_country']

for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

X = dataset.drop('wage_class', axis=1)
Y = dataset['wage_class']

X = X.drop(["Person_Id",'workclass', 'education', 'race', 'sex',
            'capital_loss', 'native_country', 'fnlwgt',"capital_gain",
            ], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

@st.cache()
def prediction( age_value, edu_num_value, marital_value,occupation_value,Relationship, hours_value):

    if marital_value == 'Married-civ-spouse':
        marital_value = 1
    elif marital_value == 'Never-married':
        marital_value = 2
    elif marital_value == 'Divorced':
        marital_value = 3
    elif marital_value == 'Separated':
        marital_value = 4
    elif marital_value == 'Widowed':
        marital_value = 5
    elif marital_value == 'Married-spouse-absent':
        marital_value = 6
    elif marital_value == 'Married-AF-spouse':
        marital_value = 7


    if Relationship == 'Husband':
        Relationship = 1
    elif Relationship == 'Not-in-family':
        Relationship = 2
    elif Relationship == 'Own-child':
        Relationship = 3
    elif Relationship == 'Unmarried':
        Relationship = 4
    elif Relationship == 'Wife':
        Relationship = 5
    elif Relationship == 'Other-relative':
        Relationship = 6
    # Making predictions
    features = [age_value, edu_num_value, marital_value,
                occupation_value,Relationship, hours_value]

    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(scaler.transform(final_features))
    if prediction == 1:
        return  "Income is more than 50K"
    elif prediction == 0:
        return  "Income is less than 50K"


def main():
    html_temp = """ 
    <div style ="background-color:#00bff;padding:14px"> 
    <h1 style ="color:black;text-align:center;">Census Income Predictor</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction

    age_value = st.number_input("age_value")
    edu_num_value = st.number_input("edu_num_value")
    marital_value = st.selectbox('Marital Status', ('Married-civ-spouse',
                                                                'Never-married',
                                                                'Divorced',
                                                                'Separated',
                                                                'Widowed',
                                                                'Married-spouse-absent',
                                                                'Married-AF-spouse'))


    occupation_value = st.number_input("occupation_Code: [range = 1 to 20]")
    Relationship = st.selectbox('Relationship', ('Husband',
                                                 'Not-in-family',
                                                 'Own-child',
                                                 'Unmarried',
                                                 'Wife',
                                                 'Other-relative',))
    # capital_gain = st.number_input("capital_gain")
    hours_value = st.number_input("Hours of work per week")
    result=""
    if st.button("Predict"):
        result = prediction(age_value, edu_num_value, marital_value,occupation_value,Relationship, hours_value)
        st.success('Your Census Income Predictor {}'.format(result))


if __name__ == '__main__':
    main()
