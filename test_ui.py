import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

# Data
# df = pd.read_csv("adult.csv")
# df = pd.read_csv("Update_adult.csv")

@st.cache(allow_output_mutation=True, persist=True)
def load_data():
    return pd.read_csv("Update_adult.csv", na_values='?')

df = load_data()

education_dict = dict(
    zip(df['education'].unique(), df['education.num'].unique()))


# Sidebar
st.sidebar.header("**User Input Parameters**")
age = st.sidebar.slider("Age", np.int(df['age'].min(
)), np.int(df['age'].max()), np.int(math.floor(df['age'].mean())))
workclass = st.sidebar.selectbox(
    "Workclass", list(df['workclass'].unique()[1:]))
fnlwgt = st.sidebar.slider("Final Weight", np.int(df['fnlwgt'].min(
)), np.int(df['fnlwgt'].max()), np.int(math.floor(df['fnlwgt'].mean())))
education = st.sidebar.selectbox("Edication", list(
    df.sort_values(by='education.num')['education'].unique()))
education_num = education_dict[education]
maritalstatus = st.sidebar.selectbox("Marital Status", list(
    df.sort_values(by='marital.status')['marital.status'].unique()))
occupation = st.sidebar.selectbox("Occupation", list(
    df.sort_values(by='occupation')['occupation'].unique()[1:]))
relationship = st.sidebar.selectbox("Relationship", list(
    df.sort_values(by='relationship')['relationship'].unique()[1:]))
race = st.sidebar.selectbox("Race", list(
    df.sort_values(by='race')['race'].unique()))
sex = st.sidebar.selectbox("Sex", list(
    df.sort_values(by='sex')['sex'].unique()))
gain = st.sidebar.slider("Capital Gain", np.int(df['capital.gain'].min(
)), np.int(df['capital.gain'].max()), np.int(math.floor(df['capital.gain'].mean())))
loss = st.sidebar.slider("Capital Loss", np.int(df['capital.loss'].min(
)), np.int(df['capital.loss'].max()), np.int(math.floor(df['capital.loss'].mean())))
hoursperweek = st.sidebar.slider("Hours Per Week", np.int(df['hours.per.week'].min(
)), np.int(df['hours.per.week'].max()), np.int(math.floor(df['hours.per.week'].mean())))
nativecountry = st.sidebar.selectbox("Native Country", list(
    df['native.country'].sort_values().unique()[1:]), 38)
# income=st.sidebar.selectbox("Income",list(df['income'].unique()))
# Main Area
# DataTab
data_expander = st.beta_expander("Dataset")

data_expander.header("Dataset")
data_expander.write("""
 **Rows:** """ + str(df.shape[0])+""" **Attributes:** """+str(df.shape[1]))

data_expander.dataframe(df.head(10))

# df.shape

# st.dataframe(df.groupby('native.country').count()['age'])

# fig=px.bar(data_frame=df, x="native.country", y="workclass",  barmode="group")
# st.plotly_chart(fig)
# st.write(df['native.country'].value_counts().plot.pie(autopct="%1.1f%%"))
# st.pyplot()

# fig, ax = plt.subplots(2, 3, figsize=(10, 8))
# for i, v in enumerate(df.select_dtypes(include=np.number)):
#     sns.kdeplot(data=df[v], ax=ax.flatten()[i])
# plt.tight_layout()
# st.pyplot(plt)

# User Input Tab
@st.cache(show_spinner=True,allow_output_mutation=True)
def input_params(age,workclass,fnlwgt,education,education_num,maritalstatus,occupation,relationship,race,sex,gain,loss,hoursperweek,nativecountry):

    selected_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education.num': [education_num], 
        'marital.status': [maritalstatus],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital.gain': [gain],
        'capital.loss': [loss],
        'hours.per.week': [hoursperweek],
        'native.country': [nativecountry]
    })
    return selected_data
    
selected_data=input_params(age,workclass,fnlwgt,education,education_num,maritalstatus,occupation,relationship,race,sex,gain,loss,hoursperweek,nativecountry)
user_input_expander = st.beta_expander("User Input")
user_input_expander.write('Data')
user_input_expander.dataframe(selected_data)

prediction_expander=st.beta_expander("Prediction")
prediction_expander.write("Result")
enc=pickle.load(open("OneHotEncoder.pkl","rb"))
cat_cols=['workclass', 'education', 'marital.status', 'occupation',
       'relationship', 'race', 'sex', 'native.country']
num_data=selected_data.select_dtypes(include=np.number)
data_enc=pd.concat([pd.DataFrame(num_data.values,columns=num_data.columns),pd.DataFrame(enc.transform(selected_data[cat_cols]).toarray(),columns=enc.get_feature_names())],axis=1)
# prediction_expander.dataframe(data_enc)
lg=pickle.load(open("lg.pkl","rb"))
res=lg.predict(data_enc.drop(columns=['fnlwgt']))
res_prob=lg.predict_proba((data_enc.drop(columns=['fnlwgt'])))
prediction_expander.dataframe(pd.DataFrame({"Class":res})['Class'].map({0:"<=50K",1:">50K"}))

prediction_expander.write("Probabilities")
prediction_expander.dataframe(pd.DataFrame(res_prob,columns=['<=50K','>50K']))

