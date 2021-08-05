import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import streamlit as st 
import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)

pd.options.mode.chained_assignment = None  # default='warn'
df=pd.read_csv('InsuranceDemoMasterData v2_reducedversion.csv')
#Data cleaning
df.Gender[df.Gender == 'Male'] = 0
df.Gender[df.Gender == 'Female'] = 1
df=pd.get_dummies(df, columns = ['Region', 'AquisitionChannel'])
data=df.copy()
data=df[df['CarInsurance']==1]
dataset1reference=data.copy()
dataset2reference=data.copy()
dataset1reference=dataset1reference.reset_index()
data.drop('CustomerID', inplace=True, axis=1)
# Scaler
from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()
scaler1= scaler.fit(data)
scaled= scaler1.transform(data)

#Model-1
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'TwoWheelerInsurance', axis = 1), 
    data['TwoWheelerInsurance'], 
    test_size=0.2)

print(' x_train: ',X_train.shape, '\n',
      'y_train:',y_train.shape,'\n',
      'x_test:',X_test.shape,'\n',
      'y_test:',y_test.shape)

#Training Model
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_clf1 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf1.fit(X_train, y_train)

#Performance Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def score1(rf_clf1):
        y_pred = rf_clf1.predict(X_test)
        print(
            rf_clf1.__class__.__name__, '\n',
            'Accuracy score: ',accuracy_score(y_test, y_pred), '\n',
            'Precision score: ',precision_score(y_test, y_pred, zero_division = 1), '\n',
            'Recall score: ',recall_score(y_test, y_pred, zero_division = 1), '\n',
            'F1 score: ',f1_score(y_test, y_pred, zero_division = 1), '\n',
            'ROC AUC score: ',roc_auc_score(y_test, y_pred), '\n',
        )
#score1(rf_clf1)
#MOdel 2
p1_2wheeler=rf_clf1.predict_proba(data.drop(columns= 'TwoWheelerInsurance', axis = 1))[:,1]
p1_2wheeler_df=pd.DataFrame(p1_2wheeler,columns = ['p1_Two_Wheeler'])
# p1_2wheeler_df

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'CommercialInsurance', axis = 1), 
    data['CommercialInsurance'], 
    test_size=0.2)

rf_clf2 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf2.fit(X_train, y_train)
# score1(rf_clf2)
p1_CommercialInsurance_df=pd.DataFrame(rf_clf2.predict_proba(data.drop(columns= 'CommercialInsurance', axis = 1))[:,1],
                            columns = ['p1_CommercialInsurance'])
# p1_CommercialInsurance_df

#MOdel 3
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'HealthInsurance', axis = 1), 
    data['HealthInsurance'], 
    test_size=0.2)

rf_clf3 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf3.fit(X_train, y_train)
# score1(rf_clf2)
p1_HealthInsurance_df=pd.DataFrame(rf_clf3.predict_proba(data.drop(columns= 'HealthInsurance', axis = 1))[:,1],
                            columns = ['p1_HealthInsurance'])
# p1_HealthInsurance_df

#Model4
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'AccidentIsnurance', axis = 1), 
    data['AccidentIsnurance'], 
    test_size=0.2)

rf_clf4 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf4.fit(X_train, y_train)
# score1(rf_clf2)
p1_AccidentIsnurance_df=pd.DataFrame(rf_clf4.predict_proba(data.drop(columns= 'AccidentIsnurance', axis = 1))[:,1],
                            columns = ['p1_AccidentIsnurance'])
# p1_AccidentIsnurance_df
#Model 5
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'TravelInsurance', axis = 1), 
    data['TravelInsurance'], 
    test_size=0.2)

rf_clf5 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf5.fit(X_train, y_train)
# score1(rf_clf2)
p1_TravelInsurance_df=pd.DataFrame(rf_clf5.predict_proba(data.drop(columns= 'TravelInsurance', axis = 1))[:,1],
                            columns = ['p1_TravelInsurance'])
# p1_TravelInsurance_df
#Model 6
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'HomeInsurance', axis = 1), 
    data['HomeInsurance'], 
    test_size=0.2)

rf_clf6 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf6.fit(X_train, y_train)
# score1(rf_clf2)
p1_HomeInsurance_df=pd.DataFrame(rf_clf6.predict_proba(data.drop(columns= 'HomeInsurance', axis = 1))[:,1],
                            columns = ['p1_HomeInsurance'])
# p1_HomeInsurance_df

#Model 7
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns= 'LifeInsurance', axis = 1), 
    data['LifeInsurance'], 
    test_size=0.2)

rf_clf7 = RandomForestClassifier(max_depth=1, random_state=0)

rf_clf7.fit(X_train, y_train)
# score1(rf_clf2)
p1_LifeInsurance_df=pd.DataFrame(rf_clf7.predict_proba(data.drop(columns= 'LifeInsurance', axis = 1))[:,1],
                            columns = ['p1_LifeInsurance'])
# p1_LifeInsurance_df

#Full table of probability
p1_prob_table=pd.concat([p1_2wheeler_df,p1_CommercialInsurance_df,p1_HealthInsurance_df,
                         p1_AccidentIsnurance_df,p1_TravelInsurance_df,p1_HomeInsurance_df,p1_LifeInsurance_df],axis=1)
#Top 7 Insurance as per customer
CarInsuranceTops =pd.DataFrame(p1_prob_table.apply(lambda x:list(p1_prob_table.columns[np.array(x).argsort()[::-1][:7]]), axis=1).to_list(),  columns=['Top1', 'Top2', 'Top3','Top4','Top5','Top6','Top7'])

#Concat initial dataset with top 7
p1_output=pd.concat([dataset1reference,CarInsuranceTops],axis=1)


#Bough status 
p1_output["Top1_Bought_Status"]=""

for i in p1_output.index:
    if p1_output["Top1"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1

                
    elif p1_output["Top1"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
        
    elif p1_output["Top1"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
        
    elif p1_output["Top1"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
    
    elif p1_output["Top1"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
    
    elif p1_output["Top1"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
    
    elif p1_output["Top1"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top1_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top1_Bought_Status"][i] = 1
   
        
    
p1_output["Top2_Bought_Status"]=""

for i in p1_output.index:
    if p1_output["Top2"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top2_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1

                
    elif p1_output["Top2"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
        
    elif p1_output["Top2"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
        
    elif p1_output["Top2"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
    
    elif p1_output["Top2"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
    
    elif p1_output["Top2"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
    
    elif p1_output["Top2"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top2_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top2_Bought_Status"][i] = 1
   
        
p1_output["Top3_Bought_Status"]=""

for i in p1_output.index:
    if p1_output["Top3"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top3_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1

                
    elif p1_output["Top3"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1
        
    elif p1_output["Top3"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1
        
    elif p1_output["Top3"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1
    
    elif p1_output["Top3"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1
    
    elif p1_output["Top3"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1
    
    elif p1_output["Top3"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top3_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top3_Bought_Status"][i] = 1

p1_output["Top4_Bought_Status"]=""
for i in p1_output.index:
    if p1_output["Top4"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top4_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1

                
    elif p1_output["Top4"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
        
    elif p1_output["Top4"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
        
    elif p1_output["Top4"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
    
    elif p1_output["Top4"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
    
    elif p1_output["Top3"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
    
    elif p1_output["Top4"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top4_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top4_Bought_Status"][i] = 1
p1_output["Top5_Bought_Status"]=""
 
for i in p1_output.index:
    if p1_output["Top5"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top5_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1

                
    elif p1_output["Top5"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
        
    elif p1_output["Top5"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
        
    elif p1_output["Top5"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
    
    elif p1_output["Top5"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
    
    elif p1_output["Top5"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
    
    elif p1_output["Top5"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top5_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
   
       
    
p1_output["Top6_Bought_Status"]=""
for i in p1_output.index:
    if p1_output["Top6"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top6_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1

                
    elif p1_output["Top6"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1
        
    elif p1_output["Top6"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1
        
    elif p1_output["Top6"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1
    
    elif p1_output["Top6"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top5_Bought_Status"][i] = 1
    
    elif p1_output["Top6"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1
    
    elif p1_output["Top6"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top6_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top6_Bought_Status"][i] = 1
   
        
    
    
    
    
p1_output["Top7_Bought_Status"]=""
for i in p1_output.index:
    if p1_output["Top7"][i] == "p1_Two_Wheeler":
        if (p1_output["TwoWheelerInsurance"][i]==0):
           # print("true")
            p1_output["Top7_Bought_Status"][i] = "TwoWheelerInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1

                
    elif p1_output["Top7"][i] == "p1_CommercialInsurance":
        if (p1_output["CommercialInsurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "CommercialInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
        
    elif p1_output["Top7"][i] == "p1_AccidentIsnurance":
        if (p1_output["AccidentIsnurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "AccidentIsnurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
        
    elif p1_output["Top7"][i] == "p1_HealthInsurance":
        if (p1_output["HealthInsurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "HealthInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
    
    elif p1_output["Top7"][i] == "p1_TravelInsurance":
        if (p1_output["TravelInsurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "TravelInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
    
    elif p1_output["Top7"][i] == "p1_HomeInsurance":
        if (p1_output["HomeInsurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "HomeInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
    
    elif p1_output["Top7"][i] == "p1_LifeInsurance":
        if (p1_output["LifeInsurance"][i]==0):
#             print("true")
            p1_output["Top7_Bought_Status"][i] = "LifeInsurance"
        else:
#             print ("false")
            p1_output["Top7_Bought_Status"][i] = 1
   #Recommendation
p1_output["Recommendation"]=p1_output["Top1_Bought_Status"].astype(str)+"_" + p1_output["Top2_Bought_Status"].astype(str)+"_"  + p1_output["Top3_Bought_Status"].astype(str)+"_"  + p1_output["Top4_Bought_Status"].astype(str)+"_"  + p1_output["Top5_Bought_Status"].astype(str)+"_" +p1_output["Top6_Bought_Status"].astype(str)+"_"  + p1_output["Top7_Bought_Status"].astype(str) 


    
#OUTPUT
st.sidebar.title("Navigation")
select = st.sidebar.radio("GO TO:",('Cross-Sell Existing Customers','Hyper-Personalization', 'Look-Alike Analysis'))

#Insurance Recommendation

if select == 'Cross-Sell Existing Customers':
    st.title('Insurance Recommendation')
    Cust_id = st.number_input("Enter Customer ID", step=1)
    if st.button("Recommend Insurance"):
        
        
        for row in range(0,p1_output.shape[0]):
            str=p1_output.loc[row,"Recommendation"]
            arr=str.split("_")
            arr2=[]
            arr3=[]
            for i in range(0,len(arr)):
                if (arr[i]=="1" or arr[i]==""):
                    arr3.append(arr[i])
                else:
                    arr2.append(arr[i])

            if (len(arr2)>=3):   
                 p1_output.loc[row,"top1recom"]=arr2[0]
                 p1_output.loc[row,"top2recom"]=arr2[1]
                 p1_output.loc[row,"top3recom"]=arr2[2]  
            elif (len(arr2)==2):
                p1_output.loc[row,"top1recom"]=arr2[0]
                p1_output.loc[row,"top2recom"]=arr2[1]
            elif (len(arr2)==1):
                p1_output.loc[row,"top1recom"]=arr2[0]
        
        st.write("1st Recommendation: ", p1_output.loc[Cust_id]["top1recom"])
        st.write("2nd Recommendation: ", p1_output.loc[Cust_id]["top2recom"])
        st.write("3rd Recommendation: ", p1_output.loc[Cust_id]["top3recom"])
        st.write(p1_output.loc[[Cust_id]].drop(["CustomerID","index"],axis=1))
        st.write(p1_prob_table.loc[[Cust_id]])
        st.write('Recommendation String:')
        st.write(p1_output['Recommendation'][Cust_id])
        
        
                 
        

#User Recommendation
if select == 'Hyper-Personalization':
    st.title('Customer Details')
    st.subheader('Please fill the details for recommendation')
    st.subheader('New Customer Information')
    Cust_id = st.number_input("Enter Customer ID", step=1)
    Cust_age = st.number_input("Enter Customer Age", step=1)
    Fam_Member = st.number_input("Enter No. of family Member",step=1)
    Region = st.selectbox('Select Region',('Region1','Region2','Region3','Region4','Region5','Region6','Region7','Region8','Region9'))
    Gender = st.selectbox('Select Gender',('Male','Female'))
    Edu = st.number_input("Enter Customer Education level(5-10)")
    Emp = st.number_input("Enter Customer Employment level(5-10)")
    Inc = st.number_input("Enter Customer Income level(5-10)")
    display = ("Married", "Unmarried")
    options = list(range(len(display)))
    Marital = st.selectbox("Select Marital Status", options, format_func=lambda x: display[x])

    #Marital = st.selectbox('Select Marital Status',('Married','Unmarried'))
    Dfrst = st.number_input("Enter number of days since first product bought", step=1)
    NoOfTransaction = st.number_input("Enter number of transaction(0-30)",step=1)
    cash = st.number_input("Enter number of Cash transaction(0-10)",step=1)
    cheque = st.number_input("Enter number of Cheque transaction(0-10)",step=1)
    card = st.number_input("Enter number of Card transaction(0-10)",step=1)
    Dlast = st.number_input("Enter number of days since last transaction", step=1)
    Dnext= st.number_input("Enter number of days to next renewal", step=1)
    DPpaid= st.number_input("Enter number of premium paid(0-39)", step=1)
    PAmount=st.number_input("Enter Premium Amount Paid(5000-14999)", step=1)
    AqChnl=st.selectbox('Select Acquisition Channel',('Online','Reseller','Direct'))
    CarOwn=st.number_input('Car Ownership',step=1)
    wheelOwn=st.number_input('Two Wheeler Ownership',step=1)
    Medical=st.number_input('Medical Bill',step=1)
    Total_Policy_Matured=st.number_input("Enter Number of Policies Matured(0-8)", step=1)
    Total_Policy_Bought=st.number_input("Enter Number of Policies Bought(1-8)", step=1)
    Annual=st.number_input("Enter Number of Annual Policies(0-8)", step=1)
    SemiAnnual=st.number_input("Enter Number of Semi-Annual Policies(0-8)", step=1)
    Quarterly=st.number_input("Enter Number of Quarterly Policies(0-8)", step=1)
    Monthly=st.number_input("Enter Number of Monthly Policies(0-8)", step=1)
    SalesRep=st.number_input('Sales Rep Rating',step=1)
    EmailRes=st.number_input('Email Response',step=1)
    CallRes=st.number_input('Call Response',step=1)
    MsgRes=st.number_input('Msg Response',step=1)
    Complaint_Loggs=st.number_input('Complain Logged',step=1)
    NoOfClaims=st.number_input('Claim Count',step=1)
    ClaimsPaid=st.number_input('Claim Paid',step=1)
    ClaimsDenied=st.number_input('Claim Denied',step=1)
    CarInsurance= 1
    Two_wheelInsurance=st.number_input('Twowheel Insurance',step=1)
    CommercialInsurance=st.number_input('Commercial Insurance',step=1)
    HealthInsurance=st.number_input('Health Insurance',step=1)
    AccidentIsnurance=st.number_input('Accident Insurance',step=1)
    TravelInsurance=st.number_input('Travel Insurance',step=1)
    HomeInsurance=st.number_input('Home Insurance',step=1)
    LifeInsurance=st.number_input('Life Insurance',step=1)
    
 
    
    X=({"CustomerID":[Cust_id],"Age":[Cust_age],"FamilyMember":[Fam_Member],"Gender":[Gender],"EducationScore":[Edu],"EmploymentScore":[Emp],"IncomeGroup":[Inc],"Marital Status":[Marital],"DaysSinceFirstProduct":[Dfrst],"NumberofTransactions":[NoOfTransaction],"CashTransactions":[cash],"ChequeTransaction":[cheque],"CardTransaction":[card],"DaysSinceLastTransaction":[Dlast],"DaystoNextRenewal":[Dnext],"NumberPremiumPaid":[DPpaid],"PremiumAmt":[PAmount],"Carwonership":[CarOwn],"TwowheelerOwnership":[wheelOwn],"MedicalBill":[Medical],"TotalPoliciesMatured":[Total_Policy_Matured],
    "TotalPoliciesBought":[Total_Policy_Bought],"AnnualPolicies":[Annual],"SemiAnnualPolicies":[SemiAnnual],"QuarterlyPolicies":[Quarterly],"MonthlyPolicies":[Monthly],"SalesRepRating":[SalesRep],
    "Email_Response":[EmailRes],"Call_Respone":[CallRes],"Message_Response":[MsgRes],"Complaintslogged":[Complaint_Loggs],"NumberofClaims":[NoOfClaims],"ClaimsPaid":[ClaimsPaid],"ClaimsDenied":[ClaimsDenied],"CarInsurance":[CarInsurance],"TwoWheelerInsurance":[Two_wheelInsurance],"CommercialInsurance":[CommercialInsurance],
    "HealthInsurance":[HealthInsurance],"AccidentIsnurance":[AccidentIsnurance],"TravelInsurance":[TravelInsurance],"HomeInsurance":[HomeInsurance],"LifeInsurance":[LifeInsurance]})

    df1=pd.DataFrame.from_dict(X)
 #Store the above details in df1
 
    df1.Gender[df1.Gender == 'Male'] = 0
    df1.Gender[df1.Gender == 'Female'] = 1
    #df1=pd.get_dummies(df1, columns = ['Region', 'AquisitionChannel'])
    if (Region=="Region1"):
        df1["Region_Region 1"]= 1
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region2"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 1
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region3"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 1
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region4"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 1
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region5"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 1
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0  
    elif (Region=="Region6"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 1
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region7"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 1
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region8"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 1
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 0
    elif (Region=="Region9"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 1
        df1["Region_Region 10"]= 0
    elif (Region=="Region10"):
        df1["Region_Region 1"]= 0
        df1["Region_Region 2"]= 0
        df1["Region_Region 3"]= 0
        df1["Region_Region 4"]= 0
        df1["Region_Region 5"]= 0
        df1["Region_Region 6"]= 0
        df1["Region_Region 7"]= 0
        df1["Region_Region 8"]= 0
        df1["Region_Region 9"]= 0
        df1["Region_Region 10"]= 1
    if (AqChnl=="Online"):
        df1["AquisitionChannel_Direct"]= 0
        df1["AquisitionChannel_Online"]= 1
        df1["AquisitionChannel_Reseller"]= 0
    elif(AqChnl=="Direct"):
        df1["AquisitionChannel_Direct"]= 1
        df1["AquisitionChannel_Online"]= 0
        df1["AquisitionChannel_Reseller"]= 0
    elif(AqChnl=="Reseller") :
        df1["AquisitionChannel_Direct"]= 0
        df1["AquisitionChannel_Online"]= 0
        df1["AquisitionChannel_Reseller"]= 1
   
    data1=df1.copy()
    data1=df1[df1['CarInsurance']==1]
    data1=df1.drop("CustomerID", axis=1)
    dataset1reference1=data1.copy()
    dataset1reference1=dataset1reference1.reset_index()
#Model-1
    
    p1_2wheeler=rf_clf1.predict_proba(data1.drop(columns= 'TwoWheelerInsurance', axis = 1))[:,1]
    p1_2wheeler_df1=pd.DataFrame(p1_2wheeler,columns = ['p1_Two_Wheeler'])

    # p1_2wheeler_df
    #MOdel 2

    # score1(rf_clf2)
    p1_CommercialInsurance_df1=pd.DataFrame(rf_clf2.predict_proba(data1.drop(columns= 'CommercialInsurance', axis = 1))[:,1],
                                columns = ['p1_CommercialInsurance'])
    # p1_CommercialInsurance_df

    #MOdel 3

    p1_HealthInsurance_df1=pd.DataFrame(rf_clf3.predict_proba(data1.drop(columns= 'HealthInsurance', axis = 1))[:,1],
                                columns = ['p1_HealthInsurance'])
    # p1_HealthInsurance_df

    #Model4

    p1_AccidentIsnurance_df1=pd.DataFrame(rf_clf4.predict_proba(data1.drop(columns= 'AccidentIsnurance', axis = 1))[:,1],
                                columns = ['p1_AccidentIsnurance'])
    # p1_AccidentIsnurance_df

    #Model 5

    p1_TravelInsurance_df1=pd.DataFrame(rf_clf5.predict_proba(data1.drop(columns= 'TravelInsurance', axis = 1))[:,1],
                                columns = ['p1_TravelInsurance'])
    # p1_TravelInsurance_df

    #Model 6

    p1_HomeInsurance_df1=pd.DataFrame(rf_clf6.predict_proba(data1.drop(columns= 'HomeInsurance', axis = 1))[:,1],
                                columns = ['p1_HomeInsurance'])
    # p1_HomeInsurance_df

    #Model 7

    p1_LifeInsurance_df1=pd.DataFrame(rf_clf7.predict_proba(data1.drop(columns= 'LifeInsurance', axis = 1))[:,1],
                                columns = ['p1_LifeInsurance'])
    # p1_LifeInsurance_df


    #Full table of probability
    p1_prob_table1=pd.concat([p1_2wheeler_df1,p1_CommercialInsurance_df1,p1_HealthInsurance_df1,
                             p1_AccidentIsnurance_df1,p1_TravelInsurance_df1,p1_HomeInsurance_df1,p1_LifeInsurance_df1],axis=1)
    #Top 7 Insurance as per customer
    CarInsuranceTops1 =pd.DataFrame(p1_prob_table1.apply(lambda x:list(p1_prob_table.columns[np.array(x).argsort()[::-1][:7]]), axis=1).to_list(),  columns=['Top1', 'Top2', 'Top3','Top4','Top5','Top6','Top7'])

    #Concat initial dataset with top 7
    p1_output1=pd.concat([dataset1reference1,CarInsuranceTops1],axis=1)


#Bough status 
    p1_output1["Top1_Bought_Status"]=""

    for i in p1_output1.index:
        if p1_output1["Top1"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1


        elif p1_output1["Top1"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1

        elif p1_output1["Top1"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "AccidentIsnurance" 
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1

        elif p1_output1["Top1"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1

        elif p1_output1["Top1"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1

        elif p1_output1["Top1"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1

        elif p1_output1["Top1"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top1_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top1_Bought_Status"][i] = 1


    p1_output1["Top2_Bought_Status"]=""

    for i in p1_output1.index:
        if p1_output1["Top2"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top2_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1


        elif p1_output1["Top2"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

        elif p1_output1["Top2"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "AccidentIsnurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

        elif p1_output1["Top2"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

        elif p1_output1["Top2"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

        elif p1_output1["Top2"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

        elif p1_output1["Top2"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top2_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top2_Bought_Status"][i] = 1

    p1_output1["Top3_Bought_Status"]=""

    for i in p1_output1.index:
        if p1_output1["Top3"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top3_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1


        elif p1_output1["Top3"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "AccidentIsnurance" 
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top3_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top3_Bought_Status"][i] = 1

    p1_output1["Top4_Bought_Status"]=""
    for i in p1_output1.index:
        if p1_output1["Top4"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top4_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1


        elif p1_output1["Top4"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

        elif p1_output1["Top4"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "AccidentIsnurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

        elif p1_output1["Top4"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

        elif p1_output1["Top4"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

        elif p1_output1["Top3"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

        elif p1_output1["Top4"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top4_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top4_Bought_Status"][i] = 1

    p1_output1["Top5_Bought_Status"]=""

    for i in p1_output1.index:
        if p1_output1["Top5"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top5_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1


        elif p1_output1["Top5"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top5"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "AccidentIsnurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top5"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top5"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top5"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top5"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top5_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

    p1_output1["Top6_Bought_Status"]=""
    for i in p1_output1.index:
        if p1_output1["Top6"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top6_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1


        elif p1_output1["Top6"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1

        elif p1_output1["Top6"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "AccidentIsnurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1

        elif p1_output1["Top6"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1

        elif p1_output1["Top6"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top5_Bought_Status"][i] = 1

        elif p1_output1["Top6"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1

        elif p1_output1["Top6"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top6_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top6_Bought_Status"][i] = 1



    p1_output1["Top7_Bought_Status"]=""
    for i in p1_output1.index:
        if p1_output1["Top7"][i] == "p1_Two_Wheeler":
            if (p1_output1["TwoWheelerInsurance"][i]==0):
               # print("true")
                p1_output1["Top7_Bought_Status"][i] = "TwoWheelerInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1


        elif p1_output1["Top7"][i] == "p1_CommercialInsurance":
            if (p1_output1["CommercialInsurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "CommercialInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1

        elif p1_output1["Top7"][i] == "p1_AccidentIsnurance":
            if (p1_output1["AccidentIsnurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "AccidentIsnurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1

        elif p1_output1["Top7"][i] == "p1_HealthInsurance":
            if (p1_output1["HealthInsurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "HealthInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1

        elif p1_output1["Top7"][i] == "p1_TravelInsurance":
            if (p1_output1["TravelInsurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "TravelInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1

        elif p1_output1["Top7"][i] == "p1_HomeInsurance":
            if (p1_output1["HomeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "HomeInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1

        elif p1_output1["Top7"][i] == "p1_LifeInsurance":
            if (p1_output1["LifeInsurance"][i]==0):
    #             print("true")
                p1_output1["Top7_Bought_Status"][i] = "LifeInsurance"
            else:
    #             print ("false")
                p1_output1["Top7_Bought_Status"][i] = 1 


     #Recommendation
    p1_output1["Recommendation"]=p1_output1["Top1_Bought_Status"].astype(str)+"_" + p1_output1["Top2_Bought_Status"].astype(str)+"_"  + p1_output1["Top3_Bought_Status"].astype(str)+"_"  + p1_output1["Top4_Bought_Status"].astype(str)+"_"  + p1_output1["Top5_Bought_Status"].astype(str)+"_" +p1_output1["Top6_Bought_Status"].astype(str)+"_"  + p1_output1["Top7_Bought_Status"].astype(str) 

    if st.button("Predict"): 

        st.write('Probability: ',p1_prob_table1)  
        st.write('Prediction: ',p1_output1['Recommendation'])  
        

if select == 'Look-Alike Analysis':
    #data=data.reset_index(inplace = True, drop = True)
#     st.write(dataset2reference)
    dataset1reference=dataset2reference.reset_index(inplace = True, drop = True)
#     st.write(dataset2reference)
    
    st.title('Look Alike Analysis')
    min_max_scaler1 = MinMaxScaler()
    scaled_df = min_max_scaler1.fit_transform(data)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(scaled_df)
    distances, indices = nbrs.kneighbors(scaled_df)
#     st.write(indices)
#     Customer_id1=0
    Customer_id1=st.number_input("Please Enter Customer ID",step=1)
    Recomm= pd.DataFrame(indices[int(Customer_id1)])
    Recomm.columns=["index"]
    st.write("Top 10 Most similar Customers",Recomm)
#     for index in indices[int(Customer_id1)][1:]:
#     #print (index)
#         st.write(data.iloc[index])
# #     result=pd.merge(p1_output, Recomm, on="index", left_index=True, right_index=True)
# #     st.write("Top 10 Most similar Customers",result)
#     for ind in indices[0][1:]:
#         data.iloc[ind]

    
    
    
    ind_df=pd.DataFrame(indices)
#     st.write(ind_df)
#     ind_df=ind_df.reset_index(inplace = True, drop = True)
#     st.write(ind_df)
   
    lk_output=pd.concat([dataset2reference,ind_df], axis=1)
#     st.write(lk_output)
    temp3=pd.DataFrame()
    user_value=Customer_id1
    for i in range(0,11):
      temp=lk_output[[0,1,2,3,4,5,6,7,8,9,10]].iloc[user_value,i]
      temp2=pd.DataFrame(lk_output.loc[lk_output.index==temp])
      temp2.drop('CustomerID', inplace=True, axis=1)
      temp3=temp3.append(temp2)
    temp3.drop([0,1,2,3,4,5,6,7,8,9,10],inplace=True,axis=1)
    st.write(temp3)
#       st.write(lk_output.loc[lk_output.index==temp])
