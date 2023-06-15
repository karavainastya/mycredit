#import sklearn

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import streamlit as st
import pickle
import numpy as np


import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('default1.jpg')


classifier_name=['LogisticRegression', 'LightGBM']
#option = st.sidebar.selectbox('Какой алгоритм запустить?', classifier_name)
#st.subheader(option)



#Importing model and label encoders
model=pickle.load(open("model.pkl","rb"))
#model_1 = pickle.load(open("final_rf_model.pkl","rb"))
#le_pik=pickle.load(open("label_encoding_for_gender.pkl","rb"))
#le1_pik=pickle.load(open("label_encoding_for_geo.pkl","rb"))

    
def predict_churn(CODE_GENDER_M, CODE_GENDER_F, CODE_GENDER_XNA, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,                
NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State_servant, NAME_INCOME_TYPE_Commercial_associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, 
NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman,  NAME_INCOME_TYPE_Maternity_leave, REG_CITY_NOT_WORK_CITY, REGION_RATING_CLIENT):
    input = np.array([[CODE_GENDER_M, CODE_GENDER_F, CODE_GENDER_XNA, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,                
NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State_servant, NAME_INCOME_TYPE_Commercial_associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, 
NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman,  NAME_INCOME_TYPE_Maternity_leave, REG_CITY_NOT_WORK_CITY, REGION_RATING_CLIENT]]).astype(np.float64)
    #if option == 'LogisticRegression':
        
    prediction = model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)

    #else:
        #pred=0.40
        #st.markdown('Вероятно, кредит будет погашен.')
    return float(pred)


def main():
    #st.title("Прогноз вероятности дефолта клиента")
    html_temp = """
    <div style="background-color:yellow ;padding:10px">
    <h2 style="color:red;text-align:center;">Прогноз вероятности дефолта клиента</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.subheader("Приложение создано для курса Diving into Darkness of Data Science")
    st.sidebar.info("Разработчик - Каравай А.Л.")
    st.sidebar.image("my.jpg", width=300)
    
 #CODE_GENDER_M, CODE_GENDER_F, CODE_GENDER_XNA, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,                
#NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State_servant, NAME_INCOME_TYPE_Commercial associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, 
#NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman,  NAME_INCOME_TYPE_Maternity_leave, REG_CITY_NOT_WORK_CITY, REGION_RATING_CLIENT
    CODE_GENDER_M =st.selectbox("Пол: женский", ['0', '1'])
    CODE_GENDER_F =st.selectbox("Пол: мужской", ['0', '1'])
    CODE_GENDER_XNA=st.selectbox("Пол: небинарный", ['0', '1'])
    if (int(CODE_GENDER_M)==1 and int(CODE_GENDER_F)==1) or (int(CODE_GENDER_M)==1 and int(CODE_GENDER_XNA)==1) or (int(CODE_GENDER_F)==1 and int(CODE_GENDER_XNA)==1) or (int(CODE_GENDER_F)==1 and int(CODE_GENDER_XNA)==1 and int(CODE_GENDER_M)==1):
        st.error('Некорректный ввод данных по полу')
        
    DAYS_BIRTH = st.slider('Возраст клиента: Примечание: кредит выдается клиентам старше 22 лет', 22, 70) 
        #if int(Возраст)- int(Стаж)< 18:
            #st.error('Некорректный ввод данных по возрасту клиента и/или стажу работы')
            
    DAYS_EMPLOYED = st.slider('Стаж работы:', 0, 50) 
    if int(DAYS_BIRTH)- int(DAYS_EMPLOYED)< 18:
            st.error('Некорректный ввод данных по возрасту клиента и/или стажу работы')
        
    CNT_CHILDREN = st.slider('Количество детей:', 0, 10)
    FLAG_OWN_CAR =st.selectbox("Наличие автомобиля:", ['0', '1'])
    AMT_INCOME_TOTAL = st.slider('Годовой доход клиента:', 0,  500000)
    AMT_CREDIT = st.slider('Сумма кредита:', 0,  500000)
    AMT_GOODS_PRICE = st.slider('Стоимость товара, который необходимо приобрести:', 0,  500000)
    NAME_EDUCATION_TYPE = st.selectbox('Уровень образования: Примечание: 0 - базовое школьное образование, 1 - среднее/среднее специальное образование, 2 - неоконченное высшее образование, 3 - высшее образование, 4 - ученая степень.',['0', '1', '2', '3', '4'])
            
    NAME_INCOME_TYPE_Working= st.selectbox('Тип дохода: Рабочий',['0', '1'])
    NAME_INCOME_TYPE_State_servant= st.selectbox('Тип дохода: Госслужащий',['0', '1'])
    NAME_INCOME_TYPE_Commercial_associate= st.selectbox('Тип дохода: Специалист по коммерции',['0', '1']) 
    NAME_INCOME_TYPE_Pensioner= st.selectbox('Тип дохода: Пенсионер',['0', '1'])  
    NAME_INCOME_TYPE_Unemployed= st.selectbox('Тип дохода: Безработный',['0', '1'])
    NAME_INCOME_TYPE_Student= st.selectbox('Тип дохода: Студент',['0', '1'])
    NAME_INCOME_TYPE_Businessman= st.selectbox('Тип дохода: Бизнесмен',['0', '1'])
    NAME_INCOME_TYPE_Maternity_leave= st.selectbox('Тип дохода: В декретном отпуске',['0', '1']) 
    
    if int(NAME_INCOME_TYPE_Working) == 1: 
        (int(NAME_INCOME_TYPE_State_servant) & int(NAME_INCOME_TYPE_Commercial_associate) & int(NAME_INCOME_TYPE_Pensioner) & int(NAME_INCOME_TYPE_Unemployed) & int(NAME_INCOME_TYPE_Student) & int(NAME_INCOME_TYPE_Businessman) & int(NAME_INCOME_TYPE_Maternity_leave)) == 0
    elif NAME_INCOME_TYPE_State_servant == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Businessman & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Commercial_associate == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Businessman & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Pensioner == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Businessman & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Unemployed == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Businessman & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Student == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Businessman & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Businessman == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Maternity_leave) == 0
    elif NAME_INCOME_TYPE_Maternity_leave == 1:
        (NAME_INCOME_TYPE_Working & NAME_INCOME_TYPE_State_servant & NAME_INCOME_TYPE_Commercial_associate & NAME_INCOME_TYPE_Pensioner & NAME_INCOME_TYPE_Unemployed & NAME_INCOME_TYPE_Student & NAME_INCOME_TYPE_Businessman) == 0
    else:
        st.error('Некорректный ввод данных по типу дохода.')
    
    
    REGION_RATING_CLIENT= st.selectbox('Рейтинг региона проживания клиента: Примечание: 1 - Минск, 2 - областные центры, 3 - остальные населенные пункты.', ['1', '2', '3'])            
    REG_CITY_NOT_WORK_CITY = st.selectbox('Совпадает ли адрес клиента с адресом по прописке', ['0', '1']) 
                                          
    churn_html = """  
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;"> К сожалению, велика вероятность, что клиент уйдет в дефолт.</h2>
               </div>
            """
    no_churn_html = """  
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;"> Вероятно, кредит будет погашен!!!</h2>
               </div>
            """

    if st.button('Сделать прогноз'):
        output = predict_churn(CODE_GENDER_M, CODE_GENDER_F, CODE_GENDER_XNA, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,                
NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State_servant, NAME_INCOME_TYPE_Commercial_associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, 
NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman,  NAME_INCOME_TYPE_Maternity_leave, REG_CITY_NOT_WORK_CITY, REGION_RATING_CLIENT)
        st.success('Вероятность дефолта составляет {}'.format(output))
        #st.balloons()

    #CODE_GENDER_M, CODE_GENDER_F, CODE_GENDER_XNA, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,                
#NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State_servant, NAME_INCOME_TYPE_Commercial associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, 
#NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman,  NAME_INCOME_TYPE_Maternity_leave, REG_CITY_NOT_WORK_CITY, REGION_RATING_CLIENT
   

    
        if output >= 0.5:
            st.markdown(churn_html, unsafe_allow_html= True)

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)

if __name__=='__main__':
    main()
