import sklearn

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
option = st.sidebar.selectbox('Какой алгоритм запустить?', classifier_name)
st.subheader(option)



#Importing model and label encoders
model=pickle.load(open("model.pkl","rb"))
#model_1 = pickle.load(open("final_rf_model.pkl","rb"))
#le_pik=pickle.load(open("label_encoding_for_gender.pkl","rb"))
#le1_pik=pickle.load(open("label_encoding_for_geo.pkl","rb"))

    
def predict_default(DAYS_EMPLOYED, CODE_GENDER_M, CODE_GENDER_F, DAYS_BIRTH, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State servant, 
NAME_INCOME_TYPE_Commercial associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman, 
NAME_INCOME_TYPE_Maternity leave, REGION_RATING_CLIENT):
    input = np.array([[DAYS_EMPLOYED, CODE_GENDER_M, CODE_GENDER_F, DAYS_BIRTH, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State servant, 
NAME_INCOME_TYPE_Commercial associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman, 
NAME_INCOME_TYPE_Maternity leave, REGION_RATING_CLIENT]]).astype(np.float64)
    if option == 'LogisticRegression':
        prediction = nastya.predict_proba(x_test)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)

    else:
        pred=0.40
        #st.markdown('Вероятно, кредит будет погашен.')

    return float(pred)


def main():
    st.title("Прогноз вероятности дефолта клиента")
    html_temp = """
    <div style="background-color:yellow ;padding:10px">
    <h2 style="color:red;text-align:center;">Заполни форму</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True


    st.sidebar.subheader("Приложение создано для курса Diving into Darkness of Data Science")
    st.sidebar.text("Разработчик - Каравай А.Л.")

    DAYS_EMPLOYED = st.slider('Стаж работы (в днях)', -17000, 0)           
      
    CODE_GENDER_M =st.selectbox("Женщина", ['0', '1']) 
    CODE_GENDER_F =st.selectbox("Мужчина", ['0', '1'])
    DAYS_BIRTH = st.slider('Возраст клиента (в днях)', -25000, 0)            
    #NAME_EDUCATION_TYPE = st.selectbox('Уровень образования', ['Lower secondary' : 0, 'Secondary / secondary special' : 1,
'Incomplete higher' : 2, 'Higher education' : 3, 'Academic degree' : 4])
    NAME_INCOME_TYPE_Working= st.selectbox('Тип дохода: Рабочий',['0', '1'])
    NAME_INCOME_TYPE_State servant= st.selectbox('Тип дохода: Госслужащий',['0', '1'])
    NAME_INCOME_TYPE_Commercial associate= st.selectbox('Тип дохода: Специалист по коммерции',['0', '1'])  
    NAME_INCOME_TYPE_Pensioner= st.selectbox('Тип дохода: Пенсионер',['0', '1'])  
    NAME_INCOME_TYPE_Unemployed= st.selectbox('Тип дохода: Безработный',['0', '1'])  
    NAME_INCOME_TYPE_Student= st.selectbox('Тип дохода: Студент',['0', '1']) 
    NAME_INCOME_TYPE_Businessman= st.selectbox('Тип дохода: Бизнесмен',['0', '1']) 
    NAME_INCOME_TYPE_Maternity leave= st.selectbox('Тип дохода: В декретном отпуске',['0', '1']) 
    REGION_RATING_CLIENT= st.selectbox('Рейтинг региона проживания клиента',['1', '2', '3'])            
          


    default_html = """  
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;"> К сожалению, велика вероятность, что клиент уйдет в дефолт.</h2>
               </div>
            """
    no_default_html = """  
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;"> Вероятно, кредит будет погашен!!!</h2>
               </div>
            """

    if st.button('Сделать прогноз'):
        output = predict_default(DAYS_EMPLOYED, CODE_GENDER_M, CODE_GENDER_F, DAYS_BIRTH, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE_Working, NAME_INCOME_TYPE_State servant, 
NAME_INCOME_TYPE_Commercial associate, NAME_INCOME_TYPE_Pensioner, NAME_INCOME_TYPE_Unemployed, NAME_INCOME_TYPE_Student, NAME_INCOME_TYPE_Businessman, 
NAME_INCOME_TYPE_Maternity leave, REGION_RATING_CLIENT)
        st.success('Вероятность дефолта составляет {}'.format(output))
        st.balloons()

        if output >= 0.5:
            st.markdown(default_html, unsafe_allow_html= True)

        else:
            st.markdown(no_default_html, unsafe_allow_html= True)

if __name__=='__main__':
    main()
