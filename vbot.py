import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.llm import OpenAI



class StreamlitResponse(ResponseParser):
    def __init__(self, context)-> None :

        super().__init__(context)

    def format_dataframe(self, result):  
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

# make the app wider
st.set_page_config(layout='wide')

# set the title
st.title ("  ChatBot : Prompt Based Data Analysis and Visualization ")
st.markdown('---') # to make line 

# file uploader using streamlit 
upload_csv_file = st.file_uploader("Upload Your CSV file for data analysis and visualization", type = ["csv"])

# if statement to make sure the data is uploaded
if upload_csv_file is not None:
    data = pd.read_csv(upload_csv_file)
    data.columns = data.columns.str.upper()   #convert the columns to uppercase 
    st.table(data.head(5))
    st.write(' Data Uploaded Successfully!')

st.markdown('---')

st.write ( '### Enter Your Analysis or Visualization Request')
query = st.text_area(" Enter your prompt")

# llm = OpenAI(api_token= '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')  # api key starts with something sk-...


if st.button ("Submit"):
    if query:
        with st.spinner("Loading wait.."):
            st.write ( '### OUTPUT : ')
            st.markdown('---')
            query_engine = SmartDataframe(data, config = {'llm':llm, "response_parser": StreamlitResponse})
            # query_engine = SmartDataframe(data, config = {'llm':llm})

            # answer = query_engine.chat(query)
            query_engine.chat(query)

            
            # st.write(answer)
    else:
        st.warning("Please enter a prompt") 