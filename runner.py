import streamlit as st
from transformers import pipeline
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer



@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model
 
tokenizer, model = load_data()

with st.sidebar:
    st.title("문장 감정 분석기")

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentiment_analysis_model():
    return pipeline("sentiment-analysis")

txt = st.text_area('Text to analyze', '''
     This is the best tasting energy bar I have ever had. My kids love them too. Great high energy snack.
     ''')

with st.spinner('Load Sentiment model...'):
    sentiment_pipeline = get_sentiment_analysis_model()

with st.spinner('Analyze sentiment....'):
    r = sentiment_pipeline(txt, truncation=True)

st.subheader('Sentiment')
df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])

# st.write(r[0])
st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))
