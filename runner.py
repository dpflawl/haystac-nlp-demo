import streamlit as st
from transformers import pipeline
import pandas as pd

with st.sidebar:
    st.image('logo.png')
    # st.title("Haystac NLP Demos")
    st.subheader("Pipelines:")


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentiment_analysis_model():
    return pipeline("sentiment-analysis")


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_summarization_model():
    return pipeline("summarization")

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_zero_shot_classification_model():
    return pipeline("zero-shot-classification")

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
