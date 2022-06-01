import streamlit as st
from transformers import pipeline
import pandas as pd

with st.sidebar:
    st.image('logo.png')
    # st.title("Haystac NLP Demos")
    st.subheader("Pipelines:")
    sentiment_analysis_flag = st.checkbox("Sentiment Analysis", False)
    zero_shot_classification_flag = st.checkbox("Zero Shot Classification", False)
    summarization_flag = st.checkbox("Summarization", True)

    if zero_shot_classification_flag:
        with st.expander("Zero Shot Classification options:"):
            categories = st.text_input("Categories:","politics, sport, reviews")
            categories_list = categories.split(',')
            categories_list = [e.strip() for e in categories_list]
            print(f'Categories: {categories_list}')
            mutli_label = st.checkbox("MultiLabel", False)


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

if sentiment_analysis_flag:
    with st.spinner('Load Sentiment model...'):
        sentiment_pipeline = get_sentiment_analysis_model()

    with st.spinner('Analyze sentiment....'):
        r = sentiment_pipeline(txt, truncation=True)

    st.subheader('Sentiment')
    df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])


    # st.write(r[0])
    st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))

if zero_shot_classification_flag:
    with st.spinner('Load Classification model...'):
        zero_shot_classification_pipeline = get_zero_shot_classification_model()

    with st.spinner('Classification....'):
        r = zero_shot_classification_pipeline(txt, categories_list, multi_label=mutli_label)

    st.subheader('Categories:')
    labels = r['labels']
    scores = r['scores']
    c_df = pd.DataFrame.from_dict({'Categories': labels, 'Scores': scores})

    st.table(c_df.style.background_gradient(cmap='RdYlGn', subset='Scores', vmin=0., vmax=1.))

if summarization_flag:
    with st.spinner('Load Summarization model...'):
        summarization_pipeline = get_summarization_model()

    with st.spinner('Generate Summarization....'):
        r = summarization_pipeline(txt)

    st.subheader('Summary')
    st.write(r[0]['summary_text'])
