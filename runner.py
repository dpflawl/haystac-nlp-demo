import streamlit as st
from transformers import pipeline
import pandas as pd
import transformers
from transformers import AutoModelWithLMHead, AutoTokenizer


@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
 model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")
 return tokenizer, model
 
tokenizer, model = load_data()

with st.sidebar:
    st.title("챗봇 & 문장 감정 분석 서비스")

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentiment_analysis_model():
    return pipeline("sentiment-analysis")

input = st.text_area('Text to analyze', '''
     This is the best tasting energy bar I have ever had. My kids love them too. Great high energy snack.
     ''')

with st.spinner('Load Sentiment model...'):
    sentiment_pipeline = get_sentiment_analysis_model()

with st.spinner('Analyze sentiment....'):
    r = sentiment_pipeline(input, truncation=True)

  
if 'count' not in st.session_state or st.session_state.count == 6:
 st.session_state.count = 0 
 st.session_state.chat_history_ids = None
 st.session_state.old_response = ''
else:
 st.session_state.count += 1
new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids


st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)


response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if st.session_state.old_response == response:
  bot_input_ids = new_user_input_ids

  st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id) 

  response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

st.subheader('챗봇 답변')
st.write(f"Chatbot: {response}")
st.session_state.old_response = response 
  
  
st.subheader('감정 분석 결과')
df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])

# st.write(r[0])
st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))
