import streamlit as st
import transformers
#from transformers import pipeline
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel
import torch


with st.sidebar:
    st.title("감정 모델 기반의 챗봇 👾")

input = st.text_input('입력:')

if 'count' not in st.session_state or st.session_state.count == 6:
 st.session_state.count = 0 
 st.session_state.chat_history_ids = None
 st.session_state.old_response = ''
else:
 st.session_state.count += 1


tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

with torch.no_grad():
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
  
  
#st.subheader('감정 분석 결과')
#df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])

# st.write(r[0])
#st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))
