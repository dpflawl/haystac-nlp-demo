import streamlit as st
from transformers import pipeline
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


with st.sidebar:
    st.title("챗봇 & 문장 감정 분석 서비스")

input = st.text_area('입력', '''
     안녕하세요. 반갑습니다.
     ''')

if 'count' not in st.session_state or st.session_state.count == 6:
 st.session_state.count = 0 
 st.session_state.chat_history_ids = None
 st.session_state.old_response = ''
else:
 st.session_state.count += 1


tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype=torch.float16, low_cpu_mem_usage=True)

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
