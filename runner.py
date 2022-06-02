import streamlit as st
import transformers
#from transformers import pipeline
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel
import torch


with st.sidebar:
    st.title("감정 모델 기반의 챗봇과 대화해보세요. 👾")
    
def get_text():
    input_text = st.text_input("You: ","안녕하세요, 반가워요.", key="input")
    return input_text 

use_input = get_text()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if user_input:
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
      pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    with torch.no_grad():
        new_user_input_ids = tokenizer.encode(use_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids
        st.session_state.chat_history_ids = gen_ids = model.generate(bot_input_ids,
                                                                    max_length=128,
                                                                    repetition_penalty=2.0,
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    eos_token_id=tokenizer.eos_token_id,
                                                                    bos_token_id=tokenizer.bos_token_id,
                                                                    use_cache=True)
        response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)       
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.write(st.session_state["generated"][i], key=str(i))
        st.write(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

#st.subheader('챗봇 답변')
#st.write(f"Chatbot: {response}")
#st.session_state.old_response = response 
  
  
#st.subheader('감정 분석 결과')
#df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])

# st.write(r[0])
#st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))
