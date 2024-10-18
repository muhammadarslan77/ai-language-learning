import streamlit as st
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Initialize translation pipeline
st.title("AI-Assisted Language Learning App")

st.sidebar.title("Features")
feature = st.sidebar.selectbox("Select a Feature", ["Text Translation", "AI Conversation"])

# Text Translation Feature
if feature == "Text Translation":
    st.subheader("Text Translation")
    source_language = st.selectbox("Select Source Language", ["English", "French", "Spanish"])
    target_language = st.selectbox("Select Target Language", ["French", "Spanish", "German"])

    translation_model = f"Helsinki-NLP/opus-mt-{source_language[:2]}-{target_language[:2]}"
    translator = pipeline("translation", model=translation_model)

    input_text = st.text_area("Enter text to translate:")
    if st.button("Translate"):
        translation = translator(input_text)[0]['translation_text']
        st.write("Translation:", translation)

# AI Conversation Feature
elif feature == "AI Conversation":
    st.subheader("AI Conversation Practice")
    st.write("Start a conversation in English with the AI and improve your language skills.")
    
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    
    user_input = st.text_input("You:")
    if st.button("Send"):
        inputs = tokenizer(user_input, return_tensors="pt")
        reply_ids = model.generate(inputs['input_ids'])
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        st.write("AI:", response)
