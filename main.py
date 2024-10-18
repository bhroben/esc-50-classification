import streamlit as st
from demo import *
from streamlit_mic_recorder import mic_recorder

# Streamlit UI

# Streamlit UI
st.title('Demo for HDA Project - ESC 50 Audio Classification')

# File upload
uploaded_file = st.file_uploader("Choose a wav file", type="wav")

audio_rec = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)

# If a file is uploaded, import the data and plot the features
if uploaded_file is not None or audio_rec is not None:

    if uploaded_file is not None:
        data = st.audio(uploaded_file) 
        data = load_audio(audio=uploaded_file.name)

    if audio_rec is not None:
        data = st.audio(audio_rec["bytes"]) 
        #save data to wav
        with open('output_mic.wav', 'wb') as f:
            f.write(audio_rec["bytes"])

        data = load_audio(audio="output_mic.wav")

    spec=make_spec(data)

    mel_spec=plot_spec(spec)
    #show plot
    st.pyplot(mel_spec)

    df=classify_spec(spec)
    #show df
    st.write(df)
    #show plot
    figure=plot_bar(df)
    st.pyplot(figure)

else:
    st.write("Please select an audio file to classify or register one with the button.")


