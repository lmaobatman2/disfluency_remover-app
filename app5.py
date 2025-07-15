import streamlit as st
import os
import assemblyai as aai
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS  # Fixed import
import numpy as np
from openai import OpenAI

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
def load_model():
    return ChatterboxTTS.from_pretrained(device="cpu")

st.title("Speech Disfluency Remover")
st.write("This app removes speech disfluencies from audio files.")
st.write("Upload an audio file to get started.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload audio file", 
    type=["wav", "m4a", "mp3"],  # Fixed: use 'type' instead of second parameter
    accept_multiple_files=False
)

# AssemblyAI setup
aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)




import re

def split_into_chunks(text, sentences_per_chunk=5):
    # Split on '.', '!', '?'
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks
















def clean_transcript(transcript_text):
    prompt = f"""
You are a speech disfluency remover. The following transcript contains filler words, repeated words, and speech disfluencies.

Your task:
- Only remove repetitions of words (e.g. "I, I think" -> "I think")
- Remove filler words such as "um", "uh", "like", "you know", "er", "ah"
- Do not change any nouns, adjectives, or the meaning of the sentence.
- Do not paraphrase or rewrite. Only clean disfluencies.
-fix grammar issues if they are related to disfluencies, like missing articles or prepositions that are part of disfluencies.
-do fix punctuation to make the text more readable, adjust punctuation to bring additional clarity and impact to the cleaned transcript.
-do not use synonyms for any particular word
Here is the transcript:

{transcript_text}

Provide only the cleaned transcript.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise text cleaner."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    cleaned_text = response.choices[0].message.content.strip()
    return cleaned_text





def edit_transcript(cleaned_transcript):
    # Initialize session state if not already set
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    if 'final_transcript' not in st.session_state:
        st.session_state.final_transcript = cleaned_transcript
   # Buttons for editing
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Edit Transcript"):
            st.session_state.edit_mode = True
    with col2:
        if st.button("✅ Use Cleaned Transcript"):
            st.session_state.edit_mode = False
            st.session_state.final_transcript = cleaned_transcript

    # Editing interface
    if st.session_state.edit_mode:
        edited_text = st.text_area("Edit your transcript:", value=st.session_state.final_transcript, height=200)
        save_col, cancel_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Changes"):
                st.session_state.final_transcript = edited_text
                st.session_state.edit_mode = False
                st.success("Changes saved!")
                st.rerun()
        with cancel_col:
            if st.button("❌ Cancel"):
               st.session_state.edit_mode = False
               st.rerun()

    # Display final transcript
    st.write("### Final Transcript to be used:")
    st.write(st.session_state.final_transcript)

    return st.session_state.final_transcript





def generate_audio_chunk(final_transcript_chunk, temp_file_path, chunk_index):
    model = load_model()
    wav = model.generate(
        final_transcript_chunk,
        exaggeration=0.7,
        cfg_weight=0.4,
        audio_prompt_path=temp_file_path
    )
    audio_path = f"chunk_{chunk_index}.wav"
    ta.save(audio_path, wav.cpu(), model.sr)
    return audio_path


import torch


def generate_all_chunks_and_concatenate(final_transcript, temp_file_path):
    # Split the final transcript into chunks of 4 sentences each
    chunks = split_into_chunks(final_transcript, sentences_per_chunk=5)
    audio_paths = []

    st.write(f"🔍 Number of chunks to process: {len(chunks)}")

    # Generate audio for each chunk and save file path
    for idx, chunk in enumerate(chunks):
        st.write(f"🔊 Generating audio for chunk {idx+1}/{len(chunks)}...")
        path = generate_audio_chunk(chunk, temp_file_path, idx)
        audio_paths.append(path)

    if not audio_paths:
        st.error("❌ No audio chunks were generated. Please check your input transcript.")
        return

    # Load all audio chunks and concatenate using torch.cat
    combined_audio_list = []
    sample_rate = None

    for path in audio_paths:
        audio, sr = ta.load(path)
        combined_audio_list.append(audio)
        sample_rate = sr  # set sample rate from first file

    combined_audio = torch.cat(combined_audio_list, dim=1)  # concatenate along time dimension

    # Save combined audio to final file
    output_path = "final_combined_audio.wav"
    ta.save(output_path, combined_audio, sample_rate)

    # Display combined audio in Streamlit
    st.write("✅ **Final combined audio generated:**")
    st.audio(output_path, format="audio/wav", sample_rate=sample_rate)

    # Provide download button
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Download Final Audio",
            data=f.read(),
            file_name="final_speech.wav",
            mime="audio/wav"
        )

    # Cleanup temporary files
    for path in audio_paths:
        os.remove(path)
    if os.path.exists(output_path):
        os.remove(output_path)








def generate_audio1(final_transcript, temp_file_path):
    model = load_model()
    wav = model.generate(
                final_transcript,
                exaggeration=0.88,
                cfg_weight=0.4,
                audio_prompt_path=temp_file_path
    )
    if hasattr(wav, 'numpy'):
        audio_array = wav.numpy()
    elif hasattr(wav, 'cpu'):
        audio_array = wav.cpu().numpy()
    else:
        audio_array = np.array(wav)
    
    if len(audio_array.shape) > 1:
        audio_array = audio_array.squeeze()
    
    st.write("✅ **Clean Audio Generated:**")
    st.audio(audio_array, format="audio/wav", sample_rate=model.sr)
    
    clean_audio_path = "clean_audio.wav"
    ta.save(clean_audio_path, wav.cpu(), model.sr)
    
    with open(clean_audio_path, "rb") as f:
        st.download_button(
            label="📥 Download Clean Audio",
            data=f.read(),
            file_name="clean_speech.wav",
            mime="audio/wav"
        )
    
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    if os.path.exists(clean_audio_path):
        os.remove(clean_audio_path)







def edit_transcript1(cleaned_transcript):
    """
    Function to handle editing of the transcript with Save and Cancel options.
    Returns the current (possibly edited) transcript.
    """

    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False

    if 'final_transcript' not in st.session_state:
        st.session_state.final_transcript = cleaned_transcript

    with st.expander("📝 Final Transcript (Click to Edit)"):
        if st.session_state.edit_mode:
            edited_text = st.text_area(
                "Edit your transcript:",
                value=st.session_state.final_transcript,
                height=200
            )
            save_col, cancel_col = st.columns(2)
            with save_col:
                if st.button("💾 Save Changes"):
                    st.session_state.final_transcript = edited_text
                    st.session_state.edit_mode = False
                    st.success("Changes saved!")
            with cancel_col:
                if st.button("❌ Cancel"):
                    st.session_state.edit_mode = False
        else:
            st.write(st.session_state.final_transcript)
            if st.button("✏️ Edit"):
                st.session_state.edit_mode = True

    return st.session_state.final_transcript

def edit_transcript2(cleaned_transcript):
    with st.expander("📝 Final Transcript (Click to Edit)"):
        txt_area = st.text_area("Edit your transcript:", cleaned_transcript, height=200)
        if st.button("💾 Save Changes"):
            st.success("Changes saved!")
            return txt_area
        elif st.button("❌ Cancel"):
            return cleaned_transcript
        else:
            return txt_area

def edit_transcript3(cleaned_transcript):
    if 'final_transcript' not in st.session_state:
        st.session_state.final_transcript = cleaned_transcript

    txt_area = st.text_area("Edit your transcript:", value=st.session_state.final_transcript, height=200, key="edit_box")

    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("💾 Save Changes"):
            st.session_state.final_transcript = txt_area
            st.success("Changes saved!")

    with cancel_col:
        if st.button("❌ Cancel"):
            st.session_state.final_transcript = cleaned_transcript
            st.info("Changes reverted.")

    return st.session_state.final_transcript
















if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = "temp_prompt.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("🎵 Audio uploaded successfully!")
    st.audio(uploaded_file, format="audio/wav")
    
    # Transcribe the audio
    st.write("🔄 Transcribing audio...")
    try:
        # Use the file path, not the uploaded_file object
        transcript = transcriber.transcribe(temp_file_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Transcription failed: {transcript.error}")
        else:
            #if(st.button("📖 Show original Transcript")):
            #    st.write("📝 **Transcription:**")
            #    st.write(transcript.text)
            expander = st.expander("original transcript")
            expander.write("📝 **Transcription:**")
            expander.write(transcript.text)               
            cleaned_transcript=clean_transcript(transcript.text)
            final_transcript = edit_transcript(cleaned_transcript)
            
            if st.button("🎙️ Generate Clean Audio"):
                st.warning("⚠️ Processing may take a couple of minutes depending on audio length. Please be patient.")
                generate_all_chunks_and_concatenate(final_transcript, temp_file_path)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a different audio file.")
else:
    st.info("👆 Please upload an audio file to begin processing.")



#if the screen shows blank on streamlit, save the file and run it again.
