import streamlit as st
from utills.retrieval import retrieve_top_k,load_faiss_index
from utills.prompt import build_prompt
from utills.completion import generate_completion


st.title("RAG App Project")
st.write("Ask questions grounded in the life of Kolawole")

query = st.text_input("Enter your question here")

if query:
    index, chunk_map = load_faiss_index()
    top_chunks = retrieve_top_k(query, index, chunk_map)
    prompt = build_prompt(top_chunks, query)
    response = generate_completion(prompt)   # pass the prompt, not query

    st.subheader("Answer")
    st.write(response)

    with st.expander("Retrieved Chunks"):
        for chunk in top_chunks:
            st.markdown(f"Chunk: {chunk}")
