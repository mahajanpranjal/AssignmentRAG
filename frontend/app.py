import streamlit as st
import requests
import logging
import os
import json


def safe_json_response(response):
    """Safely parse JSON response and handle errors."""
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {
            "error": "Invalid JSON response received from backend.",
            "status_code": response.status_code,
            "text": response.text[:500],
        }

# FastAPI backend URL
FASTAPI_URL = "https://assignmentrag.onrender.com"


if "process_success" not in st.session_state:
    st.session_state.process_success = False
if "vector_db_choice" not in st.session_state:
    st.session_state.vector_db_choice = "pinecone"
if "quarter_choice" not in st.session_state:
    st.session_state.quarter_choice = "first"
    
st.title("📄 AI Document Processing - RAG Pipeline")
tab_upload, tab_chat = st.tabs(["Upload & Process", "Chat"])

# File Upload Section
with tab_upload:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        parser_option = st.selectbox("Select Parsing Method", ["pymupdf", "docling", "mistral_ocr"], key="parser_selectbox")
        

        if st.button("Upload and Process PDF"):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"parser": parser_option}
                response = requests.post(f"{FASTAPI_URL}/process_pdf", files=files, data=data)
                try:
                    response_data = response.json()
                except requests.exceptions.JSONDecodeError:
                    st.error(f"Error: Unable to parse JSON. Response: {response.text}")
                else:
                    if response.status_code == 200:
                        st.success("File uploaded and processed successfully!")
                        st.session_state.uploaded_filename = response_data["filename"]
                    else:
                        st.error(f"Error: {response_data.get('detail', 'Unknown error')}")
            else:
                st.warning("Please upload a file first.")

        if "files" not in st.session_state:
            st.session_state.files = []
        if "selected_file" not in st.session_state:
            st.session_state.selected_file = None

        st.header("📂 Available Processed Files")
        if st.button("Refresh File List"):
            response = requests.get(f"{FASTAPI_URL}/files")
            if response.status_code == 200:
                st.session_state.files = response.json()["files"]
                if st.session_state.files:
                    st.session_state.selected_file = st.selectbox("Select a file to process", st.session_state.files, key="file_selectbox_upload", index=0)
                else:
                    st.warning("No processed files available.")
            else:
                st.error("Failed to fetch files.")

        st.header("🔍 Process File for RAG")
        chunking_strategy = st.selectbox("Select Chunking Strategy", ["Recursive", "Token", "Semantics"], key="chunking_selectbox")
        vector_db = st.selectbox("Select Vector Database", ["chromadb", "pinecone"], key="vector_db_selectbox")
        quarter = st.selectbox("Select Quarter", ["first", "second", "third", "fourth"], key="quarter_selectbox")

        if st.session_state.files:
            st.session_state.selected_file = st.selectbox("Select a file to process", st.session_state.files, key="file_selectbox_process", index=st.session_state.files.index(st.session_state.selected_file) if st.session_state.selected_file in st.session_state.files else 0)

        if st.button("Process File"):
            if not hasattr(st.session_state, "uploaded_filename"):
                st.warning("Please upload a file first.")
            elif st.session_state.selected_file:
                payload = {
                    "filename": st.session_state.selected_file,  
                    "chunking_strategy": chunking_strategy,
                    "vector_db": vector_db,
                    "quarter": quarter 
                }

                headers = {'Content-Type': 'application/json'} 
                response = requests.post(f"{FASTAPI_URL}/process_chunks", data=json.dumps(payload), headers=headers)

                response_data = safe_json_response(response)
                if "error" in response_data:
                    st.error(f"Error: {response_data['error']} (Status Code: {response_data['status_code']})")
                    st.write("Response Text:", response_data["text"])
                else:
                    st.success("File processed and embeddings stored successfully!")
                    st.session_state.process_success = True
                    st.session_state.vector_db_choice = vector_db
                    st.session_state.quarter_choice = quarter
            else:
                st.warning("Please select a file first.")

        if st.session_state.process_success:
            st.header("📝 Ask a Question About Your Document")
            doc_query = st.text_input("Enter your question about the document:", key="doc_query_input")
            
            chat_col1, chat_col2 = st.columns([1, 1])
            
            with chat_col1:
                use_vector_db = st.selectbox(
                    "Vector Database for Query", 
                    ["chromadb", "pinecone"],
                    index=0 if st.session_state.vector_db_choice == "chromadb" else 1,
                    key="inline_vector_db"
                )
            
            with chat_col2:
                quarters_list = ["first", "second", "third", "fourth"]
                default_quarter = [st.session_state.quarter_choice] if st.session_state.quarter_choice in quarters_list else ["first"]
                use_quarters = st.multiselect(
                    "Quarters to Search", 
                    quarters_list,
                    default=default_quarter,
                    key="inline_quarters"
                )
            
            if st.button("Get Answer", key="inline_answer_button"):
                if not doc_query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Searching for an answer..."):
                        if not use_quarters:
                            use_quarters = ["first"]
                        
                        query_payload = {
                            "question": doc_query,
                            "vector_db": use_vector_db,
                            "quarters": use_quarters
                        }
                        
                        try:
                            query_response = requests.post(
                                f"{FASTAPI_URL}/query", 
                                json=query_payload,
                                headers={'Content-Type': 'application/json'}
                            )
                            
                            if query_response.status_code == 200:
                                query_result = query_response.json()
                                
                                st.subheader("Answer:")
                                st.write(query_result["answer"])
                                
                                if "context" in query_result and query_result["context"]:
                                    with st.expander("View Source Context"):
                                        for i, ctx in enumerate(query_result["context"]):
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(ctx)
                            else:
                                st.error(f"Failed to get answer. Status code: {query_response.status_code}")
                                st.text(query_response.text)
                        except Exception as e:
                            st.error(f"Error querying the document: {e}")

with tab_chat:
    st.subheader("Ask a Financial Question")

    query = st.text_input("Enter your question (e.g. 'Revenue of 2024 Q1')", key="chat_query")
    year = st.selectbox("Select Year", ["2021", "2022", "2023", "2024", "2025"])
    quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": query,
                "year": year,
                "quarter": quarter
            }

            response = requests.post(f"{FASTAPI_URL}/generate_answer", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer:")
                st.markdown(result["answer"])
            else:
                st.error("Failed to fetch answer from backend.")
