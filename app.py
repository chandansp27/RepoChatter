from flask import Flask, request, jsonify, render_template, session
import scripts.functions as functions
import os
import utils
from utils import MODEL_NAME
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

app.secret_key = '1234567' 

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/chat-interface.html")
def chatbot_interface():
    default_message = "Enter your query"
    github_url = session.get('github_url', '')
    return render_template('chat-interface.html',default_message=default_message, github_url=github_url)

@app.route("/setup_chat", methods=["POST"])
def setup_chat():
    user_url = request.json.get('github_url') # type:ignore
    if not user_url:
        return jsonify({'error': 'GitHub URL is required'}), 400
    session['github_url'] = user_url
    try:
        info_dict, files_info, loaded_documents = functions.processRepository(user_url)
        if not all([info_dict, files_info, loaded_documents]):
            return jsonify({'error': 'Failed to process repository'}), 500
        global context
        context = functions.createContext(info_dict, files_info)
        embeddings_dir = utils.embeddings_directory
        embeddings_path = os.path.join(os.getcwd(), embeddings_dir)
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        embeddings = OpenAIEmbeddings(disallowed_special=())
        vectordb = Chroma.from_documents(loaded_documents, embedding=embeddings, persist_directory=embeddings_path)
        retriever = vectordb.as_retriever()
        global qa_chain
        qa_chain = RetrievalQA.from_chain_type(ChatOpenAI(temperature=0.2, model_name=MODEL_NAME), # type: ignore
                                                chain_type="stuff",
                                                retriever=retriever,
                                                return_source_documents=True)
        return jsonify({'message': 'Chat context created successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/process_chat_query", methods=["POST"])
def process_chat_query():
    user_question = request.json.get('user_question')  # type: ignore
    user_url = request.json.get('github_url')  # type: ignore
    print(f"User question received: {user_question}, URL: {user_url}")
    if not user_question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        print("QA Chain initialized, processing question")
        chat_response = functions.chatAI(qa_chain, context, user_question)
        response_text = chat_response['result']
        response_data = {'message': response_text}
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Error processing chat query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)