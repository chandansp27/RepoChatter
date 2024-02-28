import os
import utils
import requests
import config
from utils import WHITE, GREEN, RESET_COLOR, MODEL_NAME
import uuid
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import warnings
import sys
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", message="Number of requested results .*")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module="langchain")

# error handler
def custom_warning_handler(*args, **kwargs):
    pass
warnings.showwarning = custom_warning_handler


# get api token from .env (github API and OpenAI API)
github_access_token = config.GITHUB_ACCESS_TOKEN
os.environ["OPENAI_API_KEY"] = config.OPENAI_ACCESS_TOKEN

# format the user url for API request
def parseAndFormatURL(user_url):
    parts = user_url.split("/")
    N = len(parts)
    idx = parts.index("github.com")

    url_info = {'username': None,
                'repository': None,
                'blob': None,
                'branch/tag/commit': None,
                'other_url': None}

    if idx + 1 == N-1:
        url_info['username'] = parts[idx+1]
        return url_info

    for key in url_info.keys():
        idx += 1
        if idx < N:
            url_info[key] = parts[idx]

    if idx < N-1:
        other_url = url_info['other_url']
        if other_url:
            other_url += '/'
            other_url += '/'.join(parts[idx+1:])
            url_info['other_url'] = other_url

    
    user_repos_url = "https://api.github.com/users/{username}/repos"
    repo_contents_url = "https://api.github.com/repos/{username}/{repository}/contents/{other_url}"
    repo_base_url = "https://api.github.com/repos/{username}/{repository}/contents"

    username = url_info.get('username')
    repository = url_info.get('repository')
    other_url = url_info.get('other_url')
    if username:
        if repository:
            if other_url:
                return repo_contents_url.format(username=username, repository=repository, other_url=other_url), url_info if url_info else {}
            else:
                return repo_base_url.format(username=username, repository=repository), url_info if url_info else {}
        else:
            return user_repos_url.format(username=username), url_info if url_info else {}
    else:
        return None

# request the api_url to fetch data using the github URL 
def getRequest(api_url):
    headers = {
        'Authorization': f'token {github_access_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print('Error requesting data from GitHub API:', e)
        return None, None


# downloads user metadata and files from the API request to a local folder
def downloadFiles(json_output, url, local_folder):
    downloaded_files = []
    base_folder_name = url.split("/")[-1] if url.split('/')[-1] != 'contents' else url.split("/")[-2]
    base_folder_path = os.path.join(os.getcwd() + local_folder, base_folder_name)
    os.makedirs(base_folder_path, exist_ok=True)
    allowed_filetypes = utils.download_file_types

    if type(json_output) == dict:
        filename = json_output['name']
        if '.' + json_output['name'].split('.')[-1]:
                response = requests.get(json_output['download_url'])

                if response.status_code == 200:
                    with open(os.path.join(base_folder_path, filename), 'wb') as file:
                        file.write(response.content)
                        downloaded_files.append(os.path.join(base_folder_path, filename))
                else:
                    print(f'Error downloading {filename}, status code: {response.status_code}')
        return base_folder_path

    for item in json_output:
        if item['type'] == 'file':
            filename = item['name']
            if any(filename.endswith(ftype) for ftype in allowed_filetypes):
                response = requests.get(item['download_url'])
                if response.status_code == 200:
                    with open(os.path.join(base_folder_path, filename), 'wb') as file:
                        file.write(response.content)
                        downloaded_files.append(os.path.join(base_folder_path, filename))
                else:
                    print(f'Error downloading {filename}, status code: {response.status_code}')

        elif item['type'] == 'dir':
            new_url = item['url']
            new_dir_path = os.path.join(base_folder_path, item['name'])
            os.makedirs(new_dir_path, exist_ok=True)
            new_json_output = getRequest(new_url)
            downloadFiles(new_json_output, new_dir_path, local_folder)
    return base_folder_path


# take in the URL from the user
user_url = input('Enter repository/subfolder/file URL: ')

api_url, info_dict = parseAndFormatURL(str(user_url)) # type: ignore

if api_url:
    json_output = getRequest(api_url)
else:
    raise Exception('Invalid URL, enter a valid URL link')
local_repo_folder = utils.local_repository_cache_folder

if json_output:
    print('DOWNLOADING FILES')
    repo_download_path = downloadFiles(json_output, api_url, local_repo_folder)
else:
    raise Exception('Local Files download failed')

# load downloaded files using Langchain
def loadDocuments(repo_download_path):
    documents_dict = {}
    files_info = []
    loader = DirectoryLoader(repo_download_path, use_multithreading=True, show_progress=False)
    try:
        loaded_documents = loader.load() if callable(loader.load) else []
        if loaded_documents:
            for doc in loaded_documents:
                file_path = doc.metadata['source']
                relative_path = os.path.relpath(file_path, repo_download_path)
                file_id = str(uuid.uuid4())
                doc.metadata['source'] = relative_path
                doc.metadata['file_id'] = file_id
                documents_dict[file_id] = doc
    except Exception as e:
        raise Exception('Document loading failed')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 300)
    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']
        split_documents.extend(split_docs)
    return split_documents if split_documents else None, files_info if files_info else None

def delete_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

if repo_download_path:
    print('LOADING FILES')
    loaded_documents, files_info = loadDocuments(repo_download_path)
else:
    raise Exception('Loading files into Langchain Failed')

username_info = info_dict['username'] if info_dict['username'] else None # type: ignore
repository_info = info_dict['repository'] if info_dict['repository'] else None # type: ignore
idx_limit = len(loaded_documents) # type: ignore

embeddings_dir = utils.embeddings_directory
embeddings_path = os.path.join(os.getcwd(), embeddings_dir)
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
embeddings = OpenAIEmbeddings(disallowed_special=())
vectordb = Chroma.from_documents(loaded_documents, embedding=embeddings, persist_directory=embeddings_path)
query = f'{username_info},{repository_info},{files_info}'
retriever = vectordb.as_retriever()

# supressing a warning, related to a bug in the current version of langchain
# Redirect stdout and stderr to suppress all output including warnings
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

relevant_chunks = retriever.get_relevant_documents(query ,search_kwargs={'k':1})

# Restore stdout and stderr to default
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

qa_chain = RetrievalQA.from_chain_type(ChatOpenAI(temperature=0.2,model_name=MODEL_NAME), # type: ignore
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True)

def chat_loop():
    while True:
        question = input("\n" + WHITE + "Ask a question (or type 'exit()' to quit): " + RESET_COLOR)
        if question.lower().strip() == 'exit()':
            break
        print(WHITE + 'Thinking...' + RESET_COLOR)
        result = qa_chain(question)
        print(GREEN + "\nAnswer: " + result['result'] + RESET_COLOR)
    vectordb.delete_collection()
    delete_files_in_folder(local_repo_folder)
chat_loop()

