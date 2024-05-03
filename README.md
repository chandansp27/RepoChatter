## GitHub Repository ChatBot
"Chat with any GitHub codebase / own code"

- Enter a valid URL of a repository, subfolder, or even a file
- Use your GitHub API to retrieve files and metadata from the URL and store them locally
- Locally stored files are tokenized and vector-embedded using Langchain and ChromaDB
- Using a query retrieve the top k chunks, and feed it to an LLM, here in my code I'm using OpenAI GPT 3.5 turbo
- Chat with the files : )

#### Chat in Terminal
![Screenshot 2024-03-06 185142](https://github.com/chandansp27/RepositoryInterpreter/assets/72791595/8f734d77-877c-4212-ba02-ca18d0166b18)


#### Flask Application

![Screenshot 2024-03-06 180723](https://github.com/chandansp27/RepositoryInterpreter/assets/72791595/c976c700-ee67-41fa-a36a-23897dd94b64)


![Screenshot 2024-03-06 180842](https://github.com/chandansp27/RepositoryInterpreter/assets/72791595/c31c013f-ef46-4051-ad49-69075ff97885)


![Screenshot 2024-03-06 184200](https://github.com/chandansp27/RepositoryInterpreter/assets/72791595/958f1e1c-4a42-429b-b00c-b1548dde696a)


  Improvements pending:
    - Will try to use "git sparse-checkout" to get the files faster
    - Provide chat summary and metadata to the LLM
    - Provide usage of open-source LLMs HuggingFace API and Ollama
