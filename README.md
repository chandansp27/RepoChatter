** Github Repository Interpreter ** 

"Chat with any GitHub codebase / own code"

- Enter a valid URL of a repository, subfolder, or even a file
- Use your GitHub API to retrieve files and metadata from the URL and store them locally
- Locally stored files are tokenized and vector-embedded using Langchain and ChromaDB
- Using a query retrieve the top k chunks, and feed it to an LLM, here in my code I'm using OpenAI GPT 3.5 turbo
- Chat with the files : )

  Improvements pending:
    - Will try to use "git sparse-checkout" to get the files faster
    - Provide chat summary and metadata to the LLM
