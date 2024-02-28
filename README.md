Github Repository Intepreter

"Chat with your codebase"

- Enter a valid URL of a repository or a subfolder or even a file
- Use your github API to retrieve files and metadata from the URL and store them locally
- Locally stored files are tokenized and vector embedded using Langchain and ChromaDB
- Using a query retrieve the top k chunks, feed it to an LLM, here in my code is OpenAI GPT 3.5 turbo
- Chat with your files : )

  Improvements pending:
    - Will try to use "git sparse-checkout" to get the files faster
    - Provide chat summary and metadata to the LLM
