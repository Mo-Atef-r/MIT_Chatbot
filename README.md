# MIT EECS Course Helper Chatbot

An intelligent Retrieval Augmented Generation (RAG) chatbot designed to assist students with questions about MIT's Electrical Engineering and Computer Science (EECS) courses. This project provides accurate and context-aware answers by combining web scraping, semantic search, and a local Large Language Model (LLM).

##  Features

* **Comprehensive Course Information:** Answers questions on course codes, titles, descriptions, prerequisites, terms offered, instructors, and more.

* **Semantic Search:** Utilizes TF-IDF to find the most relevant courses based on natural language queries, ensuring highly pertinent information is provided to the LLM.

* **Retrieval Augmented Generation (RAG):** Enhances LLM responses by grounding them in real-time, external course data, minimizing hallucinations and improving factual accuracy.

* **Local LLM Integration:** Uses `Ollama` to run `Llama 3.2:3b` locally, allowing for privacy and offline functionality.

* **Streaming Responses:** Provides a smooth, real-time chat experience by streaming LLM responses in chunks.

* **User-Friendly Web Interface:** Built with Flask for an intuitive and responsive chat application.

##  Technologies Used

Python, Flask, Pandas, Scikit-learn, BeautifulSoup, Ollama (Llama 3.2:3b)

##  Data Source

The core data for this chatbot was obtained by web scraping the official **MIT Electrical Engineering and Computer Science (EECS) course catalog** from the MIT website.
