# app.py
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from datetime import datetime
from time import perf_counter 
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import json
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

DATA_PATH= 'D:/Data_and_AI/Projects/Chatbots/MIT LLM/data/Cleaned_EECE.csv'
OLLAMA_MODEL= "llama3.2:3b"



class MITCourseRAG:
    def __init__(self, csv_file_path: str, ollama_model: str = OLLAMA_MODEL, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the MIT Course RAG system with LLM agent
        
        Args:
            csv_file_path: Path to the CSV file containing course data
            ollama_model: Ollama model to use (llama2, llama3.2:1b, llama3.2:3b)
            ollama_url: Ollama API URL
        """
        self.df = pd.read_csv(csv_file_path)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self._preprocess_data()
        self._create_search_index()
        
    def _preprocess_data(self):
        """Preprocess the course data for better search"""
        # Fill NaN values
        text_columns = ['title', 'prereq', 'terms', 'description', 'instructor', 'section_head']
        for col in text_columns:
            self.df[col] = self.df[col].fillna('')
        
        self.df['hours'] = self.df['hours'].fillna('Not specified')
        self.df['optional'] = self.df['optional'].fillna('Not specified')
        self.df['offered_to'] = self.df['offered_to'].fillna('Not specified')
        
        # Create searchable text
        self.df['search_text'] = (
            self.df['code'] + ' ' +
            self.df['title'] + ' ' +
            self.df['description'] + ' ' +
            self.df['section_head'] + ' ' +
            self.df['instructor'] + ' ' +
            self.df['prereq']
        )
        
    def _create_search_index(self):
        """Create TF-IDF index for semantic search"""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
        
    def _call_ollama(self, prompt: str):
        """
        Call Ollama API with the given prompt and stream the output in chunks.
        This function is a generator that yields parts of the response as they are received.
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True  # Set to True to enable streaming
                },
                stream=True  # Important: This tells requests to keep the connection open for streaming
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            # Iterate over the response content line by line
            # Ollama's streaming API sends JSON objects, one per line
            for line in response.iter_lines():
                if line:  # Ensure the line is not empty
                    try:
                        json_chunk = json.loads(line.decode('utf-8'))
                        # The actual text content is in the 'response' field of each chunk
                        if "response" in json_chunk:
                            yield json_chunk["response"]
                        # The 'done' field indicates the end of the stream
                        if json_chunk.get("done"):
                            break  # Exit the loop when the stream is complete
                    except json.JSONDecodeError:
                        # Handle cases where a line might not be a complete or valid JSON object
                        # This can happen if the stream is malformed or interrupted
                        print(f"Warning: Could not decode JSON from line: {line.decode('utf-8')}")
                        continue
        except requests.exceptions.ConnectionError:
            # Yield connection error message if Ollama server is not reachable
            yield "Error: Could not connect to Ollama server. Make sure Ollama is running."
        except requests.exceptions.RequestException as e:
            # Yield other request-related errors
            yield f"An error occurred during the request to Ollama: {str(e)}"
        except Exception as e:
            # Catch any other unexpected errors
            yield f"An unexpected error occurred: {str(e)}"
    
    def _extract_course_codes(self, query: str) -> List[str]:
        """Extract all course codes from query"""
        pattern = r'6\.[A-Z0-9]+[A-Z]?'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return [match.upper() for match in matches]
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform semantic search using TF-IDF"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.05]
    
    def _retrieve_relevant_courses(self, query: str) -> List[Dict]:
        """Retrieve relevant courses based on query"""
        relevant_courses = []
        
        # First, check for specific course codes
        course_codes = self._extract_course_codes(query)
        
        if course_codes:
            for code in course_codes:
                matches = self.df[self.df['code'].str.contains(code, case=False, na=False)]
                for _, course in matches.iterrows():
                    relevant_courses.append(course.to_dict())
        
        # If no specific codes or we want more context, do semantic search
        if not relevant_courses or len(relevant_courses) < 3:
            similar_courses = self._semantic_search(query, top_k=5)
            for idx, similarity in similar_courses:
                course = self.df.iloc[idx].to_dict()
                if course not in relevant_courses:
                    relevant_courses.append(course)
        
        return relevant_courses[:5]  # Limit to top 5 for context window
    
    def _format_course_context(self, courses: List[Dict]) -> str:
        """Format course information for LLM context"""
        context = "Here are the relevant course details:\n\n"
        
        for course in courses:
            context += f"Course Code: {course['code']}\n"
            context += f"Title: {course['title']}\n"
            context += f"Credits/Hours: {course['hours']}\n"
            context += f"Category: {course['section_head']}\n"
            context += f"Prerequisites: {course['prereq']}\n"
            context += f"Terms: {course['terms']}\n"
            context += f"Instructor: {course['instructor']}\n"
            context += f"Offered to: {course['offered_to']}\n"
            context += f"Description: {course['description']}\n"
            context += f"Optional: {course['optional']}\n"
            context += "-" * 50 + "\n"
        
        return context
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM agent"""
        return """You are an intelligent assistant specializing in MIT EECE (Electrical Engineering and Computer Science) courses. 

Your job is to answer questions about courses using the provided course data. You should:

1. Answer questions naturally and conversationally
2. Be precise and accurate with the information
3. If a course code is mentioned, focus on that specific course
4. If multiple versions of a course exist (grad/undergrad), mention both if relevant
5. If information is not available or empty, say so clearly
6. For general questions, provide relevant course suggestions
7. Always be helpful and informative

Course data columns explanation:
- code: Course code (e.g., 6.100A)
- title: Course title
- prereq: Prerequisites (empty if none)
- terms: When the course is offered
- hours: Credit hours/units
- optional: Additional notes about the course
- description: Course description
- instructor: Who teaches the course
- section_head: Course category/section
- offered_to: Whether offered to undergrads (U) or grads (G)

Answer the user's question based on the provided course information."""
    
    def query(self, user_query: str):
        """
        Process user query using RAG with LLM agent
        
        Args:
            user_query: User's question about courses
            
        Returns:
            LLM-generated response based on retrieved course information
        """
        # Retrieve relevant courses
        relevant_courses = self._retrieve_relevant_courses(user_query)
        
        if not relevant_courses:
            yield "I couldn't find any relevant courses for your query. Please try rephrasing or specify a course code."
            return #to stop the generator
        
        # Format context for LLM
        course_context = self._format_course_context(relevant_courses)
        
        # Create prompt for LLM
        prompt = f"""{self._create_system_prompt()}

{course_context}

User Question: {user_query}

Answer:"""
        
        # Get response from LLM (now a generator)
        response_chunks_generator = self._call_ollama(prompt)

        # Iterate over the chunks and yield them directly
        for chunk in response_chunks_generator:
            yield chunk
    
    # def get_course_by_code(self, course_code: str) -> pd.DataFrame:
    #     """Get course information by exact code match"""
    #     return self.df[self.df['code'].str.contains(course_code, case=False, na=False)]
    
    # def search_courses(self, search_term: str, limit: int = 5) -> pd.DataFrame:
    #     """Search courses by term and return DataFrame"""
    #     similar_courses = self._semantic_search(search_term, top_k=limit)
    #     indices = [idx for idx, _ in similar_courses]
    #     return self.df.iloc[indices]


#---------------------------------APP-----------------------------------------------------
#-----------------------------------------------------------------------------------------

app = Flask(__name__)
# Set a secret key for session management.
# In a production environment, use a strong, randomly generated key
# and store it securely (e.g., in an environment variable).
app.secret_key = os.urandom(24)

# Initialize your bot globally or within the app context.
# Make sure 'data/Cleaned_EECE.csv' is in the correct path relative to app.py
try:
    # This line assumes MITCourseRAG is defined or imported.
    bot = MITCourseRAG(DATA_PATH, ollama_model=OLLAMA_MODEL)
    print("MITCourseRAG bot initialized successfully.")
except NameError:
    print("Error: MITCourseRAG class not found. Please ensure it's defined or imported correctly.")
    bot = None
except FileNotFoundError:
    print("Error: 'data/Cleaned_EECE.csv' not found. Please ensure the data file exists.")
    bot = None
except Exception as e:
    print(f"An unexpected error occurred during bot initialization: {e}")
    bot = None

@app.route('/')
def index():
    """Renders the main chat interface page."""
    # Initialize chat history in session if not already present
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user messages and streams bot responses."""
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    if bot is None:
        return jsonify({"error": "Chatbot not initialized. Please check server logs for errors during startup."}), 500

    # Add user message to history
    session['chat_history'].append({
        'sender': 'user',
        'message': user_input,
        'timestamp': datetime.now().strftime('%H:%M')
    })
    session.modified = True # Mark session as modified to save changes

    def generate():
        """Generator function to stream bot's response."""
        full_bot_response = ""
        start_time = perf_counter()
        try:
            response_chunks = bot.query(user_input)
            for chunk in response_chunks:
                full_bot_response += chunk
                yield chunk # Yield each chunk to the client
        except Exception as e:
            error_message = f"\nError during streaming: {e}"
            print(error_message)
            yield error_message
        finally:
            end_time = perf_counter()
            response_time_str = f"{end_time - start_time:.2f}s"
            print(f"Response Time: {response_time_str}")
            # Add bot's full response to history after streaming completes
            session['chat_history'].append({
                'sender': 'bot',
                'message': full_bot_response,
                'timestamp': datetime.now().strftime('%H:%M'),
                'response_time': response_time_str
            })
            session.modified = True # Mark session as modified

    # Stream the response using a generator
    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    # Create the 'templates' directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    print("Flask app starting...")
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
