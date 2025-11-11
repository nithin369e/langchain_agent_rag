# Install required libraries first:
# pip install streamlit langchain langchain-community chromadb sentence-transformers pypdf python-docx

import streamlit as st
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Document processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="LangChain AI Agent with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    .stChatMessage {
        background-color: #2b2b2b;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .agent-thought {
        background-color: #2b2b2b;
        border-left: 3px solid #FFA500;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9em;
        color: #FFA500;
    }
    .context-box {
        background-color: #2b2b2b;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Document processing functions
def extract_text_from_txt(file):
    return file.read().decode('utf-8')

def extract_text_from_pdf(file):
    if not PDF_AVAILABLE:
        return "PyPDF2 not installed. Install with: pip install pypdf2"
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    if not DOCX_AVAILABLE:
        return "python-docx not installed. Install with: pip install python-docx"
    doc = docx.Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def process_document(file):
    """Process uploaded document and return text"""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        text = extract_text_from_txt(file)
    elif file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(file)
    else:
        return None, "Unsupported file type"
    
    return text, file.name

# Initialize LangChain components
@st.cache_resource
def initialize_langchain_components(model_name):
    """Initialize LLM, embeddings, and vector store"""
    
    # Initialize Ollama LLM
    llm = Ollama(
        model=model_name,
        temperature=0.7,
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_langchain_db")
    
    return llm, embeddings, client

@st.cache_resource
def create_vector_store(_embeddings, _client):
    """Create or get vector store"""
    vector_store = Chroma(
        client=_client,
        collection_name="langchain_documents",
        embedding_function=_embeddings,
    )
    return vector_store

def add_documents_to_vectorstore(texts, sources, vector_store):
    """Add documents to vector store with chunking"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    all_chunks = []
    all_metadatas = []
    
    for text, source in zip(texts, sources):
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": source} for _ in chunks])
    
    vector_store.add_texts(texts=all_chunks, metadatas=all_metadatas)
    return len(all_chunks)

def create_rag_tool(llm, vector_store):
    """Create RAG tool for the agent"""
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    
    def rag_query(query: str) -> str:
        """Query the knowledge base"""
        try:
            result = qa_chain.invoke({"query": query})
            answer = result['result']
            sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
            
            # Store sources in session state for display
            if 'last_sources' not in st.session_state:
                st.session_state.last_sources = []
            st.session_state.last_sources = list(set(sources))
            
            return answer
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"
    
    return Tool(
        name="Knowledge_Base_Search",
        func=rag_query,
        description="Useful for answering questions about documents that have been uploaded. Use this tool when the user asks about specific information that might be in the uploaded documents."
    )

def create_general_knowledge_tool(llm):
    """Create general knowledge tool"""
    
    def general_query(query: str) -> str:
        """Answer general knowledge questions"""
        try:
            response = llm.invoke(query)
            return response
        except Exception as e:
            return f"Error with general knowledge query: {str(e)}"
    
    return Tool(
        name="General_Knowledge",
        func=general_query,
        description="Useful for answering general knowledge questions that don't require information from uploaded documents. Use this for common knowledge, explanations, calculations, or when no documents are available."
    )

def create_agent(llm, tools, memory):
    """Create ReAct agent with tools"""
    
    template = """You are a helpful AI assistant. Answer the user's question directly and concisely.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- If you can answer directly, provide the Final Answer immediately
- Only use tools if you need specific information from documents or additional knowledge
- Keep your reasoning concise

Begin!

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"],
        template=template
    )
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # Increased from 5 to 10
        max_execution_time=60,  # 60 seconds timeout
        early_stopping_method="generate",  # Generate answer even if not complete
    )
    
    return agent_executor

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-oss:20b-cloud"

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = 0

if "agent_thoughts" not in st.session_state:
    st.session_state.agent_thoughts = []

# Initialize LangChain components
llm, embeddings, client = initialize_langchain_components(st.session_state.model_name)
vector_store = create_vector_store(embeddings, client)

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    # Model selection
    model_input = st.text_input(
        "Model Name",
        value=st.session_state.model_name,
        help="Enter the Ollama model name"
    )
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    st.divider()
    
    # Document upload section
    st.header("📄 Knowledge Base")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload documents to add to knowledge base"
    )
    
    if uploaded_files:
        if st.button("📥 Process Documents", use_container_width=True):
            texts = []
            sources = []
            
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    text, source = process_document(file)
                    if text:
                        texts.append(text)
                        sources.append(source)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if texts:
                with st.spinner("Adding to vector store..."):
                    chunks_added = add_documents_to_vectorstore(texts, sources, vector_store)
                    st.session_state.documents_loaded += len(texts)
                    st.success(f"✅ Processed {len(texts)} documents ({chunks_added} chunks)")
            else:
                st.error("No documents could be processed")
    
    st.metric("Documents Loaded", st.session_state.documents_loaded)
    
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        try:
            client.delete_collection("langchain_documents")
            st.session_state.documents_loaded = 0
            st.success("Knowledge base cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Agent info
    st.header("🤖 Agent Status")
    st.info("""
    **Agent Flow:**
    1. 💬 User Input (Chatbot)
    2. 🤖 AI Agent (Decision)
    3. 📚 RAG Tool (If needed)
    4. 🧠 LLM Processing
    5. ✅ Response to User
    
    The agent automatically chooses the best tool based on your question.
    """)
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_thoughts = []
        st.session_state.memory.clear()
        st.rerun()

# Main chat interface
st.title("🤖 LangChain AI Agent with RAG")
st.caption(f"Model: {st.session_state.model_name} | Agent: Active ✓")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show agent thoughts if available
        if message["role"] == "assistant" and "thoughts" in message and message["thoughts"]:
            with st.expander("🧠 Agent Reasoning"):
                for thought in message["thoughts"]:
                    st.markdown(f'<div class="agent-thought">{thought}</div>', unsafe_allow_html=True)
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Sources Used"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
        
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Get AI agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.container()
        
        try:
            # Create tools
            rag_tool = create_rag_tool(llm, vector_store)
            general_tool = create_general_knowledge_tool(llm)
            tools = [rag_tool, general_tool]
            
            # Create agent
            agent_executor = create_agent(llm, tools, st.session_state.memory)
            
            # Show thinking indicator
            with thinking_placeholder:
                with st.spinner("🤖 Agent is thinking..."):
                    st.session_state.last_sources = []
                    
                    # Execute agent with timeout handling
                    try:
                        response = agent_executor.invoke(
                            {"input": prompt},
                            config={"max_execution_time": 60}
                        )
                        
                        full_response = response['output']
                        
                        # Extract agent thoughts from intermediate steps
                        thoughts = []
                        if 'intermediate_steps' in response:
                            for step in response['intermediate_steps']:
                                action, observation = step
                                thoughts.append(f"**Tool Used:** {action.tool}")
                                thoughts.append(f"**Action:** {action.tool_input}")
                                thoughts.append(f"**Result:** {observation[:200]}...")
                    
                    except Exception as agent_error:
                        # If agent fails, fall back to direct LLM call
                        if "iteration limit" in str(agent_error).lower() or "time limit" in str(agent_error).lower():
                            st.warning("⚠️ Agent exceeded limits. Using direct LLM response...")
                            
                            # Try RAG first
                            contexts, sources = retrieve_context(prompt, vector_store, n_results=3)
                            
                            if contexts:
                                rag_prompt = create_rag_prompt(prompt, contexts)
                                full_response = llm.invoke(rag_prompt)
                                st.session_state.last_sources = sources
                                thoughts = ["Fallback: Used RAG tool directly"]
                            else:
                                full_response = llm.invoke(prompt)
                                thoughts = ["Fallback: Used general knowledge"]
                        else:
                            raise agent_error
            
            # Display response
            message_placeholder.markdown(full_response)
            
            # Show agent reasoning
            if thoughts:
                with st.expander("🧠 Agent Reasoning", expanded=False):
                    for thought in thoughts:
                        st.markdown(f'<div class="agent-thought">{thought}</div>', unsafe_allow_html=True)
            
            # Show sources if RAG was used
            sources = st.session_state.get('last_sources', [])
            if sources:
                with st.expander("📚 Sources Used", expanded=False):
                    for source in sources:
                        st.markdown(f"- {source}")
            
            # Add timestamp
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": response_timestamp,
                "thoughts": thoughts,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = f"""
            ⚠️ **Error occurred:**
            
            {str(e)}
            
            **Possible solutions:**
            - Make sure Ollama is running: `ollama serve`
            - Check if model exists: `ollama list`
            - Pull the model: `ollama pull {st.session_state.model_name}`
            """
            message_placeholder.markdown(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = """👋 Welcome! I'm an AI Agent powered by LangChain and RAG.

**How I work:**
1. 💬 You ask a question
2. 🤖 I analyze and choose the best tool
3. 📚 I search uploaded documents (if relevant)
4. 🧠 I process with LLM
5. ✅ I provide you an answer

**Features:**
- Intelligent tool selection
- RAG-enhanced responses
- Transparent reasoning process
- Multi-document support

**Get started:**
Upload documents in the sidebar or just ask me anything!"""
        st.markdown(welcome_msg)
        st.caption("Agent ready 🤖")