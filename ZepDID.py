import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLineEdit, QFileDialog
from PyQt5.QtCore import Qt, QThreadPool, QRunnable
from PyQt5.QtGui import QFont
from pydantic.v1 import BaseModel, Field, validator
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, load_tools, initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage
from langchain import OpenAI
from langchain.memory import ZepMemory
from langchain.retrievers import ZepRetriever
from langchain.schema import HumanMessage, AIMessage
from langchain.utilities import WikipediaAPIWrapper
from uuid import uuid4
import getpass
import time
from langchain.memory import CombinedMemory, VectorStoreRetrieverMemory
from typing import Optional, Type
from langchain.tools import MoveFileTool, format_tool_to_openai_function
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from os import environ
# Set OpenAI API key
environ["OPENAI_API_KEY"] = "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"

# Set this to your Zep server URL
ZEP_API_URL = "http://localhost:8000"
# Provide your Zep API key. Note that this is optional. See https://docs.getzep.com/deployment/auth
AUTHENTICATE = False
zep_api_key = None
if AUTHENTICATE:
    zep_api_key = getpass.getpass()
session_id = str(uuid4())  # This is a unique identifier for the user/session
# Initialize the Zep Memory Class
zep_memory = ZepMemory(
    session_id=session_id, url=ZEP_API_URL, api_key=zep_api_key
)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
math_llm = OpenAI(temperature=0.0)

files = []
tools = []

class PDF_Tools(BaseModel):
    question: str = Field()


'''class CustomSearchTool(BaseModel):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")'''

class OpenFilesRunnable(QRunnable):
    def __init__(self, mainWindow, files_paths):
        super().__init__()
        self.mainWindow = mainWindow
        self.files_paths = files_paths

    def run(self):
        global tools
        global files
        tools = []
        for path in self.files_paths:
            file_name = path.split("/")[-1].split(".")[0]
            files.append({
                "name": file_name, 
                "path": path
            })
        for file in files:
            loader = PyPDFLoader(file["path"])
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            docs = text_splitter.split_documents(pages)
            embeddings = OpenAIEmbeddings()
            retriever = FAISS.from_documents(docs, embeddings).as_retriever()
            print(file)
            
            tools.append(
                Tool(
                    args_schema=PDF_Tools,
                    name=file['name'],
                    description=f"useful when you want to answer questions about {file['name']}. Also known as a document.",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                )
            )
        

        

        agent_kwargs = {
            "system_message": SystemMessage(content="You are a government engineer tasked with searching existing Data Item Descriptions (DID) to determine if duplicates exist. Review the DIDs, summarize each, and give a final opinion of the similarities, differences, and recommend if one or the other can be removed from the database.")
        }
        
        # Set up Zep Chat History
        memory = ZepMemory(
            session_id=session_id,
            url=ZEP_API_URL,
            api_key=zep_api_key,
            memory_key="chat_history",
        )
        
        self.mainWindow.agent = initialize_agent(agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                             tools=tools,
                             llm=llm,
                             verbose=True,
                             agent_kwargs=agent_kwargs,
                             memory=memory,)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initializeAgent()  # Initialize the chat agent
        self.files = []  # Initialize an empty files list

    def initUI(self):
        self.setWindowTitle('Chat Agent GUI')
        layout = QVBoxLayout()

        self.inputField = QLineEdit(self)
        self.inputField.setPlaceholderText('Type your question here...')
        layout.addWidget(self.inputField)

        self.sendButton = QPushButton('Send', self)
        self.sendButton.clicked.connect(self.onSend)
        layout.addWidget(self.sendButton)

        self.responseArea = QTextEdit(self)
        self.responseArea.setReadOnly(True)
        layout.addWidget(self.responseArea)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        # Bind the returnPressed signal to the onSend slot
        self.inputField.returnPressed.connect(self.onSend)
        
        current_size = self.size()
        self.resize(current_size.width() +100, current_size.height() +150)
        
        self.openFilesButton = QPushButton('Open PDF Files', self)
        self.openFilesButton.clicked.connect(self.onOpenFiles)
        layout.addWidget(self.openFilesButton)


    def onOpenFiles(self):
        files_paths, _ = QFileDialog.getOpenFileNames(self, "Open PDF Files", "", "PDF Files (*.pdf)")
        if files_paths:
            openFilesRunnable = OpenFilesRunnable(self, files_paths)
            QThreadPool.globalInstance().start(openFilesRunnable)
    
    def initializeAgent(self):
        # Define agent_kwargs and tools as needed
        agent_kwargs = {
            "system_message": SystemMessage(content="You are a government engineer tasked with searching existing Data Item Descriptions (DID) to determine if duplicates exist. Review the DIDs, summarize each, and give a final opinion of the similarities, differences, and recommend if one or the other can be removed from the database.")
        }
        
        tools = load_tools(["human", "llm-math"], llm)

        # Set up Zep Chat History
        memory = ZepMemory(
            session_id=session_id,
            url=ZEP_API_URL,
            api_key=zep_api_key,
            memory_key="chat_history",
        )
        # Initialize the chat agent and assign it to the self.agent attribute
        self.agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,#OPENAI_MULTI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )
    def onSend(self):
        user_input = self.inputField.text()
        if user_input:
            response = self.agent({"input": user_input})
            # Extracting the 'output' from the response dictionary.
            agent_output = response.get('output', 'Sorry, I couldn\'t process that.')

            self.responseArea.append(f"\n\nUser: {user_input}")
            self.responseArea.append(f"\n\nChat Agent: {agent_output}")
            self.inputField.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set the global font size to 12pt
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

