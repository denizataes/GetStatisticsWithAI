from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain,LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import openai
import os


API_KEY = ""


# HTTP Sunucu istemcilerinin istekleri kabul etmesi için bir işleyici sınıfı oluşturun.

class PostHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            text_chunks = split_json_into_chunks(post_data.decode('utf-8'), 500)
            print(len(text_chunks))
            # create vector store
            db = get_vectorstore(text_chunks)
            #query = "Aşağıda bir hastaneye giden hastalar ile ilgili birimler verilecektir. Bu bilgiler Hasta Bilgisi, Kabul Bilgisi ve Kabulün bağlı olduğu birim bilgisi, Hastanın üzerindeki tanılar gibi verilerden oluşmaktadır.Bu bilgilere göre 20 adet istatistik üret. Ürettiğin İstatistikler bir ekranda gösterilecek. Buna göre doktorların anlayacağı güzel bir istatistik dili kullan.Ürettiğin istatistikleri yaz."
            query = "sana verilen veri seti içerisinde en yaşlı hasta kimdir?"
            #The data set given to you contains records of a hospital. Extract statistical analysis data based on these records.For example, how many of the hospitalized patients are women?
            #Veri setinde bulunan veriler bir hastanenin verileri. Bu verileri kullanarak 10 adet istatistik üret. Bu istatiskler şu şekilde olabilir.Hastaların kaçı kadın, Hastaların yaş aralığı, en genç hasta kimdir...
            qa = get_conversation_chain(db)
            result = qa({"question": query})
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response_data = result['answer']
            self.wfile.write(response_data.encode('utf-8'))
        except Exception as e:
            # Hata durumunda yanıt gönderin
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response_data = json.dumps({"error": str(e)})
            self.wfile.write(response_data.encode('utf-8'))

def get_conversation_chain(vectorstore):
    llm = OpenAI(openai_api_key = API_KEY)
#     llm = LlamaCpp(
#     model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin",
#     temperature=0.75,
#     max_tokens=5000,
#     top_p=1,
#     n_ctx=10000,
#     #callback_manager=callback_manager,
#     verbose=True,
# )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

            
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks

def split_json_into_chunks(json_data, chunk_size):
    # JSON verilerini bir Python sözlüğüne çözümle
    data = json.loads(json_data)

    # JSON verisini güzel bir şekilde biçimlendir
    formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
    return get_text_chunks(formatted_json)
    # Belirtilen chunk boyutuna göre JSON verisini bölmek için bir döngü kullanın
    chunks = []
    for i in range(0, len(formatted_json), chunk_size):
        chunk = formatted_json[i:i+chunk_size]
        chunks.append(chunk)

    return chunks

def get_answer(text):

    #TOKEN LİMİTİNE TAKILIYOR BURASI.
    print("GİRDİ-------")
    llm = OpenAI(openai_api_key = API_KEY)
    template = """Aşağıda bir hastaneye giden hastalar ile ilgili birimler verilecektir. Bu bilgiler Hasta Bilgisi, Kabul Bilgisi ve Kabulün bağlı olduğu birim bilgisi, Hastanın üzerindeki tanılar gibi verilerden oluşmaktadır.Bu bilgilere göre 20 adet istatistik üret. Ürettiğin İstatistikler bir ekranda gösterilecek. Buna göre doktorların anlayacağı güzel bir istatistik dili kullan.Ürettiğin istatistikleri yaz.:  {text}"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(text)
    print("RESPONSE: " + response)
    return response

    


def run_server(port=8065):
    # HTTP Sunucusunu belirlediğiniz portta başlatın.
    server_address = ('', port)
    httpd = HTTPServer(server_address, PostHandler)
    print('HTTP Sunucusu {} portunda çalışıyor...'.format(port))
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
