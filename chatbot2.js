import  { GooglePaLM } from "langchain/llms/googlepalm";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { FaissStore } from "langchain/vectorstores/faiss";
import { HuggingFaceTransformersEmbeddings } from "langchain/embeddings/hf_transformers";
import { config } from "dotenv";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { PromptTemplate } from "langchain/prompts";
// import { ConversationChain } from "langchain/chains";

config();

// const { ConversationChain } = require("langchain/chains");




// const env = require('dot')

const model = new GooglePaLM({
    apiKey: process.env.GOOGLE_PALM_API_KEY, // or set it in environment variable as `GOOGLE_PALM_API_KEY`
    // other params
    temperature: 0.7, // OPTIONAL
    maxOutputTokens: 1024, // OPTIONAL
    topK: 40, // OPTIONAL
    topP: 1,
});


const loader = new CSVLoader("./PlantFAQ2.csv");

let docs = [];  // Declare the variable outside the function
const directory = "./faiss_store";


async function create_db() {
    docs = await loader.load();  // Assign the loaded data to the global variable
        // console.log(docs);
        const vectorStore = await FaissStore.fromDocuments(
            docs,
            new HuggingFaceTransformersEmbeddings(
                {
                    modelName: "Xenova/all-MiniLM-L6-v2",
                }
            )
        );

    const directory = "./faiss_store";

    await vectorStore.save(directory);
}

async function get_qa_chain(question) {
    try {
        const loadVectorStore = await FaissStore.load(
            directory,
            new HuggingFaceTransformersEmbeddings(
                {
                    modelName: "Xenova/all-MiniLM-L6-v2",
                }
            )
        )

        const retriever = loadVectorStore.asRetriever();
        const memory = new BufferMemory({ memoryKey: "chat_history" , k: 2});
        
        const template = `You are a Plant Researcher having extensive knowledge about the Ayurveda plants.For 
        the given question provide an answer based on the context only.For the answer try to provide as much text 
        as possible from "answer" section in the source document context without making many changes.

        If answer is not known kindly print "I'm unable to provide an answer on that topic as it's beyond my current scope. 
        Consulting a subject matter expert would offer the most accurate insights." 

        After providing the answer always add "Please note that the information provided here is based on general knowledge about these plants and their traditional uses. For specific details, it's advisable to seek guidance from an expert or doctor well-versed in this area." except if answer is not known.

        CONTEXT: {context}

        QUESTION: {question}`;
        
        const promptTemplate = PromptTemplate.fromTemplate(template);
        console.log(promptTemplate.inputVariables);
        const chain = ConversationalRetrievalQAChain.fromLLM(
            model,
            retriever,
            {
              memory,
            },
            promptTemplate
          );

          let res1 = await chain.call({ question: question });
        return res1;
        


      
    } catch (error) {
        return error
    }
}

export default get_qa_chain;



