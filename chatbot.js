const {C} = require('langchain');
const langchain = new Langchain();



// Create a database
async function createDB() {
    const data = await langchain.CSVLoader.load({ filePath: 'PlantFAQ2.csv', sourceColumn: 'Query' });
    const vectorDB = await langchain.FAISS.fromDocuments({ documents: data, embeddings: 'instructor' });
    await vectorDB.saveLocal('faiss_index');
}

// Get QA Chain
async function getQAChain() {
    const vectorDB = await langchain.FAISS.loadLocal('faiss_index', 'instructor');
    const retriever = vectorDB.asRetriever({ scoreThreshold: 0.65 });

    const promptTemplate = `constext: {context}\n\nQUESTION: {question}\n\nPlease note that the information provided here is based on general knowledge about these plants and their traditional uses. For specific details, it's advisable to seek guidance from an expert or doctor well-versed in this area.`;

    const prompt = await langchain.PromptTemplate.create({ template: promptTemplate, inputVariables: ['context', 'question'] });
    const memory = await langchain.ConversationBufferWindowMemory.create({ k: 2 });

    const chain = await langchain.RetrievalQA.fromChainType({
        llm: 'google_palm',
        chainType: 'stuff',
        inputKey: 'question',
        retriever,
        chainTypeKwargs: { prompt },
        memory
    });

    return chain;
}

// Usage
async function main() {
    await createDB();
    const chain = await getQAChain();
    const response = await chain('What are the traditional uses of Ayurvedic plants?');
    console.log(response);
}

module.exports = main;
