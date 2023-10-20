import { Document } from "langchain/document";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "langchain/text_splitter";

import { Pinecone } from "@pinecone-database/pinecone";

import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";

import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";

import path from 'path'

import { config } from "dotenv";

config();

const openAIApiKey = process.env.OPEN_AI_API_KEY;

async function main(filePath) {
  // create document array
  const docs = [
    new Document({
      metadata: { name: `Filepath: ${filePath}` },
    }),
  ];

  // initialize loader
  const Loader = path.extname(filePath) === `.pdf` ? PDFLoader : TextLoader;

  const loader = new Loader(filePath);

  // load and split the docs
  const loadedAndSplitted = await loader.loadAndSplit();

  // push the splitted docs to the array
  docs.push(...loadedAndSplitted);

  // create splitter
  const textSplitter = new CharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
  });

  // use the splitter to split the docs to different chunks
  const splittedDocs = await textSplitter.splitDocuments(docs);

  // create pinecone index
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  // create openai embedding
  const embeddings = new OpenAIEmbeddings({ openAIApiKey });

  // create a pinecone store using the splitted docs and the pinecone index
  const ns = new Date().getTime().toString();
  const pineconeStore = await PineconeStore.fromDocuments(
    splittedDocs,
    embeddings,
    {
      pineconeIndex,
      namespace: ns,
    }
  );
  console.log({ns})

  // initialize openai model
  const model = new OpenAI({
    openAIApiKey,
    modelName: "gpt-3.5-turbo",
  });

  // create a vector chain using the llm model and the pinecone store
  const chain = VectorDBQAChain.fromLLM(model, pineconeStore, {
    k: 1,
    returnSourceDocuments: true,
  });

  // use the chain to query my data
  const response = await chain.call({
    query: "Explain about the contents of the pdf file I provided.", // question is based on the file i provided
  });

  console.log(`\nResponse: ${response.text}`); 
}

main('./Sample.pdf')