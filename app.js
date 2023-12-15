import express from 'express';
import bodyParser from 'body-parser';
import get_qa_chain  from './chatbot2.js';
import cors from "cors";

// const express = require('express');
// const bodyParser = require('body-parser');
// const get_qa_chain = require('./chatbot2');

const app =express();


app.use(cors());
app.use(bodyParser.json());

app.listen(8080,()=>{
    console.log("Bot is up and running !!");
})

app.get('/',(req,res)=>{
    res.send({message:"chat bot is deployed!!!"})
});

app.post('/chatbot',async(req,res)=>{
    const question = req.body.question;
    const ans = await get_qa_chain(question);
    res.send(ans);
});