# 🎯 AI Engineer Interview Prep Guide
> Every concept below is linked to **something you already built**. You don't need to memorize — you need to **connect the dots**.

---

## 1. Classical ML (Your Project 1 & 5)

### Concepts You Must Explain

| Concept | What It Means | Your Example |
|---|---|---|
| **Supervised Learning** | Model learns from labeled data (input → output pairs) | Project 1: you had customer data labeled "churned" or "not churned" |
| **Feature Engineering** | Transforming raw data into useful inputs for the model | Project 1: you created features from customer data before training |
| **Train/Test Split** | Splitting data so you test on unseen data to avoid cheating | Project 1: you split customer data 80/20 |
| **Overfitting** | Model memorizes training data, fails on new data | "Like a student who memorizes answers but can't solve new problems" |
| **Underfitting** | Model is too simple to capture patterns | "Like using a straight line to fit curved data" |
| **TF-IDF** | Converts text to numbers by measuring word importance | Project 5: you used TF-IDF to convert ticket text into vectors for classification |
| **Cross-Validation** | Train/test multiple times on different splits for reliable scores | Project 1: you used this during hyperparameter tuning |
| **Hyperparameter Tuning** | Finding the best model settings (not learned from data) | Project 1: GridSearch/RandomSearch to find best parameters |

### Common Interview Questions

**Q: "What's the difference between a parameter and a hyperparameter?"**
> **Parameter** = learned by the model during training (e.g., weights in logistic regression)
> **Hyperparameter** = set by you before training (e.g., learning rate, number of trees in Random Forest)
> 
> *"In my churn prediction project, the coefficients of Logistic Regression were parameters. But the `C` regularization strength and `max_depth` of Random Forest were hyperparameters I tuned with GridSearch."*

**Q: "How do you handle imbalanced data?"**
> - **Resampling**: Oversample minority class (SMOTE) or undersample majority
> - **Class weights**: Tell the model to penalize mistakes on the minority class more
> - **Metrics**: Don't use accuracy — use Precision, Recall, F1-score
> 
> *"In my ticket classifier, some categories had very few examples. I used class_weight='balanced' in SVM and evaluated with F1-score instead of accuracy."*

**Q: "Explain bias-variance tradeoff"**
> - **High bias** = underfitting (model too simple, misses patterns)
> - **High variance** = overfitting (model too complex, memorizes noise)
> - Goal: find the sweet spot
> 
> *"In Project 1, Logistic Regression had high bias (too simple for the patterns), while a deep Random Forest had high variance. I tuned `max_depth` to find the balance."*

---

## 2. Deep Learning (Your Project 2)

### Concepts You Must Explain

| Concept | What It Means | Your Example |
|---|---|---|
| **Neural Network** | Layers of neurons that learn patterns by adjusting weights | Project 2: your CNN for digit recognition |
| **CNN (Convolutional Neural Network)** | Specialized for images — uses filters to detect edges, shapes, patterns | Project 2: filters learned to detect digit edges and curves |
| **Convolution** | Sliding a small filter over an image to detect features | "Like scanning an image with a magnifying glass to find edges" |
| **Pooling** | Shrinking the image to keep important features, reduce computation | Max pooling = "keep the strongest signal in each region" |
| **Dropout** | Randomly turning off neurons during training to prevent overfitting | "Like forcing a team to work even when some members are absent" |
| **Data Augmentation** | Creating more training data by flipping, rotating, shifting images | Project 2: you used this to improve digit classifier generalization |
| **Activation Function** | Adds non-linearity so the network can learn complex patterns | ReLU = "if positive, keep it; if negative, make it 0" |
| **Backpropagation** | How the network learns — calculates how much each weight contributed to the error, then adjusts | "Tracing back from the mistake to fix what caused it" |

### Common Interview Questions

**Q: "Why use CNN instead of a regular neural network for images?"**
> A regular (dense) neural network treats every pixel independently — a 100×100 image = 10,000 inputs, no spatial awareness.
> A CNN uses **filters that slide across the image**, so it understands that neighboring pixels form edges and shapes. It also has **weight sharing** (same filter used everywhere) = far fewer parameters.
> 
> *"In my SVHN digit classifier, a dense network would need millions of parameters. My CNN used shared convolutional filters, which was more efficient and understood spatial patterns like curves and edges."*

**Q: "What is the vanishing gradient problem?"**
> During backpropagation, gradients can become extremely small as they pass through many layers, so early layers barely learn.
> **Solutions**: ReLU activation (instead of sigmoid), Batch Normalization, Skip Connections (ResNet)

---

## 3. NLP & Transformers (Your Project 3)

### Concepts You Must Explain

| Concept | What It Means | Your Example |
|---|---|---|
| **Tokenization** | Splitting text into pieces (words or subwords) the model can process | Project 3: DistilBERT tokenizer splits "unhappiness" → ["un", "##happiness"] |
| **Embedding** | Converting a word/token into a dense vector of numbers that captures meaning | "King" and "Queen" have similar embeddings because they're used in similar contexts |
| **Attention Mechanism** | Lets the model focus on the most relevant words in a sentence | In "The bank by the river", attention helps distinguish "bank" (financial vs. riverbank) |
| **Self-Attention** | Each word looks at ALL other words to understand context | This is the core of Transformers |
| **Transfer Learning** | Using a model pre-trained on massive data, then fine-tuning on your small dataset | Project 3: DistilBERT was pre-trained on Wikipedia, you fine-tuned it on sentiment data |
| **Fine-Tuning** | Taking a pre-trained model and training it further on your specific task | Project 3: you updated DistilBERT's weights for sentiment classification |
| **BERT** | Bidirectional model — reads text left AND right simultaneously | Unlike GPT which only reads left-to-right |
| **DistilBERT** | Smaller, faster version of BERT (40% smaller, 60% faster, 97% performance) | Project 3: you chose this for efficiency |

### Common Interview Questions

**Q: "What is a Transformer and why is it important?"**
> The Transformer architecture (2017 paper "Attention is All You Need") replaced RNNs/LSTMs by using **self-attention** to process all words in parallel instead of sequentially.
> 
> Key innovation: **Self-attention** lets every word attend to every other word, capturing long-range dependencies. Plus it's **parallelizable** = much faster to train on GPUs.
> 
> *"In my sentiment classifier, DistilBERT (a Transformer) understood that 'not good' is negative because attention connects 'not' to 'good'. A bag-of-words model would see 'good' as positive."*

**Q: "What's the difference between BERT and GPT?"**
> | | BERT | GPT |
> |---|---|---|
> | Direction | Bidirectional (sees full context) | Left-to-right (autoregressive) |
> | Training | Masked Language Model (fill in blanks) | Next token prediction |
> | Best for | Classification, NER, Q&A | Text generation, chatbots |
> 
> *"I used BERT (DistilBERT) for sentiment classification because it's a classification task — I need to understand the whole sentence. GPT would be better for generating text, which is what I used Gemini for in Project 5."*

---

## 4. LLMs & Prompt Engineering (Your Project 4 & 5)

### Concepts You Must Explain

| Concept | What It Means | Your Example |
|---|---|---|
| **LLM (Large Language Model)** | A massive neural network trained on internet-scale text data | Gemini, GPT-4, Claude |
| **Prompt Engineering** | Crafting the input text to get the best output from an LLM | Project 5: your `analysis_prompt.txt` template |
| **Temperature** | Controls randomness: 0 = deterministic, 1 = creative | Project 5: you used low temperature for consistent ticket analysis |
| **Token** | The basic unit LLMs process (roughly ¾ of a word) | "Hello world" ≈ 2 tokens. Important for cost calculation |
| **Context Window** | Maximum number of tokens the model can process at once | Gemini Flash: 1M tokens. GPT-4: 128K tokens |
| **Hallucination** | When an LLM generates confident but incorrect information | This is why you built RAG — to ground responses in real documents |
| **Grounding** | Providing the LLM with factual context so it doesn't hallucinate | Project 5: your RAG retriever provides policy documents as context |
| **Few-Shot Prompting** | Giving the model examples in the prompt to guide its output | "Here are 3 examples of good ticket responses, now respond to this one" |
| **System Prompt** | Instructions that define the LLM's behavior and constraints | Project 5: your analysis prompt tells Gemini to be a support agent |

### Common Interview Questions

**Q: "How do you prevent LLM hallucination?"**
> 1. **RAG (Retrieval-Augmented Generation)** — retrieve relevant documents first, include them in the prompt
> 2. **Grounding** — instruct the model to only use provided context
> 3. **Temperature = 0** — reduce randomness
> 4. **Structured output** — force JSON schema so the model can't ramble
> 
> *"In my support ticket platform, I used RAG to retrieve relevant policy chunks from pgvector, included them in the prompt as context, and instructed Gemini to only reference provided policies. This eliminated hallucination about company policies that don't exist."*

**Q: "When would you use an LLM vs. classical ML?"**
> | Use Classical ML When | Use LLM When |
> |---|---|
> | You need speed (< 10ms) | You need text generation |
> | You have structured/tabular data | You need reasoning/understanding |
> | You need predictable costs (free) | You need flexibility for new tasks |
> | The task is well-defined classification | The task requires creativity/nuance |
> 
> *"In my architecture, I use scikit-learn as the 'traffic controller' — it classifies tickets in milliseconds for free. Then Gemini is the 'expert operator' — it generates detailed, context-aware replies. This hybrid approach is 10x cheaper than sending everything to an LLM."*

This is a **great answer** that shows system design thinking. Interviewers love this.

---

## 5. RAG — Retrieval-Augmented Generation (Your Project 4 & 5)

### This Is Your Strongest Topic — Own It

| Concept | What It Means | Your Example |
|---|---|---|
| **RAG** | Retrieve relevant documents → feed them to LLM → generate grounded answer | Project 5: retrieve policy → Gemini generates reply |
| **Embedding Model** | Converts text into a vector (array of numbers) that captures meaning | Project 5: `gemini-embedding-001` (768 dimensions) |
| **Vector Database** | Database optimized for storing and searching embeddings by similarity | Project 5: PostgreSQL + pgvector extension |
| **Cosine Similarity** | Measures how similar two vectors are (1 = identical, 0 = unrelated) | pgvector uses this to find most relevant policy chunks |
| **Chunking** | Splitting large documents into smaller pieces for better retrieval | Project 5: you split policy docs into overlapping chunks |
| **Top-K Retrieval** | Retrieve the K most similar chunks to the user's query | Project 5: you retrieve top 3-5 relevant policy chunks |
| **IVFFlat Index** | Approximate nearest neighbor index — faster search, slight accuracy loss | Project 5: you learned this can't be created on empty tables! |

### The RAG Pipeline (Explain This Fluently)

```
User Query
    ↓
[1] Embed the query (gemini-embedding-001 → 768-dim vector)
    ↓
[2] Search pgvector for similar chunks (cosine similarity, top-K)
    ↓
[3] Retrieve matching policy text chunks
    ↓
[4] Build prompt: System instructions + Retrieved context + User query
    ↓
[5] Send to Gemini LLM → Generate grounded response
    ↓
[6] Return structured response to user
```

### Common Interview Questions

**Q: "Explain RAG and why it's important"**
> RAG solves two problems: LLMs have **stale knowledge** (training cutoff) and they **hallucinate**. RAG fixes both by retrieving up-to-date, relevant documents and injecting them into the prompt so the LLM generates answers grounded in real data.
> 
> *"In my project, the LLM doesn't know our company's refund policy. So I embed all policy documents into pgvector, and when a ticket comes in, I search for the most relevant policy chunks and include them in the prompt. The LLM then generates a response that accurately references our actual policies."*

**Q: "What is an embedding and why is dimensionality important?"**
> An embedding is a dense vector representation where **similar meanings are close together** in vector space. "refund" and "return" would have similar embeddings, while "refund" and "pizza" would be far apart.
>
> Dimensionality matters for:
> - **Accuracy**: More dimensions can capture more nuance (3072-dim > 768-dim)
> - **Speed**: Fewer dimensions = faster search
> - **Storage**: Fewer dimensions = less database space
> 
> *"I chose 768 dimensions as a balance. I learned this the hard way — the embedding model defaulted to 3072 dimensions but my database schema was 768, causing a dimension mismatch error. I had to force `output_dimensionality=768` in the embedding call."*

**Q: "How would you improve RAG quality?"**
> 1. **Better chunking**: Overlap chunks so context isn't lost at boundaries
> 2. **Hybrid search**: Combine vector similarity + keyword search (BM25)
> 3. **Re-ranking**: Retrieve top-20, then re-rank with a cross-encoder to get top-5
> 4. **Metadata filtering**: Filter by document type/date before vector search
> 5. **Query expansion**: Rephrase the user's query to improve retrieval

---

## 6. System Design (Your Project 5)

### Your Architecture (Be Ready to Whiteboard This)

```
                        ┌──────────────────────────┐
                        │      Client Request       │
                        └────────────┬─────────────┘
                                     ↓
                        ┌──────────────────────────┐
                        │   FastAPI + Auth (API Key) │
                        └────────────┬─────────────┘
                                     ↓
                 ┌───────────────────┼───────────────────┐
                 ↓                   ↓                   ↓
        ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
        │  scikit-learn │    │  pgvector    │    │  Gemini LLM  │
        │  Classifier   │    │  RAG Search  │    │  Generation  │
        │  (Fast/Free)  │    │  (Context)   │    │  (Smart/Paid)│
        └──────┬──────┘    └──────┬──────┘    └──────┬───────┘
               │                  │                   │
               └──────────────────┼───────────────────┘
                                  ↓
                        ┌──────────────────────────┐
                        │   PostgreSQL Database      │
                        │   (tickets, chunks, chat)  │
                        └──────────────────────────┘
```

### Common Interview Questions

**Q: "How would you scale this to handle 10K requests/second?"**
> 1. **Separate fast path vs. slow path**: Classification (scikit-learn) can handle 10K/s easily. LLM generation is the bottleneck.
> 2. **Queue the LLM calls**: Use a message queue (Redis/RabbitMQ) so classification returns instantly, LLM response comes async.
> 3. **Cache common responses**: If the same ticket type comes in repeatedly, cache the LLM response.
> 4. **Horizontal scaling**: Run multiple FastAPI instances behind a load balancer.
> 5. **pgvector optimization**: Use HNSW index instead of IVFFlat for better search performance at scale.

**Q: "Why not just send everything to the LLM?"**
> **Cost, speed, and reliability.**
> - scikit-learn classifies a ticket in **< 10ms for free**
> - Gemini takes **1-3 seconds and costs money per token**
> - If Gemini is down, classification still works
> 
> *"I designed a hybrid architecture where classical ML handles the fast, predictable routing and the LLM handles the creative, context-dependent generation. This is 10x cheaper and more resilient."*

---

## 7. Database & Vectors (Your Project 5)

### Key Concepts

**Q: "Why pgvector instead of a dedicated vector DB like Pinecone?"**
> - **Simplicity**: One database for everything (tickets + embeddings + chat logs)
> - **SQL power**: Can combine vector search with SQL filters (WHERE category = 'billing')
> - **Cost**: Free and self-hosted vs. paid SaaS
> - **Trade-off**: Pinecone is faster at billion-scale, but pgvector is perfect for millions of vectors

**Q: "Explain IVFFlat vs HNSW indexes"**
> | | IVFFlat | HNSW |
> |---|---|---|
> | How it works | Divides vectors into clusters, searches nearby clusters | Builds a graph connecting similar vectors |
> | Speed | Fast | Faster |
> | Accuracy | Good (might miss vectors in neighboring clusters) | Better |
> | Build time | Fast | Slow (graph construction) |
> | Memory | Low | High |
> | Gotcha | **Cannot build on empty table** | Can build on empty table |

---

## 8. Quick-Fire Questions (Rapid Review)

| Question | Answer |
|---|---|
| Precision vs Recall? | Precision = "of what I predicted positive, how many were correct?" Recall = "of all actual positives, how many did I find?" |
| L1 vs L2 regularization? | L1 (Lasso) = can zero out features (feature selection). L2 (Ridge) = shrinks all weights evenly |
| Batch vs Stochastic gradient descent? | Batch = uses all data per step (stable but slow). SGD = uses 1 sample (noisy but fast). Mini-batch = compromise |
| What is a loss function? | Measures how wrong the model is. Training = minimize this. E.g., Cross-Entropy for classification |
| Epoch vs Iteration? | Epoch = one pass through ALL training data. Iteration = one weight update (one batch) |
| What is gradient descent? | Walk downhill on the loss surface to find the minimum — adjust weights in the direction that reduces error |
| What is an API? | Contract between two systems. "Send me this format, I'll return that format." REST = HTTP-based |
| Docker vs VM? | Docker = shares OS kernel (lightweight, fast). VM = full OS copy (heavy, isolated) |
| SQL vs NoSQL? | SQL = structured, relationships, ACID. NoSQL = flexible schema, horizontal scaling |
| What is CI/CD? | Continuous Integration (auto-test on push) + Continuous Deployment (auto-deploy on merge) |

---

## 📝 Interview Day Cheat Sheet

### When They Ask "Tell Me About Yourself"
> "I'm a software developer with 2 years of banking experience at Silverlake, working with Java Spring Boot and .NET. I recently built a 5-project AI/ML portfolio that goes from classical machine learning to production LLM applications. My most recent project is a Support Ticket Intelligence API that combines scikit-learn for fast classification, pgvector for RAG retrieval, and Gemini for generating context-aware responses — all containerized with Docker and tested with pytest."

### When They Ask "Tell Me About a Challenge"
> "When building my RAG pipeline, I hit a dimension mismatch error — my embedding model was outputting 3072-dimensional vectors but my database schema expected 768. I debugged this by writing a test script that checked the actual output dimensions, discovered the API had changed its default, and fixed it by forcing `output_dimensionality=768` in the embedding call. This taught me the importance of validating data shapes at integration boundaries."

### When They Ask "Why AI/ML?"
> "In my banking work, I saw many repetitive, pattern-based tasks that humans were doing manually. I realized machine learning could handle the pattern recognition (classification, routing) while LLMs could handle the nuanced communication (generating responses). That's exactly what I built in my portfolio — a system where ML does the fast routing and LLMs do the thinking."
