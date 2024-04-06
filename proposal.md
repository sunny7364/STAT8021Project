# Proposal of this project

DUE on April 7th, work!!!

钟韵涵：introduction，idea and why, past paper（？）

程水齐：algorithm，methodology

余加阳：dataset explanation? preprocessing？

杨文瀚：demo+interpretation


## Introduction
In our group project, what we are going to do is an interpretable personalized recommendation system based on Synergized LLM and KG reasoning.

### Integration of Knowledge Graphs in Recommendation Systems

The evolution of recommendation systems from 1995 to 2005 primarily encompassed content-based filtering, collaborative filtering, and hybrid methodologies. These approaches significantly mitigated issues related to information overload and the long tail phenomenon. However, they encountered several limitations stemming from their reliance on explicit or implicit user-item feedback. These limitations included data sparsity, the cold start problem, a deficiency in diversity, and inadequate generalization capabilities. The advent of Knowledge Graphs (KGs) offered a novel paradigm to address these challenges by infusing rich domain knowledge and semantic interconnections, thereby augmenting the precision, diversity, and interpretability of recommendation outputs.

**Principal Advantages of Knowledge Graph Integration:**

1. **Enriched Semantic Insight:** Knowledge Graphs encapsulate comprehensive semantic details about entities, attributes, and their interrelations, facilitating a deeper comprehension of user preferences and unveiling novel item-user connections.
2. **Enhanced Interpretability:** Through the dissection of entity relationships and path embeddings within KGs, a framework for explicating recommendation rationales is established, fostering user acceptance and understanding of the recommendation logic.
3. **Superior Inference Capabilities:** KGs empower recommendation systems with robust reasoning, enabling anomaly detection, inferential completion, and the exploration of new knowledge domains with heightened accuracy.

### The Convergence of Large Language Models and Knowledge Graphs

The amalgamation of Large Language Models (LLMs) with Knowledge Graphs harnesses the inherent strengths of both technologies: the profound natural language processing prowess of LLMs and the structured, interpretable knowledge repository of KGs. This confluence not only enhances text-related operations but also ensures the delivery of current, verifiable knowledge and broadens applicability across diverse domains.

**Synergistic Strengths:**

- **LLMs** shine in the realm of natural language comprehension but grapple with challenges related to interpretability and the incorporation of up-to-the-minute knowledge.
- **KGs** stand out for their structured, precise knowledge representation and interpretability but are often plagued by incompleteness and a lack of proficiency in natural language processing tasks.

### Advantages and Disadvantages of Knowledge Graphs

**Advantages:**

- Provision of knowledge in a structured, easily comprehensible format.
- High levels of accuracy and decisiveness in knowledge representation.
- Facilitation of symbolic reasoning, enhancing interpretability.
- Availability of domain-specific insights through expertly curated graphs.
- Capability to evolve with the inclusion of new facts and the removal of outdated information.

**Disadvantages:**

- Challenges associated with the completeness and labor-intensive construction of KGs.
- Limited capability in processing and understanding natural language within graphs.
- Difficulties in accommodating unseen entities and representing novel facts.
- The presence of path uncertainty and non-uniqueness in the relationship mappings.

### Advantages and Disadvantages of Large Language Models

**Advantages:**

- Comprehensive coverage of general knowledge and exceptional natural language processing capabilities.
- Remarkable adaptability across a wide spectrum of tasks, facilitated by few-shot learning and fine-tuning on multi-task datasets.

**Disadvantages:**

- Implicit knowledge representation poses significant hurdles to interpretability.
- Susceptibility to generating plausible yet factually incorrect content (hallucinations).
- Indecisiveness inherent in probabilistic reasoning models.
- A notable deficiency in domain-specific or newly emerging knowledge due to training on static corpora.


## Dataset Selection

### Overview of the Dataset

Our proposed recommendation system will leverage a comprehensive dataset consisting of 140,502 movie records and 72,959 individual records (actors and directors). The movie dataset encompasses a broad spectrum of information including titles, actors, directors, genres, release dates, ratings, and more, dating up to the year 2019. Similarly, the individual dataset provides detailed information on actors and directors, forming a rich basis for analysis.

### Rationale for Dataset Selection

**Richness and Depth:** The chosen dataset offers an extensive range of fields covering nearly every aspect of movie information. This depth facilitates a multifaceted analysis of movies based on various attributes such as genre, directorial style, cast ensemble, and audience reception.

**Temporal Span and Volume:** The dataset's coverage of movies up to 2019, including unreleased titles, provides a wide temporal span for trend analysis. The substantial volume of records ensures statistical significance in data-driven insights.

**Potential for Network Analysis:** With detailed actor and director information linked to movies, the dataset is primely suited for constructing complex relational networks. This aspect is crucial for our goal of building a knowledge graph that mirrors the intricate web of relationships in the film industry.

## Database Selection

### Choice of Database: Neo4j

**Neo4j** is a high-performance graph database management system renowned for its efficiency in handling highly connected data. It supports storing vast amounts of data in graph structures and executing complex queries with high efficiency.

### Justification for Database Selection

**Optimized for Connected Data:** Neo4j's graph database structure is inherently designed to manage and query connected data. This feature aligns perfectly with our objective of analyzing relationships within the movie industry, making Neo4j an ideal choice.

**Scalability and Performance:** Neo4j offers robust scalability and performance capabilities, essential for handling our dataset's volume and complexity. Its ability to efficiently execute complex queries ensures that our recommendation system can operate in real-time.

**Community and Support:** Neo4j boasts a strong community and extensive documentation, providing valuable resources for development and troubleshooting. This support network is crucial for the successful implementation and maintenance of our system.

## Data Preprocessing and Import Steps

### Data Preprocessing

1. **Data Cleaning:** We will remove duplicate entries, handle missing values by imputation or deletion, and standardize formats across fields to ensure data consistency.
   
2. **Data Standardization:** Names and aliases will be unified under a single standard, and languages and regions will be standardized using ISO codes.

3. **Data Transformation:** Relationships between movies and individuals (actors/directors) will be explicitly defined, facilitating the construction of our knowledge graph.

### Data Import into Neo4j

1. **Formatting Data:** The preprocessed data will be formatted into CSV files or directly into Cypher statements, depending on the import method chosen.

2. **Utilizing Neo4j's Import Tools:** For bulk data importation, we will employ Neo4j's `neo4j-admin import` tool. For incremental updates or smaller datasets, the `LOAD CSV` Cypher command will be used.

3. **Building the Knowledge Graph:** Once imported, we will define entities (movies, actors, directors) and their relationships within Neo4j to form the backbone of our knowledge graph.




## Methodology

We proposed our method based on Reasoning on Graph method, using Knowledge Graphs to leverage the explaination and reasoning ability of LLMs. The framework is implemented by 2 modules, which are planning module and retieval-reasoning module. With jointly optimization, we are able to tune the LLM together with KG reasoning process.

### Planning
The reasoning on graphs framework enhances the reasoning capabilities of LLM through Planning-Retrieval-Reasoning process. Here, planning module is used to generate relation paths supported by knowledge graphs, which then be constructed to retrieve effective reasoning paths.
It can be considered as an optimization problem, and the optimization target formula for Planning-Retrieval-Reasoning is articulated as:

$$
P_\theta (a|q, G) = \sum_{z \in Z} P_\theta(a|q, z, G)P_\theta(z|q)
$$

where $(P_\theta(a|q, z, G))$ is for the probability of inferring answer **a**, given question **q**, relation path **z**, and a knowledge graph **G**, $P_\theta(z|q)$ is the probability of generating a reliable relation path z supported by G.

The objective of planning optimization is to distill knowledge from KGs into LLMs to generate reliable relation paths as plans. In this case, we minimize KL Divergence of the posterior distribution $Q(z)$ to approach effective relation paths. Evidence Lower Bound (ELBO) optimization formula is given by:

$$
\log P(a|q, G) \geq E_{z \sim Q(z)}[\log P_\theta(a|q, z, G)] - D_{KL}(Q(z) \| P_\theta(z|q))
$$

Where $Q(z)$ is for the posterior distribution of reliable relation paths supported by KGs, the first term encoraging generation of correct answer based on the relation path and KGs, the second term minimizes the KL divergence between the posterior and the prior, encouraging the generation of reliable relation paths. 

### Retrieval-Reasoning

Retrieval-reasoning optimization focuses on enabling LLMs to perform reasoning based on retrieved reasoning paths. This process utilizes the FiD framework, facilitating reasoning across multiple paths using

$$
P_\theta(a|q, Z, G) = \prod_{z \in Z} P_\theta(a|q, z, G)
$$

### Joint Optimization

As RoG framework using the modules metioned to improve LLM, we jointly optimize them together during training or tuning process. In this, we add KL loss of planning and the FiD loss of retrieval-reasoning as the final objective of our model.

$$
L = \log P_\theta(a|q, Z_{K^*} , G) + \frac{1}{|Z^*|} \sum_{z \in Z^*} \log P_\theta(z|q)
$$



## Expected Outcomes



## Timeline



## Conclusion



