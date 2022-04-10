## Background
The objective of this project is to build a data pipeline related to security events in Nigeria using Natural Language Processing. The data pipeline should include security events as reported in the media. Examples of security events of interest include terrorism, kidnapings, mass shooting, looting, riots.. Data will consist of historical events (batch). The topics of each extracted news content would be ascertained using an NLP unsupervised classification/ clustering algorithm known a s LDA Topic Modeling. Topic modeling is a method for unsupervised classification of documents, similar to clustering on numeric data, which finds some natural groups of items (topics) even when we’re not sure what we’re looking for.

In a broader definition, it is about logically correlating several words. Say a telecom operator wants to identify whether the poor network is a reason for low customer satisfaction. Here, “bad network” is the topic. The document is analyzed for words like “bad”, “slow speed”, “call not connecting”, etc., which are more likely to describe network issues compared to common words like “the” or “and”. N/B - We can find relevant topics in a document by simply querying without actually going through the entire document.

Use this link to visit the deployed demo app : https://securityai.herokuapp.com/
