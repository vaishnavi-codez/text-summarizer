
from transformers import pipeline

# Create a summarization pipeline
text_summarizer = pipeline("summarization")

# Input paragraph for summarization
input_paragraph = """
Natural Language Processing (NLP) is a dynamic and interdisciplinary field within artificial intelligence (AI) that focuses on enabling computers to understand, interpret, generate, and respond to human language in a way that is both meaningful and useful. It combines elements of computer science, linguistics, and machine learning to allow machines to process language data, whether written or spoken. The importance of NLP has grown rapidly in the modern digital age, where vast amounts of data are generated in natural language every day through emails, social media, websites, and spoken interactions. NLP aims to make sense of this data, allowing machines to perform tasks that once required human intelligence, such as understanding queries, extracting information, or translating languages.

At the core of NLP are several foundational tasks that contribute to its effectiveness. One such task is **tokenization**, which involves breaking down text into individual words or phrases. This is often followed by **part-of-speech (POS) tagging**, where each word is labeled with its grammatical role, such as noun or verb. **Named Entity Recognition (NER)** is another important task that identifies proper names in text, such as people, places, or organizations. NLP systems also perform **sentiment analysis**, which detects the emotional tone of a piece of text, determining whether the sentiment expressed is positive, negative, or neutral. These tasks help computers better understand the structure and meaning of human language.

Beyond understanding, NLP also involves generating human-like language. This is seen in applications like chatbots and virtual assistants, such as Siri, Alexa, or Google Assistant, which can respond to questions, set reminders, or carry on conversations. Another major application of NLP is **machine translation**, where systems like Google Translate convert text from one language to another. NLP is also used in **speech recognition** technologies that convert spoken words into written text, enabling voice-to-text features and voice commands. Furthermore, **text summarization** tools use NLP to condense large documents into shorter summaries, making it easier for users to grasp the main points quickly.

Advances in machine learning, particularly deep learning, have significantly improved the performance of NLP systems. Modern NLP models, such as BERT, GPT, and T5, use large neural networks trained on massive datasets to understand language contextually. These models are capable of understanding nuances, slang, idioms, and even sarcasm to a certain degree, which was previously very difficult for machines to achieve. Such capabilities have made NLP an essential component in a wide range of industries, including healthcare, finance, education, customer service, and cybersecurity.
"""

# Generate summary
nlp_summary = text_summarizer(input_paragraph, max_length=50, min_length=25, do_sample=False)

# Print the summary
print("Summary:")
print(nlp_summary[0]['summary_text'])



