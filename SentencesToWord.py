from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# could you tell me about python and some information about that .
# in my file explorer search about all folders . 
# who is robert downy jr ?
# A man brings a magical monkey’s paw from India, which grants three wishes to three people. When the White family buys it from him, they realize that sometimes you do not want your wishes to come true. What Is Great About It: Sometimes we wish for things, but we do not think about their consequences. In this story, the characters immediately regret when their wishes come true because either someone dies or something worse happens. They realize that they never thought about the ways their wishes could destroy people and their lives.

words = []
sentences = "A man brings a magical monkey’s paw from India, which grants three wishes to three people. When the White family buys it from him, they realize that sometimes you do not want your wishes to come true. What Is Great About It: Sometimes we wish for things, but we do not think about their consequences. In this story, the characters immediately regret when their wishes come true because either someone dies or something worse happens. They realize that they never thought about the ways their wishes could destroy people and their lives."
words.append(sentences)
for i in sentences.split() : 
    words.append(i)

model = SentenceTransformer(r"C:\Users\smir1\Downloads\all-MiniLM-L12-v1")
print("endcoding..")
embeddings = model.encode(words)

encod_list = []

for g in enumerate(words[1::]) : 
    encod_list.append((cosine_similarity(embeddings[0].reshape(1,-1) , embeddings[g[0]+1].reshape(1,-1))[0][0] , g[1]))

print("this text is about " , max(encod_list))

for i in encod_list : 
    if i[0] >= 0.3 : 
        print(i)
