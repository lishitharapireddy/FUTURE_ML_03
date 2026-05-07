import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'candidate_name': [
        'John',
        'Alice',
        'Bob'
    ],
    'resume_text': [
        'Python Machine Learning SQL',
        'Java Web Development',
        'Python Deep Learning Data Analysis'
    ]
}

df = pd.DataFrame(data)

# Job description
job_description = """
Python Machine Learning Data Analysis
"""

# Combine documents
documents = [job_description] + df['resume_text'].tolist()

# Convert text to vectors
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(documents)

# Similarity scores
similarity = cosine_similarity(matrix[0:1], matrix[1:])

# Add scores
df['score'] = similarity[0]

# Sort candidates
ranked = df.sort_values(by='score', ascending=False)

print("Top Candidates:")
print(ranked[['candidate_name', 'score']])