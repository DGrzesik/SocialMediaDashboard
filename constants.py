METHOD_NAMES = ['t-SNE', 'PCA', 'PaCMAP', 'TriMAP', 'UMAP']
METRICS = ['euclidean', 'manhattan']
TARGETS = ['topic', 'sentiment']
ENGAGEMENT_FEATURES = ['likes', 'reach', 'isreshare', 'repostcount', 'klout']
START_SCREEN = """
Welcome to our dashboard!
\n
\nIn order to perform Social Media Interaction Analysis, please upload your data.
\n
\nYou can upload multiple files as long as their columns are the same.
\nAccepted columns include:
    \n- UserID
    \n- PostID
    \n- Language    
    \n- Text    
    \n- Likes
    \n- Reach
    \n- IsReshare
    \n- RepostCount
    \n- Klout
    \n- Topic
    \n- Sentiment
\nThe names are case-insensitive. Not all columns are required for each analysis.
You will be informed about available options after uploading your file.
\n
\nHave fun!
"""
