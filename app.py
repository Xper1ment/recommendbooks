import pickle ,json,math
from flask import Flask,jsonify,request,abort
from flask_cors import CORS
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#from flask.ext.aiohttp import async

app = Flask(__name__)
CORS(app)

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.route('/')

def default():
    return "<h1>Welcome</h1>"

with open('./data/book_model.pkl', 'rb') as f:
    cosine_sim_corpus = pickle.load(f)

with open('./data/book.pkl','rb') as f:
    books = pickle.load(f)


books = books.reset_index()

#cosine_sim_corpus = cosine_similarity(tfidf_matrix_corpus, tfidf_matrix_corpus)

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = {'name': value , 'website': dict1[key]}   
    return dict3

# Build a 1-dimensional array with book titles
titles = books['title']
url = books['url']
indices1 = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def recomm_books(title):
    try:
        idx = indices1[title]
        sim_scores = list(enumerate(cosine_sim_corpus[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        book_indices = [i[0] for i in sim_scores]
        urls = url.iloc[book_indices].to_dict()
        name = titles.iloc[book_indices].to_dict()
        ans = mergeDict(urls,name)
        keys_values = ans.items()
        result = {str(key): value for key, value in keys_values}
        return result
    except KeyError:
        pass

@app.route('/recommend_books/',methods=['GET'])

def recommend_books():
    title = request.args.get("title", None)
    result = recomm_books(title)
    if result == None:
        return jsonify({"error_code":"404" , "message":"No title found"})
    else:
        return jsonify(result)


if __name__ == "__main__":
    app.run(debug = False ,threaded = True)    
