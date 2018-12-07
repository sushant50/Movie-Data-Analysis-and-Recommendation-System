# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:47:57 2018

@author: rishi
"""


import script as sc
from flask import Flask,request
from flask import jsonify

from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Recommender(Resource):
    def get(self):
        print("Top Movies")
        movie_name = request.args.get('movie')
        df = sc.genre_recommendations(movie_name)
        return df.to_json()
        
    
api.add_resource(Recommender, '/recommender') # Route_1

if __name__ == '__main__':
    app.run(port='5002')