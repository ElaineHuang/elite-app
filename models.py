from flask import Flask
from flask.ext.mongoalchemy import MongoAlchemy

app = Flask(__name__)


def init_db_app(app):
    # app.config['MONGO_DBNAME'] = 'eb8c4f80-3ec6-428d-a4df-ce2011778d7c'
    # app.config['MONGO_URI'] = 'mongodb://d933c021-d8d3-402b-aad2-bb67e8d40601:iEGGAujhUObOBJyFeakrOt3T3@192.168.100.12:27017/eb8c4f80-3ec6-428d-a4df-ce2011778d7c'

    #localhost db for test
    app.config['MONGOALCHEMY_DATABASE'] = 'tic100'
    app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017'
    db  = MongoAlchemy(app)  

    return db

db = init_db_app(app)

class TestCollection(db.Document):
    number = db.IntField()
    name = db.StringField()


# class Book(db.Document):
#     title = db.StringField()
#     author = db.DocumentField(Author)
#     year = db.IntField()

