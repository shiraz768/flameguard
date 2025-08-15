from pymongo import MongoClient;




flameguard_client = MongoClient('mongodb://localhost:27017/');


flameguard_db = flameguard_client['flameguard_db']

users_collection = flameguard_db['users']
print("Connected to MongoDB with authentication")