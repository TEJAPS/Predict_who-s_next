
import os
from flask import Flask, jsonify,render_template,request
from predict import callme
import h5py
app = Flask(__name__)

@app.route('/')
def Welcome():
    return app.send_static_file('index.html')

@app.route("/take")
def take():
	return render_template("take.html")

@app.route('/myapp',methods=['POST','GET'])
def WelcomeToMyapp():
    framework = request.form['text1']
    return  " <h1 style='width:100%; height:100%'>"+ callme(framework) +"<h1><a action='/take'> Check One More </a>"

@app.route('/api/people')
def GetPeople():
    list = [
        {'name': 'John', 'age': 28},
        {'name': 'Bill', 'val': 26}
    ]
    return jsonify(results=list)

@app.route('/api/people/<name>')
def SayHello(name):
    message = {
        'message': 'Hello ' + name
    }
    return jsonify(results=message)

port = os.getenv('PORT', '5000')
if __name__ == "__main__":
	app.run(host='127.0.0.1', port=int(port))
