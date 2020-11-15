from flask import *
from deploy import get_label

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/result", methods = ['POST', 'GET'])
def result():
    f = request.files['file']
    f.save(f.filename)
    name = get_label(f.filename)
    # result_ans = get_label(name)

    return render_template("result.html", name = f.filename, result = name)


if __name__ == "__main__":  
    app.run(debug = True)