from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization', methods=['POST'])
def visualization():
    # Getting topology data from form input
    num_spines = int(request.form['num_spines'])
    num_leaves = int(request.form['num_leaves'])
    num_hosts_per_leaf = int(request.form['num_hosts_per_leaf'])

    # Example of simple Leaf-Spine Topology Data Structure
    topology = {
        'spines': [f"Spine-{i+1}" for i in range(num_spines)],
        'leaves': [f"Leaf-{i+1}" for i in range(num_leaves)],
        'hosts': [f"Host-{i+1}" for i in range(num_leaves * num_hosts_per_leaf)],
        'bandwidth': {
            'spine_to_leaf': 100,
            'leaf_to_host': 50
        }
    }

    return render_template('visualization.html', topology=topology)

if __name__ == '__main__':
    app.run(debug=True)
