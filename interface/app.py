from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization', methods=['POST'])
def visualization():
    # Getting topology data from form input
    num_spines = request.form.get('num_spines')
    num_leaves = request.form.get('num_leaves')
    num_hosts_per_leaf = request.form.get('num_hosts_per_leaf')

    # Check if all required fields are present
    if not num_spines or not num_leaves or not num_hosts_per_leaf:
        return "Missing required fields", 400

    # Convert values to integers
    num_spines = int(num_spines)
    num_leaves = int(num_leaves)
    num_hosts_per_leaf = int(num_hosts_per_leaf)

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

    # Debugging step: Check if form values are processed correctly
    print(f"num_spines: {num_spines}, num_leaves: {num_leaves}, num_hosts_per_leaf: {num_hosts_per_leaf}")
    
    # Return the 'visualization.html' template with the topology data
    return render_template('visualization.html', topology=topology)

if __name__ == '__main__':
    app.run(debug=True)
