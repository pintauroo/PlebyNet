import pytest
from app import app

# Fixture to set up a test client for Flask
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test the index route ('/')
def test_index(client):
    """Test the index route renders the form page."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Configure Topology" in response.data  # Check if the title exists in the response HTML

def test_visualization(client):
    """Test the visualization route returns the topology page with the correct data."""
    # Prepare data to send in the form
    form_data = {
        'num_spines': 2,
        'num_leaves': 2,
        'num_hosts_per_leaf': 3
    }

    # Send a POST request to the '/visualization' route
    response = client.post('/visualization', data=form_data)

    # Check if the response status code is OK (200)
    assert response.status_code == 200

    # Check if the rendered template is 'visualization.html'
    assert b"Network Topology Visualization" in response.data

    # Check if JavaScript or JSON data is embedded in the page
    assert b"var topologyData" in response.data  # Check if topologyData is embedded in a <script> tag

    # Additional check: Verify the presence of spines, leaves, and hosts in the JavaScript variable
    assert b"Spine-1" in response.data
    assert b"Spine-2" in response.data
    assert b"Leaf-1" in response.data
    assert b"Leaf-2" in response.data
    assert b"Host-1" in response.data
    assert b"Host-2" in response.data
    assert b"Host-3" in response.data
    assert b"Host-4" in response.data
    assert b"Host-5" in response.data
    assert b"Host-6" in response.data

    # Optionally, you can print out the response to inspect the data
    # print(response.data.decode('utf-8'))  # Uncomment for debugging if necessary


# Test the visualization route with invalid form data (e.g., missing fields)
def test_visualization_invalid(client):
    """Test the visualization route with invalid form data."""
    # Missing 'num_hosts_per_leaf'
    form_data = {
        'num_spines': 2,
        'num_leaves': 2,
    }
    response = client.post('/visualization', data=form_data)
    assert response.status_code == 400  # Expecting bad request because of missing data
    assert b"Missing required fields" in response.data  # Check if a validation message exists

# Test the form validation when data is not provided (edge case)
def test_form_validation(client):
    """Test if form validation works for missing input fields."""
    # Send an empty POST request without data
    response = client.post('/visualization', data={})
    assert response.status_code == 400  # Expecting a bad request
    assert b"Missing required fields" in response.data  # Check for error message
