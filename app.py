from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the models
AV = pickle.load(open('AV.pkl', 'rb'))
Gmb = pickle.load(open('Gmb.pkl', 'rb'))
VMA = pickle.load(open('VMA.pkl', 'rb'))
VFB = pickle.load(open('VFB.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Redirect to the input page with the specified model
        return redirect(url_for('input_page', model='volumetrics'))
        
    return render_template('index.html')


@app.route('/input_page/<model>', methods=['GET', 'POST'])
def input_page(model):
    pred1, pred2, pred3, pred4 = None, None, None, None  # Initialize predictions

    if request.method == 'POST':
        # Retrieve form data
        aggregate = request.form['Aggregate']
        source = request.form['source']
        
        try:
            viscosity = float(request.form['viscosity'])  # Convert viscosity to float
        except ValueError:
            return render_template('input_page.html', model=model, error="Viscosity must be a number.")
        
        dag = request.form['DAG']
        compaction = request.form['compaction']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, compaction]])

        # Make prediction using the selected models
        if model == 'volumetrics':
            pred1 = np.round(AV.predict(input_data), 2)
            pred2 = np.round(Gmb.predict(input_data), 2)
            pred3 = np.round(VMA.predict(input_data), 2)
            pred4 = np.round(VFB.predict(input_data), 2)
            
            # Return the rendered template with predictions
            return render_template('input_page.html', model=model, pred1=pred1, pred2=pred2, pred3=pred3, pred4=pred4)

    return render_template('input_page.html', model=model, pred1=pred1, pred2=pred2, pred3=pred3, pred4=pred4)

if __name__ == '__main__':
    app.run(debug=True)
