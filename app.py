from flask import Flask, render_template, request, redirect, url_for, send_file, session
import os
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
from io import BytesIO
from fpdf import FPDF
import json
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '1234'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens


import re

def extract_customer_requirements(doc):
    requirements = {"car_type": None, "fuel_type": None, "color": None, "budget": None, "transmission_type": None}
    
    # Prioritize specific car types first (e.g., "Compact SUV" before "SUV")
    car_types = ["Compact SUV", "SUV", "Sedan", "Hatchback", "Coupe", "Convertible", "Wagon", "Van", "Truck"]
    colors = ["white", "black", "blue", "red", "green", "yellow", "silver", "grey", "gray", "gold", "orange", "brown"]

    # Convert the text to lowercase for case-insensitive matching
    doc_text_lower = doc.text.lower()

    # Match car type by checking from most specific to general
    for car_type in car_types:
        if car_type.lower() in doc_text_lower:
            requirements["car_type"] = car_type
            break

    # Match fuel type
    if "diesel" in doc_text_lower:
        requirements["fuel_type"] = "Diesel"
    elif "petrol" in doc_text_lower:
        requirements["fuel_type"] = "Petrol"

    # Match transmission type
    if "manual" in doc_text_lower:
        requirements["transmission_type"] = "Manual"
    elif "automatic" in doc_text_lower:
        requirements["transmission_type"] = "Automatic"

    # Match color
    for color in colors:
        if color in doc_text_lower:
            requirements["color"] = color
            break

    # Improved budget extraction using context-specific regex
    budget_match = re.search(r'budget\s*of\s*rs\.?\s?(\d[\d,]*\d)', doc_text_lower)
    if budget_match:
        requirements["budget"] = budget_match.group(1).replace(",", "")
    else:
        # If "budget" isn't found, fall back to the last number preceded by 'Rs'
        fallback_budget = re.findall(r'rs\.?\s?(\d[\d,]*\d)', doc_text_lower)
        if fallback_budget:
            requirements["budget"] = fallback_budget[-1].replace(",", "")

    return requirements


def extract_customer_objections(doc):
    objections = {"refurbishment_quality": None, "car_issues": None, "price_issues": None, "customer_experience_issues": None}
    if "problems" in doc.text.lower() or "issues" in doc.text.lower():
        objections["car_issues"] = "Customer mentioned issues or problems with cars."
    if "price" in doc.text.lower():
        objections["price_issues"] = "Customer mentioned concerns about price."
    if "trust" in doc.text.lower() or "maintained" in doc.text.lower():
        objections["refurbishment_quality"] = "Customer concerned about refurbishment quality."
    return objections

def process_conversation(conversation):
    doc = nlp(conversation)
    customer_requirements = extract_customer_requirements(doc)
    customer_objections = extract_customer_objections(doc)
    company_policies = {
        "free_rc_transfer": "free rc transfer" in conversation.lower(),
        "money_back_guarantee": "money-back guarantee" in conversation.lower(),
        "free_rsa": "free rsa" in conversation.lower(),
        "return_policy": "return policy" in conversation.lower()
    }
    result = {
        "customer_requirements": customer_requirements,
        "company_policies": company_policies,
        "customer_objections": customer_objections
    }
    return result

def analyze_conversations(results):
    # Aggregate results
    colors = []
    car_types = []
    budgets = []
    objections = {"refurbishment_quality": 0, "car_issues": 0, "price_issues": 0, "customer_experience_issues": 0}

    for result in results:
        if result['customer_requirements']['color']:
            colors.append(result['customer_requirements']['color'])
        if result['customer_requirements']['car_type']:
            car_types.append(result['customer_requirements']['car_type'])
        if result['customer_requirements']['budget']:
            budgets.append(result['customer_requirements']['budget'])
        for key in objections.keys():
            if result['customer_objections'][key]:
                objections[key] += 1
    
    return colors, car_types, budgets, objections

def create_visualizations(colors, car_types, budgets, objections):
    # Visualization: Distribution of Most Requested Car Colors
    if len(set(colors)) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(colors, bins=len(set(colors)), color='skyblue')
        plt.title("Distribution of Most Requested Car Colors")
        plt.xlabel("Color")
        plt.ylabel("Frequency")
        plt.savefig("static/colors_distribution.png")
        plt.clf()
    else:
        print("No colors to visualize.")

    # Visualization: Preferred Car Types
    if len(set(car_types)) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(car_types, bins=len(set(car_types)), color='lightgreen')
        plt.title("Preferred Car Types")
        plt.xlabel("Car Type")
        plt.ylabel("Frequency")
        plt.savefig("static/car_types_distribution.png")
        plt.clf()
    else:
        print("No car types to visualize.")

    # Visualization: Popular Price Ranges
    if len(budgets) > 0:
        budget_values = [int(b.replace('rs', '').replace(',', '').strip()) for b in budgets if b and b.lower() != "salesperson"]
        if len(budget_values) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(budget_values, bins=10, color='orange')
            plt.title("Popular Price Ranges")
            plt.xlabel("Price (in Rs)")
            plt.ylabel("Frequency")
            plt.savefig("static/budget_distribution.png")
            plt.clf()
        else:
            print("No valid budgets to visualize.")
    else:
        print("No budgets to visualize.")

    # Visualization: Common Refurbishment Issues and Objections
    if sum(objections.values()) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(objections.keys(), objections.values(), color='red')
        plt.title("Common Refurbishment Issues and Objections")
        plt.xlabel("Objection Type")
        plt.ylabel("Frequency")
        plt.savefig("static/objections_distribution.png")
        plt.clf()
    else:
        print("No objections to visualize.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files')
        if not files:
            return redirect(request.url)
        
        all_results = []
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations = f.read().split('---')  # Split conversations by delimiter '---'
                results = [process_conversation(conversation) for conversation in conversations]
                all_results.extend(results)

        session['results'] = all_results  # Store in session

        # Save results to JSON
        with open('labeled_conversations.json', 'w') as json_file:
            json.dump(all_results, json_file, indent=4)

        colors, car_types, budgets, objections = analyze_conversations(all_results)
        create_visualizations(colors, car_types, budgets, objections)

        return render_template('analysis.html', results=all_results, colors=colors, car_types=car_types, budgets=budgets, objections=objections)
    return render_template('index.html')

@app.route('/download/<file_type>')
def download_file(file_type):
    results = session.get('results')  # Retrieve from session

    if not results:
        return redirect(url_for('index'))

    if file_type == 'csv':
        # Export analysis results to CSV
        csv_output = "color,car_type,budget,refurbishment_quality,car_issues,price_issues,customer_experience_issues\n"
        for result in results:
            csv_output += f"{result['customer_requirements']['color']},{result['customer_requirements']['car_type']},{result['customer_requirements']['budget']},{result['customer_objections']['refurbishment_quality']},{result['customer_objections']['car_issues']},{result['customer_objections']['price_issues']},{result['customer_objections']['customer_experience_issues']}\n"
        
        return send_file(BytesIO(csv_output.encode()), mimetype="text/csv", as_attachment=True, download_name="analysis.csv")

    elif file_type == 'pdf':
        # Export analysis results to PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Car Sales Conversation Analysis", ln=True, align='C')

        # Add images
        pdf.image("static/colors_distribution.png", x=10, y=20, w=100)
        pdf.image("static/car_types_distribution.png", x=10, y=80, w=100)
        pdf.image("static/budget_distribution.png", x=10, y=140, w=100)
        pdf.image("static/objections_distribution.png", x=10, y=200, w=100)
        
        # Generate PDF in memory
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name="analysis.pdf")

    elif file_type == 'json':
        # Export analysis results to JSON
        json_output = json.dumps(results, indent=4)
        return send_file(BytesIO(json_output.encode()), mimetype='application/json', as_attachment=True, download_name="analysis.json")

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
