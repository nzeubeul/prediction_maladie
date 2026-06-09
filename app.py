from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'dataset_medical_final_sans_sexe.csv'

symptomes = [
    'Fievre', 'Frissons', 'Céphalées', 'Polyurie', 'Douleur_Lombaire',
    'Jet_Urine_Faible', 'Saignement_Vaginal', 'Douleur_Pelvienne',
    'Diarrhée_Aqueuse', 'Déshydratation', 'Raideur_Nuque', 'Photophobie',
    'Polydipsie', 'Oedèmes', 'Oligurie', 'Vertiges', 'Palpitations',
    'Masse_Mammaire', 'Modification_Peau_Sein', 'Ecoulement_Mamelon',
    'Retraction_Mamelon', 'Ganglion_Aisselle'
]

app = Flask(__name__)


def load_model():
    df = pd.read_csv(DATASET_PATH)
    df = df.copy()
    le = LabelEncoder()
    df['Diagnostic_Final'] = df['Diagnostic_Final'].astype(str)
    df['Diagnostic_Code'] = le.fit_transform(df['Diagnostic_Final'])

    X = df[symptomes]
    y = df['Diagnostic_Code']

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X, y)

    return df, model, le


df, model, label_encoder = load_model()
class_names = list(label_encoder.classes_)


def build_patient_input(selected_symptoms):
    return pd.DataFrame([{s: int(s in selected_symptoms) for s in symptomes}])


def find_exact_match(selected_symptoms):
    patient_df = build_patient_input(selected_symptoms)
    matched = df[df[symptomes].eq(patient_df.iloc[0]).all(axis=1)]
    if matched.empty:
        return None
    exact_counts = (
        matched['Diagnostic_Final']
        .value_counts(normalize=True)
        .mul(100)
        .rename_axis('disease')
        .reset_index(name='percent')
    )
    return exact_counts


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    result = None
    selected_symptoms = []

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        if not selected_symptoms:
            error = 'Veuillez sélectionner au moins un symptôme.'
        else:
            exact_result = find_exact_match(selected_symptoms)
            if exact_result is not None:
                predictions = [
                    {
                        'disease': row['disease'],
                        'percent': float(row['percent'])
                    }
                    for _, row in exact_result.iterrows()
                ]
                probable = predictions[0]
                result = {
                    'type': 'exact',
                    'predictions': predictions,
                    'probable': probable
                }
            else:
                patient_input = build_patient_input(selected_symptoms)
                probs = model.predict_proba(patient_input)[0]
                order = np.argsort(probs)[::-1]
                predictions = [
                    {
                        'disease': class_names[i],
                        'percent': float(probs[i] * 100)
                    }
                    for i in order[:5]
                ]
                probable = predictions[0]
                result = {
                    'type': 'probable',
                    'predictions': predictions,
                    'probable': probable
                }

    return render_template(
        'index.html',
        symptoms=symptomes,
        selected_symptoms=selected_symptoms,
        result=result,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
