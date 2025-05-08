
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Model ve scaler
model_failure = joblib.load("mlp_failure_model.pkl")
model_treatment = joblib.load("mlp_treatment_model.pkl")
scaler = joblib.load("scaler.pkl")
failure_label = joblib.load("failure_label.pkl")
treatment_label = joblib.load("treatment_label.pkl")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/form-predict/", response_class=HTMLResponse)
async def form_predict(
    request: Request,
    Age: int = Form(...),
    Sex: int = Form(...),
    Weight_kg: float = Form(...),
    Height_cm: float = Form(...),
    RR_pre: int = Form(...),
    HR_pre: int = Form(...),
    SBP_pre: int = Form(...),
    DBP_pre: int = Form(...),
    SpO2_pre: float = Form(...),
    GCS_pre: int = Form(...),
    pH_pre: float = Form(...),
    pCO2_pre: float = Form(...),
    pO2_pre: float = Form(...),
    HCO3_pre: float = Form(...),
    BE_pre: float = Form(...),
    Lactate_pre: float = Form(...),
    FiO2_pre: int = Form(...),
    Accessory_Muscle_Use: int = Form(...),
    Clinical_Diagnosis: int = Form(...),
    Primary_Complaint: int = Form(...)
):
    input_data = np.array([[Age, Sex, Weight_kg, Height_cm, RR_pre, HR_pre, SBP_pre, DBP_pre,
                            SpO2_pre, GCS_pre, pH_pre, pCO2_pre, pO2_pre, HCO3_pre, BE_pre,
                            Lactate_pre, FiO2_pre, Accessory_Muscle_Use, Clinical_Diagnosis,
                            Primary_Complaint]])
    input_scaled = scaler.transform(input_data)
    failure_pred = model_failure.predict(input_scaled)[0]
    treatment_pred = model_treatment.predict(input_scaled)[0]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "failure_result": failure_label.inverse_transform([failure_pred])[0],
        "treatment_result": treatment_label.inverse_transform([treatment_pred])[0]
    })
