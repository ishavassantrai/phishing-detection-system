# Phishing Detection System

This project was developed for my dissertation for the BSc (Hons) Cyber Security degree at Nottingham Trent University (2025). It analyses emails to determine whether they are phishing, with a focus on two types of scam: financial urgency and sextortion.

## Overview

The system examines each email in three different ways and combines the results to produce a final verdict of Phishing, Likely Phishing or Not Phishing.

- **Text analysis:** a RoBERTa model, fine-tuned to analyse the wording of the email
- **URL analysis:** a Random Forest model that examines any links or Bitcoin wallets in the email
- **Behavioural analysis:** an Isolation Forest model that identifies emails that behave differently from normal ones

If an email contains no links or Bitcoin wallet, the URL analysis is skipped. The system is presented through a simple Gradio interface, where an email can be pasted in and the result from each layer can be viewed.

## Files

- `UI-Detection-System.py`: the Gradio interface
- `roberta-training-model.py`, `URL-training-model.py` and `Behavioural-training-model.py`: the scripts used to train each model
- `full_dataset`: the dataset of emails created to train and test the system
- `roberta-model` and the `.pkl` files: the saved models and scalers

## Dataset

The dataset was created for this project and contains around 120 synthetic emails, made up of both phishing and legitimate examples. They are based on real scams identified during my research, but none of them are real emails.

## Running the System

The scripts require Python and the libraries they import, such as transformers, scikit-learn and gradio. Once these are installed, run `UI-Detection-System.py` and open the link provided in a web browser.

## Results

The system was tested on 12 example emails:

- All 7 phishing emails were detected (4 as Phishing and 3 as Likely Phishing)
- 3 out of 5 legitimate emails were correctly classified as Not Phishing
- 2 legitimate emails were incorrectly classified as Likely Phishing: a genuine Google password reset and an HR email containing grammatical errors

One of the phishing examples was a highly polished fake recruiter email. The RoBERTa model scored it at only 17%, but the URL and behavioural analyses scored it at 76% and 91%, so it was still detected. This shows the benefit of combining multiple layers rather than relying on a single model.

## Limitations

- The dataset is small and made up of synthetic emails, so the results are not a reliable measure of real-world performance
- Some legitimate emails were flagged as phishing, so the scoring would need further tuning
- The system only covers financial urgency and sextortion emails, not other types of phishing

## About

Isha Vassantrai, BSc (Hons) Cyber Security, Nottingham Trent University, First Class Honours (2025)
