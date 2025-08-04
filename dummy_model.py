import random

CLASSES = ["Acne", "Eczema", "Urticaria", "Healthy", "Allergy"]

def dummy_predict(image):
    prediction = random.choice(CLASSES)
    confidence = round(random.uniform(0.7, 0.99), 2)
    return prediction, confidence
