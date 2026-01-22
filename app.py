from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# ---- IA ----
mensajes = []
etiquetas = []

ruta_datos = os.path.join(os.path.dirname(__file__), "datos.txt")
with open(ruta_datos, "r", encoding="utf-8") as f:
    for linea in f:
        etiqueta, texto = linea.strip().split(";")
        mensajes.append(texto)
        etiquetas.append(etiqueta)

vectorizador = CountVectorizer()
X = vectorizador.fit_transform(mensajes)

modelo = MultinomialNB()
modelo.fit(X, etiquetas)

# ---- WEB ----
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None

    if request.method == "POST":
        texto = request.form["texto"]
        frases = texto.split("\n")

        pos = neg = neu = 0

        for frase in frases:
            if frase.strip() == "":
                continue

            X_nuevo = vectorizador.transform([frase])
            pred = modelo.predict(X_nuevo)[0]

            if pred == "positivo":
                pos += 1
            elif pred == "negativo":
                neg += 1
            else:
                neu += 1

        if pos > neg:
            resumen = "MAYORMENTE ALEGRE 🙂"
        elif neg > pos:
            resumen = "MAYORMENTE TRISTE 😕"
        else:
            resumen = "NEUTRAL 😐"

        resultado = {
            "positivas": pos,
            "negativas": neg,
            "neutrales": neu,
            "resumen": resumen
        }

    return render_template("index.html", resultado=resultado)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


