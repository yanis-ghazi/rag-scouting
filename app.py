import gradio as gr
import os
import sys

# Ajoute le dossier src au path pour pouvoir importer nos modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag_engine import init_components, ask

# Initialisation au démarrage de l'app (une seule fois)
print("Démarrage de l'application...")
model, collection, groq_client = init_components()
print("✅ Application prête !")


def respond(question):
    """Fonction appelée par Gradio quand l'utilisateur pose une question."""
    if not question.strip():
        return "❌ Veuillez entrer une question."
    try:
        answer = ask(question, model, collection, groq_client)
        return answer
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


# ============================================================
# Interface Gradio
# ============================================================

exemples = [
    "Quel joueur NBA sous 25 ans a le plus d'assists cette saison ?",
    "Trouve moi un meneur NBA avec plus de 8 assists et moins de 3 turnovers",
    "Quel est le meilleur buteur de Premier League cette saison ?",
    "Quel est le meilleur passeur décisif de Premier League cette saison ?",
    "Trouve moi un défenseur de Premier League de moins de 23 ans",
    "Quel joueur NBA marque plus de 25 points avec plus de 50% au tir ?",
    "Qui sont les meilleurs rebondeurs NBA cette saison ?",
]

with gr.Blocks(
    title="🏆 RAG Scouting Sport",
    theme=gr.themes.Soft(),
) as demo:

    # Header
    gr.Markdown("""
    # 🏆 RAG Scouting Sport
    ### Posez vos questions en langage naturel sur les stats NBA et Premier League 2024/25
    *Propulsé par Groq (Llama 3.3) + ChromaDB + Sentence Transformers*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Zone de question
            question_input = gr.Textbox(
                label="🔍 Votre question",
                placeholder="Ex: Quel joueur NBA sous 25 ans a le plus d'assists ?",
                lines=2,
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Rechercher", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Effacer", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 À propos des données")
            gr.Markdown("""
            - 🏀 **508 joueurs NBA** (saison 2024/25)
            - ⚽ **539 joueurs PL** (saison 2024/25)
            - Stats : points, assists, rebonds, buts, passes déc...
            """)

    # Zone de réponse
    answer_output = gr.Markdown(
        label="💬 Réponse",
        value="*La réponse apparaîtra ici...*"
    )

    # Exemples cliquables
    gr.Markdown("### 💡 Exemples de questions")
    gr.Examples(
        examples=exemples,
        inputs=question_input,
        label="Cliquez pour essayer"
    )

    # Séparateur
    gr.Markdown("---")
    gr.Markdown("""
    *Projet portfolio RAG — Stack : Groq API + ChromaDB + Sentence Transformers + Gradio*
    """)

    # Actions des boutons
    submit_btn.click(
        fn=respond,
        inputs=question_input,
        outputs=answer_output,
        show_progress=True
    )

    question_input.submit(
        fn=respond,
        inputs=question_input,
        outputs=answer_output,
        show_progress=True
    )

    clear_btn.click(
        fn=lambda: ("", "*La réponse apparaîtra ici...*"),
        outputs=[question_input, answer_output]
    )


if __name__ == "__main__":
    demo.launch(
        share=False,      # True = lien public temporaire
        show_error=True,
    )