import streamlit as st
from pysentimiento import create_analyzer
from openai import OpenAI
from pythonosc.udp_client import SimpleUDPClient
from transformers import MarianMTModel, MarianTokenizer

# Configura il client di OpenAI con la tua API key
client = OpenAI(api_key='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')
# Funzione per mostrare i messaggi dell'utente con bordi arrotondati
def user_message(message):
    st.markdown(f'<div class="user-message" style="display: flex; justify-content: flex-end; padding: 5px;">'
                f'<div style="background-color: #196b1c; color: white; padding: 10px; border-radius: 10px;font-family:"Open Sauce One";font-size:18px; margin-bottom:10px; margin-left:20px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)

# Funzione per mostrare i messaggi del bot con bordi arrotondati
def bot_message(message):
    st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #074c85; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-right:20px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)

class ChatGUI:
    def __init__(self):
        self.emotion_analyzer = create_analyzer(task="emotion", lang="it")
        self.translation_model = None  # Initialize to None, will be set later
        self.translation_tokenizer = None  # Initialize to None, will be set later
        self.osc_client = SimpleUDPClient("127.0.0.1", 23456)  # Initialize OSC client with appropriate IP address and port

    def send_message(self, message):
        emotion_result = self.analyze_emotion(message)
        modification_suggestion, anger_score = self.suggest_modification(emotion_result, message)
        return modification_suggestion, anger_score

    def analyze_emotion(self, message):
        emotion_result = self.emotion_analyzer.predict(message)
        return emotion_result

    def suggest_modification(self, emotion_result, user_message):
        anger_score = emotion_result.probas.get('anger', 0.0)
        modified_message = ""

        # Modifica il messaggio dell'utente solo se l'Anger Score è alto
        if anger_score > 0.7:
            # Se l'Anger Score è alto, usa l'API per ottenere una risposta riscritta
            prompt = f'Analizza il messaggio dell\'utente: "{user_message}". Se l\'Anger Score è superiore a 0.7, riscrivi il messaggio in modo da ridurre lo score della rabbia.'

            # Effettua una richiesta utilizzando l'API di OpenAI
            api_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Estrai il testo modificato dalla risposta API
            modified_message = api_response.choices[0].message.content.strip()
        
        self.message_color = self.get_color_from_anger_score(anger_score)
        self.color_name = self.get_color_name_from_anger_score(anger_score)

        # Aggiungi il colore al messaggio modificato
        modified_message = f"Il tuo messaggio verrà visualizzato in: {self.color_name}. {modified_message}"

        return modified_message, anger_score

    def translate_to_italian(self, text):
        inputs = self.translation_tokenizer(text, return_tensors="pt")
        outputs = self.translation_model.generate(**inputs)
        translated_text = self.translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_text[0]

    def publish_data(self, user_message, anger_score, publish_button_pressed=False):
        # Converti l'Anger Score in un numero
        anger_score_as_number = float(anger_score)

        # Invia il messaggio dell'utente a TouchDesigner tramite OSC
        self.osc_client.send_message("/user_message", [user_message, anger_score_as_number])
        st.write(f"Messaggio dell'utente pubblicato: {user_message}")

    def get_color_name_from_anger_score(self, anger_score):
        # Definisci il colore in base all'Anger Score
        if 0 <= anger_score <= 0.33:
            return "blu"
        elif 0.33 < anger_score <= 0.66:
            return "grigio"
        else:
            return "arancio"

    def get_color_from_anger_score(self, anger_score):
        # Definisci il colore in base all'Anger Score
        if 0 <= anger_score <= 0.33:
            return "r=0 g=1 b=0"  # verde
        elif 0.33 < anger_score <= 0.66:
            return "r=0.5 g=0.5 b=0.5"  # Grigio
        else:
            return "r=1 g=0 b=0"  # rosso


# Definisci l'app principale di Streamlit
def main():
    st.title("Sentiment Analysis Chat")

    # Inizializza la cronologia della chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_gui = ChatGUI()

    # Inizializza anger_score nello stato della sessione
    if "anger_score" not in st.session_state:
        st.session_state.anger_score = 0.0

    # Campo di input per l'utente per inserire un messaggio
    user_input = st.text_input("Inserisci il messaggio:")

    # Bottone per inviare il messaggio dell'utente
    if st.button("Analizza"):
        if user_input:
            modification_suggestion, anger_score = chat_gui.send_message(user_input)

            # Memorizza l'anger_score nello stato della sessione
            st.session_state.anger_score = anger_score

            # Aggiungi il messaggio dell'utente alla cronologia della chat
            st.session_state.chat_history.append((user_input, False))

            # Mostra il messaggio dell'utente
            user_message(user_input)

            # Mostra la modifica suggerita dal bot
            bot_message(modification_suggestion)

            # Mostra l'anger score (opzionale)
            st.write(f"Anger Score: {anger_score}")

    # Bottone per pubblicare i dati
    if st.button("Pubblica"):
        if user_input:            
            chat_gui.publish_data(user_input, st.session_state.anger_score)


# Esegui l'app
if __name__ == "__main__":
    main()


# import streamlit as st
# from pysentimiento import create_analyzer
# from openai import OpenAI

# client = OpenAI(api_key='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')
# from pythonosc.udp_client import SimpleUDPClient
# from transformers import MarianMTModel, MarianTokenizer

# # Configura l'API key di OpenAI correttamente

# # Funzione per mostrare i messaggi dell'utente con bordi arrotondati
# def user_message(message):
#     st.markdown(f'<div class="user-message" style="display: flex; justify-content: flex-end; padding: 5px;">'
#                 f'<div style="background-color: #196b1c; color: white; padding: 10px; border-radius: 10px;font-family:"Open Sauce One";font-size:18px; margin-bottom:10px; margin-left:20px;">{message}</div>'
#                 f'</div>', unsafe_allow_html=True)

# # Funzione per mostrare i messaggi del bot con bordi arrotondati
# def bot_message(message):
#     st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
#                 f'<div style="background-color: #074c85; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-right:20px;">{message}</div>'
#                 f'</div>', unsafe_allow_html=True)

# class ChatGUI:
#     def __init__(self):
#         self.emotion_analyzer = create_analyzer(task="emotion", lang="it")
#         self.translation_model = None  # Initialize to None, will be set later
#         self.translation_tokenizer = None  # Initialize to None, will be set later
#         self.osc_client = SimpleUDPClient("127.0.0.1", 23456)  # Initialize OSC client with appropriate IP address and port

#     def send_message(self, message):
#         emotion_result = self.analyze_emotion(message)
#         modification_suggestion, anger_score = self.suggest_modification(emotion_result, message)
#         return modification_suggestion, anger_score

#     def analyze_emotion(self, message):
#         emotion_result = self.emotion_analyzer.predict(message)
#         return emotion_result

#     def suggest_modification(self, emotion_result, user_message):
#         anger_score = emotion_result.probas.get('anger', 0.0)
#         modified_message = ""

#         # Modifica il messaggio dell'utente solo se l'Anger Score è alto
#         if anger_score > 0.7:
#             # Se l'Anger Score è alto, usa l'API per ottenere una risposta riscritta
#             prompt = f'Analizza il messaggio dell\'utente: "{user_message}". Se l\'Anger Score è superiore a 0.7, riscrivi il messaggio in modo da ridurre lo score della rabbia.'

#             # Effettua una richiesta utilizzando l'API di OpenAI
#             api_response = client.chat.completions.create(model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ])

#             # Estrai il testo modificato dalla risposta API
#             modified_message = api_response.choices[0].message.content.strip()
        
#         self.message_color = self.get_color_from_anger_score(anger_score)
#         self.color_name = self.get_color_name_from_anger_score(anger_score)

#         # Aggiungi il colore al messaggio modificato
#         modified_message = f"Il tuo messaggio verrà visualizzato in: {self.color_name}. {modified_message}"

#         return modified_message, anger_score

#     def translate_to_italian(self, text):
#         inputs = self.translation_tokenizer(text, return_tensors="pt")
#         outputs = self.translation_model.generate(**inputs)
#         translated_text = self.translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return translated_text[0]

#     def publish_data(self, user_message, anger_score, publish_button_pressed=False):
#         # Converti l'Anger Score in un numero
#         anger_score_as_number = float(anger_score)

#         # Invia il messaggio dell'utente a TouchDesigner tramite OSC
#         self.osc_client.send_message("/user_message", [user_message, anger_score_as_number])
#         st.write(f"Messaggio dell'utente pubblicato: {user_message}")

#     def get_color_name_from_anger_score(self, anger_score):
#         # Definisci il colore in base all'Anger Score
#         if 0 <= anger_score <= 0.33:
#             return "blu"
#         elif 0.33 < anger_score <= 0.66:
#             return "grigio"
#         else:
#             return "arancio"

#     def get_color_from_anger_score(self, anger_score):
#         # Definisci il colore in base all'Anger Score
#         if 0 <= anger_score <= 0.33:
#             return "r=0 g=1 b=0"  # verde
#         elif 0.33 < anger_score <= 0.66:
#             return "r=0.5 g=0.5 b=0.5"  # Grigio
#         else:
#             return "r=1 g=0 b=0"  # rosso


# # Definisci l'app principale di Streamlit
# def main():
#     st.title("Sentiment Analysis Chat")

#     # Inizializza la cronologia della chat
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     chat_gui = ChatGUI()

#     # Inizializza anger_score nello stato della sessione
#     if "anger_score" not in st.session_state:
#         st.session_state.anger_score = 0.0

#     # Campo di input per l'utente per inserire un messaggio
#     user_input = st.text_input("Inserisci il messaggio:")

#     # Bottone per inviare il messaggio dell'utente
#     if st.button("Analizza"):
#         if user_input:
#             modification_suggestion, anger_score = chat_gui.send_message(user_input)

#             # Memorizza l'anger_score nello stato della sessione
#             st.session_state.anger_score = anger_score

#             # Aggiungi il messaggio dell'utente alla cronologia della chat
#             st.session_state.chat_history.append((user_input, False))

#             # Mostra il messaggio dell'utente
#             user_message(user_input)

#             # Mostra la modifica suggerita dal bot
#             bot_message(modification_suggestion)

#             # Mostra l'anger score (opzionale)
#             st.write(f"Anger Score: {anger_score}")

#     # Bottone per pubblicare i dati
#     if st.button("Pubblica"):
#         if user_input:            
#             chat_gui.publish_data(user_input, st.session_state.anger_score)


# # Esegui l'app
# if __name__ == "__main__":
#     main()