import streamlit as st
from pysentimiento import create_analyzer
from openai import OpenAI


from openai import OpenAI

client = OpenAI(
  organization='org-V05ALogBi5zwQYpqamNVpjRm',
  project='$proj_fd7QAScm10pfoThPhZntwoI3',
  api_key='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',
)

from pythonosc.udp_client import SimpleUDPClient
from transformers import MarianMTModel, MarianTokenizer


# Function to display user messages with rounded rectangle borders
def user_message(message):
    st.markdown(f'<div class="user-message" style="display: flex; justify-content: flex-end; padding: 5px;">'
                f'<div style="background-color: #196b1c; color: white; padding: 10px; border-radius: 10px;font-family:"Open Sauce One";font-size:18px; margin-bottom:10px; margin-left:20px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)

# Function to display bot messages with rounded rectangle borders
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
            prompt = f'{"Assistant:"} Analizza il messaggio dell\'utente: {user_message}. Se l\'Anger Score è superiore a 0.7, riscrivi il messaggio in modo da ridurre lo score della rabbia.'
            api_response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])

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

        # if publish_button_pressed:
        #     # Invia il segnale che il tasto "Pubblica" è stato premuto
        #     self.osc_client.send_message("/publish_button", 1)

        # # Invia l'Anger Score come numero a TouchDesigner tramite OSC
        # self.osc_client.send_message("/anger_score", anger_score_as_number)
        # st.write(f"Anger Score pubblicato: {anger_score_as_number}")

        # Invia l'informazione del colore a TouchDesigner
        # self.osc_client.send_message("/message_color", self.message_color)
        
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


# Define the main Streamlit app
def main():
    st.title("Sentiment Analysis Chat")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_gui = ChatGUI()

    # Initialize anger_score in session state
    if "anger_score" not in st.session_state:
        st.session_state.anger_score = 0.0


    # Input field for user to enter a message
    user_input = st.text_input("Inserisci il messaggio:")
    # Button to send the user's message
    if st.button("Analizza"):
        if user_input:
            modification_suggestion, anger_score = chat_gui.send_message(user_input)

            # Store anger_score in session state
            st.session_state.anger_score = anger_score

            # Add the user's message to the chat history
            st.session_state.chat_history.append((user_input, False))

            # Display the user's message
            user_message(user_input)

            # Display the modification suggestion from the bot
            bot_message(modification_suggestion)

            # Display anger score (optional)
            st.write(f"Anger Score: {anger_score}")
        
        # Button to publish data
    if st.button("Pubblica"):
         if user_input:            
            chat_gui.publish_data(user_input, st.session_state.anger_score)


# Run the app
if __name__ == "__main__":
    main()
