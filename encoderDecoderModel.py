# Step 1: Import Libraries and Load dataset
import numpy as np, pandas as pd, string
from string import digits
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lines = pd.read_csv("/content/Hindi_English_Truncated_Corpus.csv", encoding='utf-8')
lines = lines[lines['source'] == 'ted'][['english_sentence', 'hindi_sentence']].dropna().drop_duplicates()
lines = lines.sample(n=25000, random_state=42)

# Step 2: Text Cleaning
def clean_text(text):
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = text.translate(str.maketrans('', '', digits))
    return text.strip().lower()

lines['english_sentence'] = lines['english_sentence'].apply(clean_text)
lines['hindi_sentence'] = lines['hindi_sentence'].apply(clean_text)
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: 'start_ ' + x + ' _end')


# Step 4: Tokenization
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(lines['english_sentence'])
eng_seq = eng_tokenizer.texts_to_sequences(lines['english_sentence'])

hin_tokenizer = Tokenizer(filters='')
hin_tokenizer.fit_on_texts(lines['hindi_sentence'])
hin_seq = hin_tokenizer.texts_to_sequences(lines['hindi_sentence'])


# Step 5: Padding
max_eng_len = max(len(seq) for seq in eng_seq)
max_hin_len = max(len(seq) for seq in hin_seq)

encoder_input = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
decoder_input = pad_sequences(hin_seq, maxlen=max_hin_len, padding='post')

decoder_target = np.zeros((decoder_input.shape[0], decoder_input.shape[1], 1))
decoder_target[:, 0:-1, 0] = decoder_input[:, 1:]


# Step 6: Define Model Architecture
# Encoder:
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(eng_vocab_size, latent_dim)(encoder_inputs)
enc_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder:
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(hin_vocab_size, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(hin_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Step 7. Compile and Train
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.fit([encoder_input, decoder_input], decoder_target, batch_size=64, epochs=20, validation_split=0.2)


# Step 8: Inference Models
# Encoder Inference: Returns hidden/cell states given an English sentence.
encoder_model_inf = Model(encoder_inputs, encoder_states)

# Decoder Inference:
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_inf_emb = dec_emb_layer(decoder_inputs)
dec_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_inf_emb, initial_state=decoder_states_inputs)
decoder_outputs_inf = decoder_dense(dec_outputs_inf)
decoder_model_inf = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs_inf, state_h_inf, state_c_inf])

# Step 9: Reverse Lookup
reverse_eng = {v: k for k, v in eng_tokenizer.word_index.items()}
reverse_hin = {v: k for k, v in hin_tokenizer.word_index.items()}

# Step 10: Translate Function
def translate(sentence):
    sentence = clean_text(sentence)
    seq = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    states = encoder_model_inf.predict(padded)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin_tokenizer.word_index['start_']

    decoded = []
    while True:
        output, h, c = decoder_model_inf.predict([target_seq] + states)
        token_index = np.argmax(output[0, -1, :])
        word = reverse_hin.get(token_index, '')

        if word == '_end' or len(decoded) >= max_hin_len:
            break

        decoded.append(word)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = token_index
        states = [h, c]

    return ' '.join(decoded)


print("English: And")
print("Hindi:", translate("And"))