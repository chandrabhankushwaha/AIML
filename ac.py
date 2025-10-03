import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow import keras

from keras.models import Model
from keras.layers import Input,LSTM,Dense

char_set = list(" abcdefghijklmnopqrstuvwxyz0123456789")
char2int = { char_set[x]:x for x in range(len(char_set)) }
int2char = { char2int[x]:x for x in char_set }

count = len(char_set)
codes = ["\t","\n",'#']
for i in range(len(codes)):
    code = codes[i]
    char2int[code]=count
    int2char[count]=code
    count+=1


max_enc_len=15
max_dec_len=15

batch_size = 128
epochs = 1000
latent_dim = 256

num_enc_tokens = len(char_set)
num_dec_tokens = len(char_set) + 2 # includes \n \t

model = keras.models.load_model('s2s.h5')
encoder_model=keras.models.load_model('encoder.h5')
decoder_model=keras.models.load_model('decoder.h5')

def train():
    wellstr = ['well Faro',
'Wells Farbo',
'Wells Farno',
'Wells Farfo',
'Wells Farro',
'Wells Farto',
'Wells Faryo',
'Wells Farvo',
'Wells Fatgo',
'Wells Fafgo',
'Wells Faggo',
'Wells Faego',
'Wells Fadgo',
'Wells Fsrgo',
'Wells Fzrgo',
'Wells Fxrgo',
'Wells Fqrgo',
'Wells Fwrgo',
'Welld Fargo',
'Wellx Fargo',
'Wellc Fargo',
'Wella Fargo',
'Wellq Fargo',
'Wellw Fargo',
'Welle Fargo',
'Wellz Fargo',
'Wekls Fargo',
'Weils Fargo',
'Weols Fargo',
'Wepls Fargo',
'Wekls Fargo',
'Weils Fargo',
'Weols Fargo',
'Wepls Fargo',
'Wrlls Fargo',
'Wdlls Fargo',
'Wflls Fargo',
'Wwlls Fargo',
'Wslls Fargo',
'Wells Farg',
'Wells Faro',
'Wells Fago',
'Wells Frgo',
'Wells argo',
'WellsFargo',
'Well Fargo',
'Wels Fargo',
'Wels Fargo',
'Wlls Fargo',
'ells Fargo',
'Wells Fargoo',
'Wells Farggo',
'Wells Farrgo',
'Wells Faargo',
'Wells FFargo',
'Wells  Fargo',
'Wellss Fargo',
'Wellls Fargo',
'Wellls Fargo',
'Weells Fargo',
'WWells Fargo',
'Wells Farog',
'Wells Fagro',
'Wells Frago',
'Wells aFrgo',
'WellsF argo',
'Well sFargo',
'Welsl Fargo',
'Wells Fargo',
'Wlels Fargo',
'eWlls Fargo']


    jpmcstr = ['JPM',
    'JPC',
    'JMC',
    'PMC',
    'JPMCC',
    'JPMMC',
    'JPPMC',
    'JJPMC',
    'JPCM',
    'JMPC',
    'PJMC',]

    capstr = ['Capital On',
    'Capital Oe',
    'Capital ne',
    'CapitalOne',
    'Capita One',
    'Cpital One',
    'Capial One',
    'Captal One',
    'Caital One',
    'Cpital One',
    'apital One',
    'Capital Onee',
    'Capital Onne',
    'Capital OOne',
    'Capital  One',
    'Capitall One',
    'Capitaal One',
    'Capittal One',
    'Capiital One',
    'Cappital One',
    'Caapital One',
    'CCapital One',
    'Capital Oen',
    'Capital nOe',
    'CapitalO ne',
    'Capita lOne',
    'Capitla One',
    'Capiatl One',
    'Captial One',
    'Caiptal One',
    'Cpaital One',
    'aCpital One',]

    goldstr = ['Goldman Sach',
    'Goldman Sacs',
    'Goldman Sahs',
    'Goldmn Sachs',
    'Goldman achs',
    'GoldmanSachs',
    'Goldma Sachs',
    'Goldmn Sachs',
    'Goldan Sachs',
    'Golman Sachs',
    'Godman Sachs',
    'Gldman Sachs',
    'oldman Sachs',
    'Goldman Sachss',
    'Goldman Sachhs',
    'Goldman Sacchs',
    'Goldman Saachs',
    'Goldman SSachs',
    'Goldman  Sachs',
    'Goldmann Sachs',
    'Goldmaan Sachs',
    'Goldmman Sachs',
    'Golddman Sachs',
    'Golldman Sachs',
    'Gooldman Sachs',
    'GGoldman Sachs',
    'Goldman Sacsh',
    'Goldman Sahcs',
    'Goldman Scahs',
    'Goldman aSchs',
    'GoldmanS achs',
    'Goldma nSachs',
    'Goldmna Sachs',
    'Goldamn Sachs',
    'Golmdan Sachs',
    'Godlman Sachs',
    'Glodman Sachs',
    'oGldman Sachs',]

    input_texts=[]
    target_texts=[]

    for i in range(len(wellstr)):
        input_texts.append(wellstr[i].lower())
        target_texts.append('\t'+"wells fargo"+'\n')

    for i in range(len(jpmcstr)):
        input_texts.append(jpmcstr[i].lower())
        target_texts.append('\t'+"jpmc"+'\n')

    for i in range(len(capstr)):
        input_texts.append(capstr[i].lower())
        target_texts.append('\t'+"capital one"+'\n')

    for i in range(len(goldstr)):
        input_texts.append(goldstr[i].lower())
        target_texts.append('\t'+"goldman sachs"+'\n')


    max_enc_len = max([len(x) for x in input_texts])
    max_dec_len = max([len(x) for x in target_texts])
    print("Max Enc Len:",max_enc_len)
    print("Max Dec Len:",max_dec_len)


    num_samples = len(input_texts)
    encoder_input_data = np.zeros( (num_samples , max_enc_len , len(char_set)),dtype='float32' )
    decoder_input_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )
    decoder_target_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )
    print("CREATED ZERO VECTORS")


    num_enc_tokens = len(char_set)
    num_dec_tokens = len(char_set) + 2 # includes \n \t
    encoder_inputs = Input(shape=(None,num_enc_tokens))
    encoder = LSTM(latent_dim,return_state=True)
    encoder_outputs , state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h,state_c]


    decoder_inputs = Input(shape=(None,num_dec_tokens))
    decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
    decoder_ouputs,_,_ = decoder_lstm(decoder_inputs,initial_state = encoder_states)

    decoder_dense = Dense(num_dec_tokens, activation='softmax')
    decoder_ouputs = decoder_dense(decoder_ouputs)

    model = Model([encoder_inputs,decoder_inputs],decoder_ouputs)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

    for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
        for t,char in enumerate(input_text): #Wells Farmgo
            # print(char)
            # print(i, t, char2int[char])
            encoder_input_data[ i , t , char2int[char] ] = 1
        for t,char in enumerate(target_text):
            decoder_input_data[ i, t , char2int[char] ] = 1
            if t > 0 :
                decoder_target_data[ i , t-1 , char2int[char] ] = 1

    h=model.fit([encoder_input_data,decoder_input_data],decoder_target_data
         ,epochs = epochs,
          batch_size = batch_size,
          validation_split = 0.2
         )
    
    model.save('s2s.h5')

    encoder_model = Model(encoder_inputs,encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]
    decoder_outputs,state_h,state_c = decoder_lstm(
            decoder_inputs,initial_state = decoder_states_inputs
    )
    decoder_states = [state_h,state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    encoder_model.save('encoder.h5')
    decoder_model.save('decoder.h5')

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    # print(input_seq.shape())
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_dec_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, char2int['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int2char[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_dec_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_dec_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def custom_input(ip_text):
    test_encoded_data = np.zeros((1, max_enc_len , len(char_set)),dtype='float32')
    # ip_text="jpnc"
    # correct_text = ""
    for t,char in enumerate(ip_text): #Wells Farmgo
        test_encoded_data[0, t , char2int[char] ] = 1

    decoded_sentence = decode_sequence(test_encoded_data)
    # print('-')
    # print('Wrong sentence:', ip_text)
    # print('Corrected sentence:', decoded_sentence)
    # print('Ground Truth:'+correct_text)
    return decoded_sentence

def main():
    print(custom_input("jpnc"))
    print(custom_input("wellm farg"))


if __name__ == "__main__":
    main()