import pandas as pd

df_temp = pd.read_csv('./data/puri_train_76253.csv')

df_temp = df_temp.iloc[:10000]
df_temp.iloc[12]

df_result = pd.DataFrame(columns=['text', 'label'])
df_result['text'] = df_temp.comment
df_result['label'] = df_temp.label

df_result['text'] = df_result['text'].apply(lambda x: str(x).replace("\t", "").replace("\n", "").replace("\r", ""))
df_result['text'].iloc[12]

########## gpt 생성 data
df_temp2 = pd.read_csv('./data/final_gpt2_generate_sents.csv')
df_temp2['text'] = df_temp2['text'].apply(lambda x: str(x).replace("\t", "").replace("\n", "").replace("\r", ""))
df_temp2

######## 워마드 관련 data
df_temp3 = pd.read_csv('./data/hate_speech_data_fixed.csv')
df_temp3['text'] = df_temp3['text'].apply(lambda x: str(x).replace("\t", "").replace("\n", "").replace("\r", ""))
df_temp3

df_result = pd.concat([df_result, df_temp2, df_temp3]).reset_index(drop=True)

df_result.to_csv('./data/puri/puri_train.txt', sep="\t", index=False, header=None)