# DACON_Classification_of_Motion
운동 동작 분류 AI 경진대회
<br><br>
### 딥러닝 모델
|File 명|설명|Log Loss값|비고|
|-----|-------|:-----:|----|
|[21.02.11][1] Baseline+Deep LSTM| 1개의 LSTM 층(BASELINE)(baseline_submission.csv)|2.4148||
|[21.02.11][1] Baseline+Deep LSTM| 4개의 LSTM 층을 쌓음(deepLSTM_submission.csv)|1.739|단일층보단 나은 성능|
|[21.02.11][2] GRU|4개의 GRU층을 쌓음(deepGRU_submission.csv)|2.289|같은 층 LSTM이 더 나은 성능|
|[21.02.11][3] GRU_ES| 4개의 GRU 층 + Early Stopping 기능 추가(deepGRU_es_submission.csv)|1.8980|Early Stopping이 안한 것보다 나은 성능|
|[21.02.11][4] DeepLSTM_ES| 4개의 LSTM 층 + Early Stopping 기능 추가(deepLSTM_es_submission.csv)|1.7265|총 epoch 33번 진행, deeper 진행(epochs:33) 1.879|
|[21.02.11][4] DeepLSTM_ES| 5개의 LSTM 층 + Early Stopping 기능 추가(deeper_lstm_submission.csv)|1.879||
|[21.02.11][4] DeepLSTM_ES| 3개의 LSTM 층 + Early Stopping 기능 추가(deep__lstm_3_submission.csv)|1.8394||
|[21.02.11][5] DeepLSTM_BN| 3개의 LSTM 층 + Early Stopping + BatchNormalization 추가(deepLSTM_es_bn_submission.csv)|3.392|patience = 20, epochs = 300 -> logloss : 더 별로|
|[21.02.11][6] DeepLSTM_DropOut|3개의 LSTM 층 + Early Stopping + Dropout 추가(deepLSTM_es_drop_submission.csv)|1.8322|Batchnormalization보단 dropout, but 안해주는 것이 나음|
|[21.02.12][9] Bi-LSTM|biLSTM 3개의 층 + Early Stopping(bi_LSTM_submission.csv)|2.239| 더 안좋아짐 ㅇㅅㅇ|
|[21.02.12][9] Bi-LSTM|biLSTM 4개의 층 + Early Stopping(bi_LSTM_submission.csv)|2.127|3개의 층 보다는 좋은데 여전히 별로임|
|[21.02.13][10] CNN_LSTM|Conv1D(1개의 층) + LSTM(1개의 층) + Early Stopping(cnn_lstm.csv)|**1.2267**|Conv1D + LSTM이 좋은 성능을 보임|
|[21.02.13][10] CNN_LSTM|Conv1D(2개의 층) + LSTM(1개의 층) + Early Stopping(cnn_lstm_v2.csv)|||

### 머신러닝 모델
|File 명|설명|Log Loss값|비고|
|-----|-------|:-----:|----|
|[21.02.11][7] RandomForest_baseline|RandomForest 사용(baseline_rf)|1.50886|다른 딥러닝들 보다 좋은 성능을 보임|
|[21.02.11][8] GradientBoostingClassifier|GradientBoostingClassifier사용(GradientBoostingClassifier)|**1.3918**|RandomForest보다 나은 성능|


- 다른 딥러닝 모델 보다 Random Forest가 좋은 성능을 보임
- **Early Stopping** 을 사용했을때 훨씬 좋은 성능을 보인다. > 높은 epoch은 성능을 저하시킨다.
- 같은 층일 때 LSTM이 GRU보다 나은 성능
- 현재는 LSTM에 Early Stopping 만 적용했을때 가장 높은 성능을 보인다.
<br><br>
### 시도
#### 모델
- Bidirectional(LSTM) + Bidirectional(GRU)
- CNN + Bi-LSTM
- Bi-GRU
- Transformer
- FB- Prophet
- ARIMA
- SARIMA
- Logistic-Reg
#### 추가
- 가중치 초기화 (kernel_initializer = 'he_normal')
- C -> BN -> Dropout
- optimizer => Nadam, RMSprop, adam
- optimizer => weight_decay
<br><br>
**여러가지 Single Model에서 썻던 구조와 에폭 수, 배치 사이즈를 그대로 가지고와서 model averaging 가능**

```python
num_models=15
model_list=[]

for i in tqdm(range(num_models)):
    model = build_fn()
    model.fit(Xtrain_dbmel, Ytrain, epochs=187, batch_size=16)
    model_list.append(model)ej
    model.save(f"model_{i}.h5")
```
```python
# 저장된 모델 불러오기

models = []
for i in tqdm(range(0, 15)):
    model_name = f"model_{i}.h5"
    models.append(keras.models.load_model(model_name, custom_objects={'mish' : mish}))
print(f"{len(models)} models reloaded")
```

```python
preds = np.zeros(shape=submission.shape)
train_preds = np.zeros(shape = Ytrain.shape)

train_preds_list=[]
test_preds_list=[]
score_list=[]

for model, i in zip(models, range(len(models))):
    a = model.predict(Xtrain_dbmel)
    b = model.predict(Xtest_dbmel)
    eval_score = eval_kldiv(Ytrain, a)
    
    print(f"Model {i+1} Evaluation Score : {eval_score}")
    train_preds = train_preds + a
    preds = preds + b
    
    train_preds_list.append(a)
    test_preds_list.append(b)
    score_list.append(eval_score)
    
train_preds = train_preds / len(models)
preds = preds / len(models)
print(f"\nMean Predictions Evaluation Score : {eval_kldiv(Ytrain, train_preds)}")
simple_average = pd.DataFrame(preds, index=submission.index, columns=submission.columns)
simple_average.to_csv('15 Average Ensemble model.csv')
simple_average.head(10)
```
