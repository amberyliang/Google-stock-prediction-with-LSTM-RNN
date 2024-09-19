# Google-stock-prediction-with-LSTM-RNN
Introduction Video：https://youtu.be/vMjsQQHLtBc
Google Stock Prediction報告補充內容

模型正則化技術
	Dropout：每個LSTM層和全連接層之間都應用了Dropout層，這有助於防止模型過度擬合。
  
資料預處理
•	資料正規劃Data Normalization
o	MinMaxScaler()用於縮放（歸一化）特徵，使得特徵的最小值和最大值落在一個預定的範圍內，通常是 [0, 1]。

•	資料集分割Data Splitting
o	20%測試集
o	80%訓練集
•	序列生成Sequence Generation
o	形成時間序列數據
o	將一個包含多個特徵的 DataFrame 轉換為適合用於訓練和測試序列模型的格式(NumPy array)。
這在金融市場、物聯網、語音識別等領域的預測和分析中非常有用。
•	時間序列預測： 在金融市場中，我們通常需要根據過去的一段時間內的市場數據（如股票價格、交易量等）來預測未來的價格走勢。該代碼可以幫助生成用於訓練模型的序列數據和對應的標籤，使得模型能夠學習這些時間序列數據的模式和趨勢。







LSTM模型介紹
這個 LSTM 模型是為了預測金融時間序列數據而設計的，包含了三層 LSTM 層以及三個 Dropout 層，最終通過一個 Dense 層輸出預測結果。

1.第一層 LSTM 層：
	LSTM(units=50, return_sequences=True, input_shape=(50, 5))
	這一層有 50 個隱藏單元，`return_sequences=True` 表示這一層會返回整個輸出序列而不只是最後一個隱藏狀態。這對於構建多層 LSTM 很重要。
	`input_shape=(50, 5)` 指定了輸入數據的形狀，其中 50 是時間步長（序列長度），5 是特徵數量（如開盤價、高價、低價、成交量、收盤價）。
2. 第一層 Dropout 層：
	Dropout(0.3)
	Dropout 是防止過擬合的一種正則化技術。`0.3` 表示有 30% 的神經元會在每次訓練迭代中隨機丟棄。
3. 第二層 LSTM 層：
	LSTM(units=50, return_sequences=True)
	這一層同樣有 50 個隱藏單元，並返回整個輸出序列。
4. 第二層 Dropout 層：
	Dropout(0.3)
	再次使用 Dropout 來防止過擬合。
5. 第三層 LSTM 層：
	LSTM(units=50)
	這一層有 50 個隱藏單元，但不返回序列，而是返回最終的隱藏狀態（也就是最後一個時間步的輸出），這通常用於最終的 Dense 層輸出預測值。
6. 第三層 Dropout 層：
	Dropout(0.3)
	再次使用 Dropout 來防止過擬合。
7. 輸出層：
	Dense(units=5)
	最後是一個全連接層（Dense 層），輸出單位數為 5，對應於我們要預測的五個特徵（開盤價、高價、低價、成交量、收盤價）。



RNN模型介紹
•	SimpleRNN(50, input_shape=(train_sequences.shape[1], train_sequences.shape[2]), return_sequences=False)
•	這一層包含 50 個隱藏單元（神經元），input_shape=(train_sequences.shape[1], train_sequences.shape[2]) 指定了輸入數據的形狀，其中 train_sequences.shape[1] 是時間步長（序列長度），train_sequences.shape[2] 是特徵數量（如開盤價、高價、低價、成交量、收盤價）。
•	return_sequences=False 表示這一層只返回最後一個隱藏狀態，而不是整個序列。這意味著後續層只會接收到一個時間步的輸出。
•  第一個 Dropout 層：
•	Dropout(0.3)
•	Dropout 層是一種防止過擬合的正則化技術。這裡的 0.3 表示在每次訓練迭代中，30% 的神經元將隨機被忽略，以增加模型的泛化能力。
•  全連接層（Dense Layer）：
•	Dense(50, activation='relu')
•	這是一個全連接層，包含 50 個神經元，使用 ReLU（Rectified Linear Unit）激活函數。這層將輸入映射到一個更高維度的特徵空間，有助於捕捉數據中的非線性關係。
•  第二個 Dropout 層：
•	Dropout(0.3)
•	再次使用 Dropout 層來防止過擬合，這一層的設置同樣是隨機忽略 30% 的神經元。
•  輸出層：
•	Dense(5)
•	最後是一個全連接層，包含 5 個神經元，對應於我們要預測的五個特徵（開盤價、高價、低價、成交量、收盤價）。這一層沒有激活函數，因為我們需要預測的是連續值。
模型訓練:
兩個模型採用相同訓練參數
•	訓練次數（epochs=200）：模型將在整個訓練數據集上訓練 200 次。
•	批次大小（batch_size=32）：每次更新模型參數時使用 32 條數據。
•	驗證分割（validation_split=0.2）：20% 的訓練數據將用於驗證模型的性能。

模型評估:
	均方誤差（Mean squared error，MSE）：損失越低，模型的預測性能越好
	平均絕對誤差(Mean absolute error，MAE)：MAE 越低，模型的預測值與實際值之間的誤差越小。
	決定係數（R²）：R² 分數表示模型預測值與實際值之間的擬合程度。它的取值範圍在 [0, 1] 之間，越接近 1 表示模型的擬合效果越好。
	LSTM模型有較低的損失值與平均絕對誤差
	MSE	MAE	R^2
RNN	0.02022913098335266	0.11299212276935577	0.706288844954938

LSTM	0.0006351796328090131	0.020060734823346138	0.976651943946694







模型預測結果：
	RNN
 
	LSTM
 

未來十天預測結果:
兩個模型預測趨勢一致
	RNN
  














	LSTM
  
預測比較:
1.	evaluate() 函數：對測試數據進行評估，計算測試損失（Test Loss）和平均絕對誤差（MAE）。
•	rnn_model.evaluate(test_sequences, test_labels) 返回 RNN 模型在測試數據上的損失和 MAE。
•	lstm_model.evaluate(test_sequences, test_labels) 返回 LSTM 模型在測試數據上的損失和 MAE。
2.	predict() 函數：對測試數據進行預測，返回模型的預測值。
•	rnn_model.predict(test_sequences) 返回 RNN 模型對測試數據的預測值。
•	lstm_model.predict(test_sequences) 返回 LSTM 模型對測試數據的預測值。
3.	實際收盤價：從測試標籤中提取實際的收盤價。
•	test_close_actual = test_labels[:, 0] 提取測試標籤中的收盤價（假設收盤價在第 0 列）。
4.	預測收盤價：從預測結果中提取 RNN 和 LSTM 模型的預測收盤價。
•	rnn_test_close_pred = rnn_test_predictions[:, 0] 提取 RNN 模型的預測收盤價。
•	lstm_test_close_pred = lstm_test_predictions[:, 0] 提取 LSTM 模型的預測收盤價。
•	圖表設定：創建一個 12x6 的圖表，用於顯示預測結果。
•	繪製實際收盤價：將實際的收盤價繪製在圖表上，標籤為 Actual Close。
•	繪製 RNN 預測收盤價：將 RNN 模型預測的收盤價繪製在圖表上，標籤為 RNN Predicted Close。
•	繪製 LSTM 預測收盤價：將 LSTM 模型預測的收盤價繪製在圖表上，標籤為 LSTM Predicted Close。
•	設定標籤和標題：
o	橫軸標籤為 Time Step，表示時間步長。
o	縱軸標籤為 Close Price，表示收盤價。
o	圖表標題為 Close Predictions on Test Data，表示測試數據上的收盤價預測。
5.	LSTM模型相較RNN模型有較好的預測結果
 
總結:
1.	LSTM和RNN都是適合做為預測序列資料的模型
2.	LSTM相較於RNN有更好的預測效果，可能是因為LSTM 能夠更有效地捕捉並處理長期依賴性問題。LSTM 通過其特殊的結構和門控機制，有效地克服了 RNN 在長期依賴性問題上的不足，使其在處理複雜的序列數據和長時間跨度的預測任務時表現更好。 
3.	在模型訓練中，我學到要提高預測準確有以下方法：
•	增加 LSTM 層數：根據數據集的複雜性，可以考慮增加 LSTM 層數以捕捉更多的時間依賴關係。
•	調整 Dropout 率：根據訓練過程中的過擬合情況，適當調整 Dropout 率。
•	調整隱藏單元數量：根據數據的特徵和模型的表現，適當調整每層 LSTM 的隱藏單元數量。

參考資料：
https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_neural_networks_work.html

資料視覺化介紹
模型介紹

