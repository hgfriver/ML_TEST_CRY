
311581019 何立平 智能所

## 資料分析：

* Correlation between all Features and the Price 
 ![](https://i.imgur.com/zfXU3jE.jpg)

 ![](https://i.imgur.com/C99GD0K.png)

由上圖Correlation分析可知, bathroom、sqft_living、grade、sqft_above、sqft_living15等feature，都是correlation大於0.5的feature。代表這個dataset的feature與price是有一定程度的關聯性。

## 做法：
* 原本我是使用一般自行疊加的MLP做為model，發現效果不理想，最後使用了XGBoost作為model。

* 以下是XGBoost的介紹：
XGboost 全名為 eXtreme Gradient Boosting。此機器學習模型是以 Gradient Boosting 為基礎下去實作，並添加一些新的技巧。它可以說是結合 Bagging 和 Boosting 的優點。XGboost 保有 Gradient Boosting 的做法，每一棵樹是互相關聯的，目標是希望後面生成的樹能夠修正前面一棵樹犯錯的地方。此外 XGboost 是採用特徵隨機採樣的技巧，和隨機森林一樣在生成每一棵樹的時候隨機抽取特徵，因此在每棵樹的生成中並不會每一次都拿全部的特徵參與決策。此外為了讓模型過於複雜，XGboost 在目標函數添加了標準化。因為模型在訓練時為了擬合訓練資料，會產生很多高次項的函數，但反而容易被雜訊干擾導致過度擬合。因此 L1/L2 Regularization 目的是讓損失函數更佳平滑，且抗雜訊干擾能力更大。最後 XGboost 還用到了一階導數和二階導數來生成下一棵樹。其中 Gradient 就是所謂的一階導數，而 Hessian 即為二階導數。

* XGBoost 優點：
XGBoost 除了可以做分類也能進行迴歸連續性數值的預測，而且效果通常都不差。並透過 Boosting 技巧將許多弱決策樹集成在一起形成一個強的預測模型。
•	利用了二階梯度來對節點進行劃分
•	利用局部近似算法對分裂節點進行優化
•	在損失函數中加入了 L1/L2 項，控制模型的複雜度
•	提供 GPU 平行化運算






* 參考資料：https://ithelp.ithome.com.tw/articles/10273094





## 程式寫法：

Step1: include library dependencies
Step2: feature engineering 
Step3: model fitting and prediction
Step4: result



## 結果分析：

* 過程中我調整了一些超參數，讓準確率提升，超參數包含：
    * n_estimators=1800,
    * learning_rate=0.04,
    * max_depth=7

* 要注要model size不能太大，可能會overfitting



## 檢討與改進：
* 可以比較多種model的performance，進而找到更適合的model，或是更好的超參數組合。
* 可以試試超參數搜尋的技巧和工具，讓超參數的調整更有效率。

