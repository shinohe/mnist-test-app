# exampleFun 概要
## 手書き文字認識（数字）
1. andAnalizerの環境構築を終わらせる
1. `python minist.py`  
   を実行し学習させる。（学習データはsaverフォルダ下に作成される）
1. `python ministServer.py`  
    を実行し手書き文字認識サーバーを立ち上げる。
1. [http;//localhost:5000/](http;//localhost:5000/)にアクセスする。
1. 書き込めるフィールドに数字を書くとサーバーにデータが送信され、判定結果が表示される。

※regressionは簡単なロジスティック回帰実装興味あれば
