# Fireworks Algorithm(FWA)のJavaコード
FWAの一部をJava言語によって実装しました。  
実装したFWAは以下の二つです。  
1. Enhanced Fireworks Algorithms(EFWA)  
2. A Self-Adaptive Fireworks Algorithms(SaFWA)

比較検証用にPSOのコードも付属しています。  
ちなみにこのPSOのコードは因子収縮法を用いた係数で実装されているので精度が段違いで良いです。

# 実行方法
以下に手順を記していきます。  
1. ルートディレクトリに移動
2. javacで以下のようにコンパイルする。今回はSaFWAの例です
```bash
javac -d bin src/algorithms/safwa/*.java src/optimization/benchmarks/*.java src/optimization/OptimizationProblem.java src/utils/StatisticsUtils.java
```
3. `bin`フォルダに作成されたクラスファイルを以下のプロンプトで実行します。
```bash
java -cp bin efwa.EFWAApplication
```
4. 実行結果がターミナルに記載されます

# 参考文献  
noteフォルダに参考文献のPDFファイルを保存しています。  
元論文が読みたい場合はダウンロードしてください。
