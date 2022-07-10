# PyTorch Lib

# 概要
PyTorchで使えるコード例です．
* [PyTorch-qat](#pytorch-qat)
* [PyTorch-qml](#pytorch-qml)
* [PyTorch-xnor](#pytorch-xnor)

# PyTorch-qat
量子化を考慮しながら学習するQuantization Aware Training (QAT) を行うためのコードです．
これにより，推論時の演算を固定小数点数で行うことが可能となります．  
PytorchにはもとからQATが実装されていますが，量子化誤差をハードウェア効率の良い，2の累乗に制限する機能を新たに追加しています (myobserver.py, myfake_quantize.py)．  
また，公式のドキュメントには，8bitの固定小数点数に量子化する例しか存在しないため，8bitより低い精度でmnistを学習する例をtestコードとして示しています．  

# PyTorch-qml
量子計算を取り入れた量子機械学習をqiskitとともに行うためのコードです．

# PyTorch-xnor
XNOR-Netの実装例です(未完成)．