# Copyright (c) 2024 Visao Robotica Imagem (VRI)
#  - Felipe Bombardelli <felipebombardelli@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

# =============================================================================
#  Header
# =============================================================================

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
#  Main
# =============================================================================

print("Carregando modelo e dados")
clf = clf_svc
clf.fit(x_train, y_train)

print("Testando")
y_pred = clf.predict(x_test)

print("Mostra o Resultado")
for i in range(len(y_pred)):
    print(y_pred[i], end=", ")
print()

print(classification_report(y_test, y_pred))
cont = 1
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        cont += 1
print(f"Total: {len(y_pred)}, Acertos: {cont} ({cont/len(y_pred)}%)")